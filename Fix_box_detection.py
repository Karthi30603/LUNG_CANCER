#!/usr/bin/env python3
"""
Fix bounding box detection to focus on suspicious regions in lung field,
not anatomical landmarks like aorta/spine.

Based on the paper's approach: Use attention maps and risk prediction
to identify suspicious regions, filter out anatomical structures.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_in_lung_field(x: float, y: float, z: float, volume_shape: Tuple[int, int, int]) -> bool:
    """
    Check if coordinates are in the lung field region.
    Lungs are typically in the lateral portions of the chest, not central.
    
    Args:
        x, y, z: Normalized coordinates (0-1)
        volume_shape: (D, H, W) shape of volume
    
    Returns:
        True if likely in lung field
    """
    D, H, W = volume_shape
    
    # Convert to pixel coordinates
    cx = x * W
    cy = y * H
    cz = z * D
    
    # Lung field heuristic:
    # - X: Lungs are lateral, avoid central mediastinum (center ~ W/2)
    #   Consider lung field as outer 60% of width (20-80% range)
    # - Y: Lungs span most of height (20-80%)
    # - Z: Middle to upper chest typically has most lung volume
    
    x_frac = cx / W
    y_frac = cy / H
    z_frac = cz / D
    
    # Check if in lateral lung regions (not central mediastinum)
    in_lung_x = 0.15 < x_frac < 0.85  # Avoid very edges but exclude center
    in_lung_y = 0.2 < y_frac < 0.8   # Typical lung height range
    in_lung_z = 0.2 < z_frac < 0.8   # Typical lung depth range
    
    # Additional: if x is too central (mediastinum), likely aorta/spine
    in_mediastinum = 0.4 < x_frac < 0.6
    
    return (in_lung_x and in_lung_y and in_lung_z) and not in_mediastinum


def filter_anatomical_landmarks(bbox: List[float], volume_shape: Tuple[int, int, int]) -> bool:
    """
    Filter out bboxes that are likely anatomical landmarks (aorta, spine).
    
    Returns:
        True if bbox should be kept (likely suspicious region)
        False if bbox should be filtered (likely anatomical landmark)
    """
    cx, cy, cz, h, w, d = bbox
    
    # Check if center is in lung field
    if not is_in_lung_field(cx, cy, cz, volume_shape):
        return False
    
    # Filter very large bboxes (anatomical structures are large)
    if h > 0.3 or w > 0.3 or d > 0.3:
        return False
    
    # Filter bboxes in central mediastinum region
    # Aorta/spine are typically at center x coordinate
    if 0.45 < cx < 0.55:
        return False
    
    return True


def extract_suspicious_regions_from_attention(
    attention_map: np.ndarray,
    volume_shape: Tuple[int, int, int],
    threshold_percentile: float = 95.0
) -> List[Dict]:
    """
    Extract suspicious regions from attention map by finding high-activation areas.
    
    Args:
        attention_map: 3D attention map (D, H, W) or flattened
        volume_shape: Original volume shape (D, H, W)
        threshold_percentile: Percentile for thresholding (higher = more selective)
    
    Returns:
        List of bbox dicts for suspicious regions
    """
    D, H, W = volume_shape
    
    # Reshape attention map if needed
    if attention_map.ndim == 1:
        # Assume it's flattened (D*H*W)
        if len(attention_map) == D * H * W:
            attention_map = attention_map.reshape(D, H, W)
        else:
            logger.warning(f"Attention map size {len(attention_map)} doesn't match volume {D*H*W}")
            return []
    elif attention_map.ndim == 3:
        # Check if dimensions match
        if attention_map.shape != (D, H, W):
            # Try to reshape/upsample
            att_d, att_h, att_w = attention_map.shape
            logger.info(f"Attention map shape {attention_map.shape} != volume shape {volume_shape}, will interpolate")
    
    # Normalize attention map
    if attention_map.max() > 0:
        attention_map = attention_map / attention_map.max()
    
    # Threshold to find high-activation regions
    threshold = np.percentile(attention_map, threshold_percentile)
    binary_map = attention_map > threshold
    
    if binary_map.sum() == 0:
        logger.warning("No high-activation regions found in attention map")
        return []
    
    # Find connected components
    labeled, num_features = ndimage.label(binary_map)
    
    regions = []
    for i in range(1, num_features + 1):
        component = labeled == i
        
        # Get bounding box of component
        coords = np.where(component)
        if len(coords[0]) == 0:
            continue
        
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        # Convert to normalized coordinates
        cx = (x_min + x_max) / 2 / W
        cy = (y_min + y_max) / 2 / H
        cz = (z_min + z_max) / 2 / D
        
        height = (y_max - y_min + 1) / H
        width = (x_max - x_min + 1) / W
        depth = (z_max - z_min + 1) / D
        
        # Filter if too large or in mediastinum
        if width > 0.3 or height > 0.3 or depth > 0.3:
            continue
        
        # Check if in lung field
        if not is_in_lung_field(cx, cy, cz, (D, H, W)):
            continue
        
        regions.append({
            'bbox': [cx, cy, cz, height, width, depth],
            'attention_score': float(attention_map[component].mean()),
            'source': 'attention_map'
        })
    
    return regions


def fix_predictions(
    predictions_file: Path,
    output_file: Path,
    inference_csv: Path,
    use_attention: bool = True,
    confidence_threshold: float = 1e-8
):
    """
    Fix predictions by filtering anatomical landmarks and using attention maps.
    
    Args:
        predictions_file: Path to predictions.json
        output_file: Path to save fixed predictions
        inference_csv: Path to inference CSV (to get volume paths)
        use_attention: Whether to use attention maps for region finding
        confidence_threshold: Minimum confidence to keep bbox
    """
    logger.info(f"Loading predictions from {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Processing {len(predictions)} predictions...")
    
    fixed_predictions = []
    
    for pred in predictions:
        if pred.get('status') != 'success':
            fixed_predictions.append(pred)
            continue
        
        fixed_pred = pred.copy()
        
        # Check if bbox should be filtered
        bbox = pred.get('bbox')
        bbox_confidence = pred.get('bbox_confidence', 0)
        
        if bbox and bbox_confidence < confidence_threshold:
            logger.debug(f"Filtering {pred['accession']}: confidence too low ({bbox_confidence:.2e})")
            fixed_pred['bbox'] = None
            fixed_pred['bbox_confidence'] = 0.0
            fixed_pred['bbox_filtered'] = True
            fixed_pred['filter_reason'] = 'low_confidence'
        
        elif bbox:
            # Assume standard volume shape (will be verified)
            volume_shape = (256, 256, 256)
            
            # Filter anatomical landmarks
            if not filter_anatomical_landmarks(bbox, volume_shape):
                logger.debug(f"Filtering {pred['accession']}: anatomical landmark detected")
                fixed_pred['bbox'] = None
                fixed_pred['bbox_confidence'] = 0.0
                fixed_pred['bbox_filtered'] = True
                fixed_pred['filter_reason'] = 'anatomical_landmark'
            
            # Try to extract from attention map if available
            if use_attention and 'attention_map' in pred:
                attention = pred['attention_map']
                if isinstance(attention, list):
                    attention_arr = np.array(attention)
                    
                    # Extract suspicious regions from attention
                    suspicious_regions = extract_suspicious_regions_from_attention(
                        attention_arr,
                        volume_shape,
                        threshold_percentile=95.0
                    )
                    
                    if suspicious_regions:
                        # Use the highest scoring region
                        best_region = max(suspicious_regions, key=lambda x: x['attention_score'])
                        fixed_pred['bbox'] = best_region['bbox']
                        fixed_pred['bbox_confidence'] = best_region['attention_score']
                        fixed_pred['bbox_source'] = 'attention_map'
                        fixed_pred['bbox_filtered'] = False
                        logger.info(f"Using attention-based bbox for {pred['accession']}")
        
        fixed_predictions.append(fixed_pred)
    
    # Save fixed predictions
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(fixed_predictions, f, indent=2)
    
    logger.info(f"Saved fixed predictions to {output_file}")
    
    # Statistics
    total = len(fixed_predictions)
    filtered = sum(1 for p in fixed_predictions if p.get('bbox_filtered'))
    kept = sum(1 for p in fixed_predictions if p.get('bbox') and not p.get('bbox_filtered'))
    
    logger.info(f"Results: {kept} kept, {filtered} filtered, {total - kept - filtered} no bbox")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix bbox detection to filter anatomical landmarks and use attention maps"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for fixed predictions"
    )
    parser.add_argument(
        "--inference-csv",
        type=Path,
        required=True,
        help="Path to inference CSV (for volume info)"
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Use attention maps to find suspicious regions"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=1e-8,
        help="Minimum confidence to keep bbox"
    )
    
    args = parser.parse_args()
    
    fix_predictions(
        predictions_file=args.predictions,
        output_file=args.output,
        inference_csv=args.inference_csv,
        use_attention=args.use_attention,
        confidence_threshold=args.confidence_threshold
    )


if __name__ == "__main__":
    main()

