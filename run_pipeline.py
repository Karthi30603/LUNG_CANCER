#!/usr/bin/env python3
"""
Complete Pipeline for Lung Cancer Risk Prediction with Bounding Box Detection

This script orchestrates the entire pipeline:
1. Preprocess DICOM files with RAVE
2. Run inference with trained model
3. Visualize results (risk predictions + bounding boxes)

Usage:
    python run_pipeline.py \
        --input-dir /path/to/dicom/directories \
        --checkpoint /path/to/model.ckpt \
        --config configs/inference_config.yaml \
        --output-dir results/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from preprocess_data import main as preprocess_main
from inference import LungCancerPredictor
from visualize_results import create_summary_report


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    workers: int = 4,
) -> Path:
    """Run data preprocessing step."""
    print("\n" + "="*60)
    print("STEP 1: Data Preprocessing")
    print("="*60)
    
    # Create preprocessing output directory
    preprocess_output = output_dir / "preprocessed"
    preprocess_output.mkdir(parents=True, exist_ok=True)
    
    # Build preprocessing command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "preprocess_data.py"),
        "--input-dir", str(input_dir),
        "--output-dir", str(preprocess_output),
        "--workers", str(workers),
    ]
    
    if config_path:
        cmd.extend(["--config", str(config_path)])
    
    # Run preprocessing
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError("Preprocessing failed")
    
    # Return path to inference CSV
    inference_csv = preprocess_output / "inference.csv"
    
    if not inference_csv.exists():
        raise FileNotFoundError(f"Inference CSV not found at {inference_csv}")
    
    return inference_csv


def run_inference(
    checkpoint_path: Path,
    config_path: Path,
    inference_csv: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 1,
) -> Path:
    """Run inference step."""
    print("\n" + "="*60)
    print("STEP 2: Model Inference")
    print("="*60)
    
    # Create inference output directory
    inference_output = output_dir / "inference"
    inference_output.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = LungCancerPredictor(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        device=device,
        batch_size=batch_size,
    )
    
    # Run inference
    predictions_path = inference_output / "predictions.json"
    predictor.predict_batch(str(inference_csv), str(predictions_path))
    
    return predictions_path


def run_visualization(
    predictions_path: Path,
    output_dir: Path,
    image_paths: Optional[dict] = None,
):
    """Run visualization step."""
    print("\n" + "="*60)
    print("STEP 3: Visualization")
    print("="*60)
    
    # Create visualization output directory
    vis_output = output_dir / "visualizations"
    vis_output.mkdir(parents=True, exist_ok=True)
    
    # Build visualization command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "visualize_results.py"),
        "--predictions", str(predictions_path),
        "--output-dir", str(vis_output),
    ]
    
    # Run visualization
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("Warning: Visualization had errors, but continuing...")
    
    # Create summary report
    with open(predictions_path, "r") as f:
        predictions = json.load(f)
    
    if isinstance(predictions, list):
        summary_path = vis_output / "summary_report.json"
        create_summary_report(predictions, summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for lung cancer risk prediction with bbox detection"
    )
    
    # Required arguments
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with DICOM files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML")
    parser.add_argument("--output-dir", type=str, default="./pipeline_results", help="Output directory")
    
    # Optional arguments
    parser.add_argument("--rave-config", type=str, help="RAVE config file (default: ct_chest.yaml)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for preprocessing")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization step")
    parser.add_argument("--inference-csv", type=str, help="Path to existing inference CSV (skip preprocessing)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not input_dir.exists() and not args.skip_preprocess:
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("LUNG CANCER RISK PREDICTION PIPELINE")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Step 1: Preprocessing
    if args.skip_preprocess:
        if args.inference_csv:
            inference_csv = Path(args.inference_csv)
        else:
            # Try to find existing inference CSV
            inference_csv = output_dir / "preprocessed" / "inference.csv"
            if not inference_csv.exists():
                raise FileNotFoundError(
                    "Preprocessing skipped but no inference CSV found. "
                    "Provide --inference-csv or run preprocessing."
                )
        print(f"\nSkipping preprocessing. Using existing CSV: {inference_csv}")
    else:
        rave_config = Path(args.rave_config) if args.rave_config else None
        inference_csv = run_preprocessing(input_dir, output_dir, rave_config, args.workers)
    
    # Step 2: Inference
    predictions_path = run_inference(
        checkpoint_path,
        config_path,
        inference_csv,
        output_dir,
        args.device,
        args.batch_size,
    )
    
    # Step 3: Visualization
    if not args.skip_visualization:
        run_visualization(predictions_path, output_dir)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Predictions: {predictions_path}")
    if not args.skip_visualization:
        print(f"  - Visualizations: {output_dir / 'visualizations'}")
    print("="*60)


if __name__ == "__main__":
    main()



