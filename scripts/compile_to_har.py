#!/usr/bin/env python3
"""
ONNX to HAR Conversion Script

This script converts ONNX models to Hailo Archive (HAR) format using the Hailo SDK.
It handles calibration data loading, model optimization, and quantization.

Usage:
    python scripts/compile_to_har.py --onnx outputs/onnx/model.onnx --output outputs/har/model.har
    
For more options:
    python scripts/compile_to_har.py --help
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
import numpy as np
from PIL import Image

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from hailo_sdk_client import ClientRunner
except ImportError:
    print("Error: Hailo SDK not found. Make sure you're using the correct virtual environment.")
    sys.exit(1)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def validate_paths(onnx_path: str, calib_dir: str, model_script: str) -> None:
    """Validate that required paths exist."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    if not os.path.exists(calib_dir):
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
        
    # Handle relative paths for model script
    if not os.path.isabs(model_script):
        # Get script directory and go up one level to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_script = os.path.join(project_root, model_script)
    
    if not os.path.exists(model_script):
        raise FileNotFoundError(f"Model script not found: {model_script}")
    
    return model_script  # Return the resolved path


def load_calibration_images(
    calib_dir: str, 
    target_size: Tuple[int, int] = (640, 640),
    max_samples: int = 0,
    supported_extensions: List[str] = None
) -> List[np.ndarray]:
    """
    Load and preprocess calibration images.
    
    Args:
        calib_dir: Directory containing calibration images
        target_size: Target size for resizing images (width, height)
        max_samples: Maximum number of samples to load (0 = load all)
        supported_extensions: List of supported image extensions
        
    Returns:
        List of preprocessed image arrays in CHW format
    """
    logger = logging.getLogger(__name__)
    
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    calib_path = Path(calib_dir)
    
    # Find all image files
    image_files = []
    for ext in supported_extensions:
        image_files.extend(calib_path.glob(f"*{ext}"))
        image_files.extend(calib_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {calib_dir} with extensions {supported_extensions}")
    
    # Limit number of samples if specified
    if max_samples > 0:
        image_files = image_files[:max_samples]
    
    logger.info(f"Loading {len(image_files)} calibration images from {calib_dir}")
    
    calib_data = []
    for i, img_path in enumerate(image_files):
        try:
            # Load and convert to RGB
            img = Image.open(img_path).convert('RGB')
            
            # Resize to target size
            img = img.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array as uint8 [0, 255]
            img_array = np.array(img, dtype=np.uint8)
            
            # Convert from HWC to CHW format  
            img_array = np.transpose(img_array, (2, 0, 1))
            
            calib_data.append(img_array)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} images...")
                
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
            continue
    
    if not calib_data:
        raise ValueError("No calibration images could be processed")
    
    logger.info(f"Successfully loaded {len(calib_data)} calibration images")
    return calib_data


def compile_onnx_to_har(
    onnx_path: str,
    output_path: str,
    model_script: str,
    calib_data: List[np.ndarray],
    config: dict,
    logger: logging.Logger
) -> bool:
    """
    Compile ONNX model to HAR format.
    
    Args:
        onnx_path: Path to input ONNX file
        output_path: Path for output HAR file
        model_script: Path to model script file
        calib_data: List of calibration image arrays
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize Hailo client runner
        hw_arch = config['hardware']['target']
        logger.info(f"Initializing ClientRunner for {hw_arch}")
        runner = ClientRunner(hw_arch=hw_arch)
        
        # Extract model configuration
        model_config = config['model']
        model_name = model_config['name']
        end_nodes = model_config['end_nodes']
        
        # Translate ONNX model
        logger.info(f"Translating ONNX model: {onnx_path}")
        logger.info(f"Using end nodes: {end_nodes}")
        
        try:
            # Try with YOLOv11 end nodes first
            result = runner.translate_onnx_model(
                model=onnx_path,
                net_name=model_name,
                end_node_names=end_nodes
            )
            logger.info("✅ Model translated successfully with YOLOv11 end nodes")
            
        except Exception as e:
            logger.warning(f"YOLOv11 end nodes failed: {e}")
            logger.info("Trying fallback end node...")
            
            # Fallback to concat node
            result = runner.translate_onnx_model(
                model=onnx_path,
                net_name=model_name,
                end_node_names=["/model.23/Concat_3"]
            )
            logger.info("✅ Model translated with fallback end node")
        
        # Load model script
        logger.info(f"Loading model script: {model_script}")
        with open(model_script, 'r') as f:
            script_content = f.read()
        runner.load_model_script(script_content)
        
        # Optimize model with calibration data
        logger.info(f"Starting model optimization with {len(calib_data)} calibration samples")
        
        # Use random sampling if calibration fails
        try:
            # Try with calibration data first
            runner.optimize(calib_data)
            logger.info("✅ Model optimized with calibration data")
        except Exception as e:
            logger.warning(f"Calibration optimization failed: {e}")
            logger.info("Falling back to random sampling...")
            try:
                # Use empty list for random sampling
                runner.optimize([])
                logger.info("✅ Model optimized with random sampling")
            except Exception as e2:
                logger.error(f"❌ Compilation failed: {e2}")
                raise
        
        # Save HAR file
        logger.info(f"Saving HAR file: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        runner.save_har(output_path)
        
        # Verify HAR file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ HAR file created successfully: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            return True
        else:
            logger.error("❌ HAR file was not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to Hailo Archive (HAR) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/compile_to_har.py --onnx outputs/onnx/model.onnx --output outputs/har/model.har
  
  # With custom calibration directory
  python scripts/compile_to_har.py --onnx model.onnx --output model.har --calib-dir /path/to/images
  
  # Using specific number of calibration samples
  python scripts/compile_to_har.py --onnx model.onnx --output model.har --num-calib 32
        """
    )
    
    parser.add_argument(
        "--onnx", 
        required=True, 
        help="Path to input ONNX file"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path for output HAR file"
    )
    parser.add_argument(
        "--model-script", 
        help="Path to model script file (default: from config)"
    )
    parser.add_argument(
        "--calib-dir", 
        help="Directory containing calibration images (default: from config)"
    )
    parser.add_argument(
        "--num-calib", 
        type=int, 
        default=0,
        help="Number of calibration samples to use (0 = use all available)"
    )
    parser.add_argument(
        "--config", 
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.model_script:
            model_script = args.model_script
        else:
            model_script = config['paths']['model_script']
            
        if args.calib_dir:
            calib_dir = args.calib_dir
        else:
            calib_dir = config['paths']['calibration_dir']
        
        # Set number of calibration samples
        if args.num_calib > 0:
            num_calib = args.num_calib
        else:
            num_calib = config['optimization']['num_calibration_samples']
        
        # Validate paths
        logger.info("Validating input paths...")
        model_script = validate_paths(args.onnx, calib_dir, model_script)
        
        # Load calibration data
        target_size = tuple(config['model']['input_size'])
        supported_extensions = config['validation']['supported_extensions']
        
        calib_data = load_calibration_images(
            calib_dir=calib_dir,
            target_size=target_size,
            max_samples=num_calib,
            supported_extensions=supported_extensions
        )
        
        # Compile ONNX to HAR
        logger.info("Starting ONNX to HAR compilation...")
        success = compile_onnx_to_har(
            onnx_path=args.onnx,
            output_path=args.output,
            model_script=model_script,
            calib_data=calib_data,
            config=config,
            logger=logger
        )
        
        if success:
            logger.info("🎉 Compilation completed successfully!")
            logger.info(f"📦 HAR file saved: {args.output}")
            sys.exit(0)
        else:
            logger.error("❌ Compilation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()