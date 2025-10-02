#!/usr/bin/env python3
"""
HAR to HEF Conversion Script

This script converts Hailo Archive (HAR) files to Hailo Executable Format (HEF)
for deployment on Hailo hardware.

Usage:
    python scripts/har_to_hef.py --har outputs/har/model.har --output outputs/hef/model.hef
    
For more options:
    python scripts/har_to_hef.py --help
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import yaml

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


def validate_har_file(har_path: str) -> None:
    """Validate that HAR file exists and is readable."""
    if not os.path.exists(har_path):
        raise FileNotFoundError(f"HAR file not found: {har_path}")
    
    if not os.path.isfile(har_path):
        raise ValueError(f"Path is not a file: {har_path}")
    
    # Check file size
    file_size = os.path.getsize(har_path)
    if file_size == 0:
        raise ValueError(f"HAR file is empty: {har_path}")
    
    if file_size < 1024:  # Less than 1KB seems suspicious
        raise ValueError(f"HAR file seems too small ({file_size} bytes): {har_path}")


def compile_har_to_hef(
    har_path: str,
    output_path: str,
    config: dict,
    compiler_optimization_level: str,
    logger: logging.Logger
) -> bool:
    """
    Compile HAR file to HEF format.
    
    Args:
        har_path: Path to input HAR file
        output_path: Path for output HEF file
        config: Configuration dictionary
        compiler_optimization_level: Compiler optimization level (0, 1, 2, max)
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize Hailo client runner
        hw_arch = config['hardware']['target']
        logger.info(f"Initializing ClientRunner for {hw_arch}")
        runner = ClientRunner(hw_arch=hw_arch)
        
        # Load HAR file
        logger.info(f"Loading HAR file: {har_path}")
        runner.load_har(har_path)
        
        file_size = os.path.getsize(har_path)
        logger.info(f"HAR file loaded: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        # Set compiler optimization level if not already in model script
        if compiler_optimization_level != "default":
            logger.info(f"Setting compiler optimization level to: {compiler_optimization_level}")
            
            # Add performance parameter to model script
            if compiler_optimization_level == "max":
                optimization_script = "performance_param(compiler_optimization_level=max)"
            else:
                optimization_script = f"performance_param(compiler_optimization_level={compiler_optimization_level})"
            
            runner.load_model_script(optimization_script, append=True)
        
        # Compile to HEF
        logger.info("Starting HAR to HEF compilation...")
        logger.info("This may take several minutes depending on model complexity...")
        
        runner.compile()
        
        # Extract HEF data
        logger.info("Extracting HEF data...")
        hef_data = runner.hef
        
        if hef_data is None:
            logger.error("❌ Failed to generate HEF data")
            return False
        
        if not isinstance(hef_data, bytes):
            logger.error(f"❌ Unexpected HEF data type: {type(hef_data)}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save HEF file
        logger.info(f"Saving HEF file: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(hef_data)
        
        # Verify HEF file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ HEF file created successfully: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            return True
        else:
            logger.error("❌ HEF file was not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert Hailo Archive (HAR) to Hailo Executable Format (HEF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/har_to_hef.py --har outputs/har/model.har --output outputs/hef/model.hef
  
  # With maximum compiler optimization
  python scripts/har_to_hef.py --har model.har --output model.hef --compiler-optimization-level max
  
  # With verbose logging
  python scripts/har_to_hef.py --har model.har --output model.hef --verbose
        """
    )
    
    parser.add_argument(
        "--har", 
        required=True, 
        help="Path to input HAR file"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path for output HEF file"
    )
    parser.add_argument(
        "--compiler-optimization-level",
        choices=["0", "1", "2", "max", "default"],
        default="default",
        help="Compiler optimization level (default: use config or model script setting)"
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
        
        # Validate HAR file
        logger.info("Validating HAR file...")
        validate_har_file(args.har)
        
        # Use config optimization level if not specified
        if args.compiler_optimization_level == "default":
            compiler_optimization_level = config['optimization']['compiler_optimization_level']
        else:
            compiler_optimization_level = args.compiler_optimization_level
        
        # Compile HAR to HEF
        logger.info("Starting HAR to HEF compilation...")
        success = compile_har_to_hef(
            har_path=args.har,
            output_path=args.output,
            config=config,
            compiler_optimization_level=compiler_optimization_level,
            logger=logger
        )
        
        if success:
            logger.info("🎉 Compilation completed successfully!")
            logger.info(f"📦 HEF file saved: {args.output}")
            logger.info("🚀 Model is ready for deployment on Hailo hardware!")
            sys.exit(0)
        else:
            logger.error("❌ Compilation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()