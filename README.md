# YOLO to Hailo Conversion Pipeline

This repository provides a clean, production-ready pipeline for converting YOLO models to Hailo Executable Format (HEF) for deployment on Hailo AI accelerators.

## 🎯 Overview

The pipeline supports the complete conversion workflow:
- **ONNX → HAR**: Model optimization and quantization with calibration data
- **HAR → HEF**: Final compilation for Hailo hardware deployment

### Supported Models
- YOLOv8, YOLOv11, and compatible YOLO architectures
- Input format: RGB uint8, 640x640 (configurable)
- Target hardware: Hailo8L (configurable)

## 📁 Project Structure

```
repo/
├── README.md                           # This file
├── scripts/
│   ├── compile_to_har.py              # ONNX → HAR conversion
│   └── har_to_hef.py                  # HAR → HEF compilation
├── model_assets/
│   ├── model_script.hls               # Hailo optimization script
│   └── sample_calibration/            # Sample calibration images
├── configs/
│   └── default.yaml                   # Configuration file
├── outputs/
│   ├── onnx/                          # ONNX models
│   ├── har/                           # HAR files  
│   └── hef/                           # HEF files
├── .gitignore
└── requirements.txt                   # For reference only
```

## 🚀 Quick Start

### Prerequisites

**Virtual Environment**: A pre-configured Python environment is available at:
```bash
/home/thanhan/Downloads/compiler/myenv
```

⚠️ **Important**: Do NOT install new dependencies. Use the existing environment.

### Activate Environment

```bash
source /home/thanhan/Downloads/compiler/myenv/bin/activate
```

### Basic Usage

1. **Place your ONNX model** in `outputs/onnx/`
2. **Run ONNX → HAR conversion**:
   ```bash
   python scripts/compile_to_har.py \
     --onnx outputs/onnx/your_model.onnx \
     --output outputs/har/your_model.har
   ```

3. **Run HAR → HEF compilation**:
   ```bash
   python scripts/har_to_hef.py \
     --har outputs/har/your_model.har \
     --output outputs/hef/your_model.hef
   ```

## 📋 Detailed Usage

### Step A: ONNX to HAR Conversion

Convert ONNX model to HAR format with optimization and quantization:

```bash
python scripts/compile_to_har.py \
  --onnx outputs/onnx/model.onnx \
  --output outputs/har/model.har \
  --calib-dir model_assets/sample_calibration \
  --num-calib 64 \
  --verbose
```

**Parameters:**
- `--onnx`: Path to input ONNX file (required)
- `--output`: Path for output HAR file (required)  
- `--calib-dir`: Directory with calibration images (default: from config)
- `--num-calib`: Number of calibration samples (0 = use all)
- `--model-script`: Custom model script path
- `--config`: Configuration file (default: configs/default.yaml)
- `--verbose`: Enable detailed logging

### Step B: HAR to HEF Compilation

Compile HAR file to final HEF format:

```bash
python scripts/har_to_hef.py \
  --har outputs/har/model.har \
  --output outputs/hef/model.hef \
  --compiler-optimization-level max \
  --verbose
```

**Parameters:**
- `--har`: Path to input HAR file (required)
- `--output`: Path for output HEF file (required)
- `--compiler-optimization-level`: Optimization level (0, 1, 2, max, default)
- `--config`: Configuration file
- `--verbose`: Enable detailed logging

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

### Model Settings
```yaml
model:
  name: "your_model_name"
  input_size: [640, 640]
  input_format: "RGB"
  end_nodes:  # YOLOv11 end nodes
    - "/model.23/cv3.0/cv3.0.2/Conv"
    - "/model.23/cv2.0/cv2.0.2/Conv"
    # ... more nodes
```

### Hardware & Optimization
```yaml
hardware:
  target: "hailo8l"

optimization:
  num_calibration_samples: 64
  compiler_optimization_level: "max"
  model_optimization_level: 2
```

### Paths
```yaml
paths:
  calibration_dir: "/path/to/your/calibration/images"
  model_script: "model_assets/model_script.hls"
```

## 🖼️ Calibration Images

### Using Sample Images
The repository includes 10 sample images in `model_assets/sample_calibration/` for testing.

### Using Your Own Images
1. Replace images in `model_assets/sample_calibration/` or update `calibration_dir` in config
2. Use 10-100 representative images from your dataset
3. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
4. Images will be automatically resized to 640x640

### Calibration Directory Structure
```
your_calibration_dir/
├── image001.jpg
├── image002.jpg
└── ...
```

## 🔧 Model Script Customization

Edit `model_assets/model_script.hls` for model-specific optimizations:

```hls
# Performance optimization
performance_param(compiler_optimization_level=max)

# RGB normalization (0-255 → 0-1)
normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])

# Model optimization level
model_optimization_config(optimization_level=2)

# Bias correction for better accuracy
post_quantization_optimization(bias_correction=True)
```

## 📊 Example Workflow

Complete example with the provided paper_ball model:

```bash
# Activate environment
source /home/thanhan/Downloads/compiler/myenv/bin/activate

# Step 1: ONNX → HAR (using sample calibration)
python scripts/compile_to_har.py \
  --onnx outputs/onnx/paper_ball.onnx \
  --output outputs/har/paper_ball.har \
  --calib-dir model_assets/sample_calibration \
  --num-calib 10

# Step 2: HAR → HEF  
python scripts/har_to_hef.py \
  --har outputs/har/paper_ball.har \
  --output outputs/hef/paper_ball.hef \
  --compiler-optimization-level max

# Verify outputs
ls -la outputs/har/paper_ball.har
ls -la outputs/hef/paper_ball.hef
```

## 🐛 Troubleshooting

### Common Issues

**1. Hailo SDK not found**
```
Error: Hailo SDK not found. Make sure you're using the correct virtual environment.
```
→ Activate the correct environment: `source /home/thanhan/Downloads/compiler/myenv/bin/activate`

**2. ONNX file not found**
```
FileNotFoundError: ONNX file not found: path/to/model.onnx
```
→ Check file path and ensure ONNX file exists

**3. No calibration images**
```
ValueError: No image files found in directory
```
→ Ensure calibration directory contains supported image formats

**4. Compilation fails with end nodes**
```
YOLOv11 end nodes failed: ...
```
→ Script automatically falls back to `/model.23/Concat_3` node

### Logging

Enable verbose logging for debugging:
```bash
python scripts/compile_to_har.py --verbose [other args]
python scripts/har_to_hef.py --verbose [other args]
```

## 📈 Performance Notes

- **Compilation time**: 5-15 minutes depending on model complexity and optimization level
- **HAR file size**: ~50-100 MB for typical YOLO models  
- **HEF file size**: ~10-20 MB for typical YOLO models
- **Optimization levels**:
  - `0`: Fastest compilation, basic optimization
  - `max`: Slower compilation, best performance

## 🔄 Updating for New Models

To convert a new YOLO model:

1. **Update config** (`configs/default.yaml`):
   ```yaml
   model:
     name: "new_model_name"
     # Update end_nodes if needed
   ```

2. **Add calibration images**:
   ```bash
   cp /path/to/new/calibration/images/* model_assets/sample_calibration/
   ```

3. **Run conversion**:
   ```bash
   python scripts/compile_to_har.py --onnx outputs/onnx/new_model.onnx --output outputs/har/new_model.har
   python scripts/har_to_hef.py --har outputs/har/new_model.har --output outputs/hef/new_model.hef
   ```

## 📝 Sample Log Output

Successful compilation log excerpt:
```
2025-10-02 10:44:32 - __main__ - INFO - Loading configuration from configs/default.yaml
2025-10-02 10:44:32 - __main__ - INFO - Validating input paths...
2025-10-02 10:44:32 - __main__ - INFO - Loading 10 calibration images from model_assets/sample_calibration
2025-10-02 10:44:33 - __main__ - INFO - Successfully loaded 10 calibration images
2025-10-02 10:44:33 - __main__ - INFO - Initializing ClientRunner for hailo8l
2025-10-02 10:44:34 - __main__ - INFO - Translating ONNX model: outputs/onnx/paper_ball.onnx
[info] Translation completed on ONNX model paper_ball (completion time: 00:00:00.98)
2025-10-02 10:44:35 - __main__ - INFO - ✅ Model translated successfully with YOLOv11 end nodes
2025-10-02 10:44:35 - __main__ - INFO - Starting model optimization with 10 calibration samples
[info] Model Optimization is done
2025-10-02 10:47:22 - __main__ - INFO - ✅ Model optimized with calibration data  
2025-10-02 10:47:22 - __main__ - INFO - Saving HAR file: outputs/har/paper_ball.har
2025-10-02 10:47:23 - __main__ - INFO - ✅ HAR file created successfully: 57,190,400 bytes (54.5 MB)
2025-10-02 10:47:23 - __main__ - INFO - 🎉 Compilation completed successfully!
```

---

## 📞 Support

- Check logs with `--verbose` flag for detailed error information
- Ensure virtual environment is activated before running scripts
- Verify all file paths are correct and accessible
- For model-specific issues, check ONNX model compatibility with Hailo SDK

**Ready to deploy on Hailo hardware!** 🚀