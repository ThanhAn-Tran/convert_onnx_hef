# Sample Calibration Images

This directory contains sample calibration images for model quantization.

## Usage

These images are used during the quantization process to ensure the model maintains accuracy when converted to INT8 precision.

## Image Details

- **Format**: JPG
- **Count**: 10 sample images
- **Resolution**: Variable (will be resized to 640x640 during processing)
- **Content**: Sample frames from paper ball detection dataset

## Adding Your Own Images

To use your own calibration images:

1. Replace the images in this directory with representative samples from your dataset
2. Ensure images are in supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
3. Use 10-100 images for best results
4. Images should be representative of your real-world data

## Note

For production use, you should use calibration images that are representative of your actual inference data to ensure optimal model accuracy.