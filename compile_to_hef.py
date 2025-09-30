#!/usr/bin/env python3
"""
Script để compile YOLOv11 ONNX model sang Hailo .hef format
"""

from hailo_sdk_client import ClientRunner
import os

def compile_yolov11_to_hef():
    """
    Compile ONNX model sang HEF format cho Hailo
    """
    
    # Khởi tạo Hailo SDK Client
    runner = ClientRunner(hw_arch='hailo8l')  # hoặc 'hailo8l', 'hailo15'
    
    print("📥 Đang load ONNX model...")
    
    # Parse ONNX model
    onnx_path = 'paper_ball.onnx'
    runner.parse_onnx(
        onnx_model_path=onnx_path,
        net_name='yolov11_custom'
    )
    
    print("✓ Load ONNX thành công!")
    
    # Optimize model
    print("\n🔧 Đang optimize model...")
    runner.optimize()
    print("✓ Optimize hoàn tất!")
    
    # Quantize model với calibration dataset
    print("\n📊 Đang quantize model (FP32 → INT8)...")
    print("   Đang load calibration images...")
    
    calib_dataset_path = './calibration_images'
    
    # Load calibration images
    runner.load_model_script(
        f"""
        import glob
        import numpy as np
        from PIL import Image
        
        def preprocess_image(img_path):
            img = Image.open(img_path).convert('RGB')
            img = img.resize((640, 640))
            img_array = np.array(img).astype(np.float32)
            img_array = img_array / 255.0  # Normalize
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
            return img_array
        
        calib_images = glob.glob('{calib_dataset_path}/*.jpg')
        calib_data = [preprocess_image(img) for img in calib_images[:100]]
        """
    )
    
    # Quantize
    runner.quantize(
        calib_dataset=runner.model_script.calib_data,
        quantization_algorithm='symmetric'
    )
    
    print("✓ Quantization hoàn tất!")
    
    # Compile sang HEF
    print("\n🚀 Đang compile sang HEF format...")
    hef_path = 'yolov11_custom.hef'
    
    runner.compile(
        batch_size=1,
        optimization_level=2  # 0-4
    )
    
    # Save HEF file
    runner.save_har('./yolov11_custom.har')  # HAR = Hailo Archive (intermediate)
    
    print(f"\n✅ Compile thành công!")
    print(f"📦 File HEF: {hef_path}")
    
    # Thông tin model
    print("\n📋 Model Info:")
    print(f"   Input shape: {runner.model.input_shapes}")
    print(f"   Output shape: {runner.model.output_shapes}")
    print(f"   FPS ước tính: ~{runner.get_fps()} fps")
    
    return hef_path

if __name__ == "__main__":
    try:
        hef_file = compile_yolov11_to_hef()
        print(f"\n🎉 Hoàn tất! File .hef của bạn: {hef_file}")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi compile: {str(e)}")
        print("\n💡 Gợi ý debug:")
        print("   1. Kiểm tra ONNX model có hợp lệ không")
        print("   2. Đảm bảo có đủ calibration images (50-100 ảnh)")
        print("   3. Kiểm tra input shape khớp với model")
        print("   4. Xem log chi tiết trong ./logs/")