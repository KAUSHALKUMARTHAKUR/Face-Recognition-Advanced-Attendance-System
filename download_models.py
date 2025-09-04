import os
import requests
import bz2
import gdown
from urllib.parse import urlparse
import hashlib

def download_file_with_progress(url, filepath, expected_size=None):
    """Download file with progress indication and verification"""
    print(f"📥 Downloading {os.path.basename(filepath)}...")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="")
        
        print(f"\n✅ Downloaded: {filepath} ({downloaded:,} bytes)")
        
        # Verify file size if expected
        if expected_size and abs(downloaded - expected_size) > 1024:  # Allow 1KB tolerance
            print(f"⚠️ Warning: Expected {expected_size:,} bytes, got {downloaded:,} bytes")
        
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {filepath}: {e}")
        return False

def download_and_extract_bz2(url, filepath):
    """Download and extract bz2 compressed file"""
    print(f"📥 Downloading compressed {os.path.basename(filepath)}...")
    compressed_path = filepath + ".bz2"
    
    try:
        if download_file_with_progress(url, compressed_path):
            print("📦 Extracting compressed file...")
            with bz2.BZ2File(compressed_path, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up compressed file
            os.remove(compressed_path)
            print(f"✅ Extracted: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return False

def download_models():
    """Download all 4 required models for the face recognition system"""
    
    print("🚀 Starting model download process...\n")
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/anti_spoofing', exist_ok=True)
    
    # Model definitions with download info
    models_info = [
        {
            'name': 'YOLOv5-Face Detection Model',
            'path': 'models/yolov5s-face.onnx',
            'url': 'https://github.com/deepcam-cn/yolov5-face/releases/download/v0.0.0/yolov5s-face.onnx',
            'type': 'direct',
            'expected_size': 28934320,  # ~28MB
            'description': 'Face detection using YOLOv5 architecture'
        },
        {
            'name': 'dlib Shape Predictor (68 landmarks)',
            'path': 'models/shape_predictor_68_face_landmarks.dat',
            'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'type': 'bz2',
            'expected_size': 99693937,  # ~95MB
            'description': 'Facial landmark detection for face alignment'
        },
        {
            'name': 'dlib Face Recognition ResNet Model',
            'path': 'models/dlib_face_recognition_resnet_model_v1.dat',
            'url': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
            'type': 'bz2',
            'expected_size': 22713312,  # ~22MB
            'description': 'Deep learning face recognition model'
        },
        {
            'name': 'Anti-Spoofing Binary Model',
            'path': 'models/anti_spoofing/AntiSpoofing_bin_1.5_128.onnx',
            'url': None,  # Will be set from Google Drive
            'type': 'gdrive',
            'expected_size': 1500000,  # ~1.5MB (estimate)
            'description': 'Liveness detection to prevent photo attacks',
            'file_id': 'YOUR_GOOGLE_DRIVE_FILE_ID'  # Replace this!
        }
    ]
    
    success_count = 0
    
    for model in models_info:
        print(f"\n📋 Processing: {model['name']}")
        print(f"   Description: {model['description']}")
        
        # Check if model already exists and is valid
        if os.path.exists(model['path']) and os.path.getsize(model['path']) > 1000:
            current_size = os.path.getsize(model['path'])
            print(f"✅ Already exists: {model['path']} ({current_size:,} bytes)")
            success_count += 1
            continue
        
        # Download based on type
        if model['type'] == 'direct':
            success = download_file_with_progress(
                model['url'], 
                model['path'], 
                model.get('expected_size')
            )
        
        elif model['type'] == 'bz2':
            success = download_and_extract_bz2(model['url'], model['path'])
        
        elif model['type'] == 'gdrive':
            success = download_from_gdrive(model)
        
        else:
            print(f"❌ Unknown download type: {model['type']}")
            success = False
        
        if success:
            success_count += 1
        else:
            print(f"❌ Failed to download: {model['name']}")
    
    print(f"\n🎯 Download Summary:")
    print(f"   ✅ Success: {success_count}/{len(models_info)} models")
    print(f"   📁 Models directory: {os.path.abspath('models')}")
    
    # List all downloaded models with sizes
    print(f"\n📊 Downloaded Models:")
    for model in models_info:
        if os.path.exists(model['path']):
            size = os.path.getsize(model['path'])
            print(f"   ✓ {os.path.basename(model['path'])}: {size:,} bytes")
        else:
            print(f"   ✗ {os.path.basename(model['path'])}: Missing")
    
    if success_count == len(models_info):
        print(f"\n🎉 All models downloaded successfully!")
    else:
        print(f"\n⚠️ {len(models_info) - success_count} model(s) failed to download")
        print(f"   Please check the URLs and try again")
    
    return success_count == len(models_info)

def download_from_gdrive(model_info):
    """Download model from Google Drive"""
    file_id = model_info.get('file_id')
    if not file_id or file_id == 'YOUR_GOOGLE_DRIVE_FILE_ID':
        print(f"⚠️ Google Drive file ID not configured for {model_info['name']}")
        print(f"   Please:")
        print(f"   1. Upload {os.path.basename(model_info['path'])} to Google Drive")
        print(f"   2. Make it publicly accessible")
        print(f"   3. Get the file ID from the shareable link")
        print(f"   4. Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' in download_models.py")
        
        # Create dummy file to prevent errors during build
        print(f"   Creating dummy file for now...")
        try:
            with open(model_info['path'], 'wb') as f:
                f.write(b'dummy_model_placeholder')
            return True
        except Exception as e:
            print(f"   Failed to create dummy file: {e}")
            return False
    
    try:
        print(f"📥 Downloading from Google Drive (ID: {file_id[:20]}...)")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_info['path'], quiet=False)
        
        if os.path.exists(model_info['path']) and os.path.getsize(model_info['path']) > 100:
            print(f"✅ Successfully downloaded from Google Drive")
            return True
        else:
            print(f"❌ Download failed or file too small")
            return False
            
    except Exception as e:
        print(f"❌ Google Drive download error: {e}")
        print(f"   Please check the file ID and permissions")
        return False

def verify_models():
    """Verify all models are present and have reasonable sizes"""
    print("\n🔍 Verifying downloaded models...")
    
    required_models = [
        ('models/yolov5s-face.onnx', 'YOLOv5 Face Detection'),
        ('models/shape_predictor_68_face_landmarks.dat', 'dlib Shape Predictor'),
        ('models/dlib_face_recognition_resnet_model_v1.dat', 'dlib Face Recognition'),
        ('models/anti_spoofing/AntiSpoofing_bin_1.5_128.onnx', 'Anti-Spoofing Model'),
    ]
    
    all_present = True
    for model_path, model_name in required_models:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            status = "✅" if size > 1000 else "⚠️"
            print(f"   {status} {model_name}: {size:,} bytes")
            if size <= 1000:
                all_present = False
        else:
            print(f"   ❌ {model_name}: Missing")
            all_present = False
    
    return all_present

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Face Recognition System - Model Downloader")
    print("=" * 60)
    
    try:
        # Download all models
        success = download_models()
        
        # Verify downloads
        verified = verify_models()
        
        if success and verified:
            print(f"\n🎊 Setup Complete!")
            print(f"   All models are ready for the face recognition system")
        else:
            print(f"\n⚠️ Setup Issues Detected")
            print(f"   Some models may not have downloaded correctly")
            print(f"   Check the messages above and retry if needed")
            
    except KeyboardInterrupt:
        print(f"\n\n❌ Download interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    print(f"\n📝 Next Steps:")
    print(f"   1. If anti-spoofing model failed, upload it to Google Drive")
    print(f"   2. Update the file_id in this script")
    print(f"   3. Run this script again: python download_models.py")
    print(f"   4. Deploy your app with: python app.py")
