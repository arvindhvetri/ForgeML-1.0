import os
import random
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path

def generate_ela_image(image_path, quality=90):
    """Applies Error Level Analysis to expose compression anomalies."""
    temp_filename = 'temp_ela.jpg'
    try:
        original = Image.open(image_path).convert('RGB')
        original.save(temp_filename, 'JPEG', quality=quality)
        compressed = Image.open(temp_filename)
        ela_image = ImageChops.difference(original, compressed)
        
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0: max_diff = 1
            
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def process_directory(input_dir, category_name, output_base, split_ratio=0.8):
    """Reads a specific directory, applies ELA, and splits into train/val."""
    print(f"\nScanning {category_name} directory: {input_dir}...")
    
    # Get all valid images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(all_files)} images. Shuffling and applying ELA...")
    
    # Shuffle for random train/val distribution
    random.seed(42)
    random.shuffle(all_files)
    
    # Split the data
    split_idx = int(len(all_files) * split_ratio)
    splits = {
        'train': all_files[:split_idx],
        'val': all_files[split_idx:]
    }
    
    # Process and save
    for split_name, file_list in splits.items():
        print(f"  -> Processing {split_name} split ({len(file_list)} images)...")
        
        # Ensure output directory exists
        out_dir = Path(output_base) / split_name / category_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, filename in enumerate(file_list):
            input_path = os.path.join(input_dir, filename)
            ela_result = generate_ela_image(input_path)
            
            if ela_result:
                # Save with 'ela_' prefix
                save_path = out_dir / f"ela_{filename.split('.')[0]}.jpg"
                ela_result.save(save_path, 'JPEG')
                
            if (i + 1) % 500 == 0:
                print(f"     Processed {i + 1}/{len(file_list)}...")

if __name__ == '__main__':
    # Your exact paths
    AU_DIR = r'D:\Hackathon\CASIA2.0_revised\Au'
    TP_DIR = r'D:\Hackathon\CASIA2.0_revised\Tp'
    OUTPUT_DIR = r'D:\Hackathon\dataset'
    
    print("Starting ELA Pipeline...")
    
    # Process Authentic Images
    if os.path.exists(AU_DIR):
        process_directory(AU_DIR, 'authentic', OUTPUT_DIR)
    else:
        print(f"Error: Could not find {AU_DIR}")
        
    # Process Tampered Images
    if os.path.exists(TP_DIR):
        process_directory(TP_DIR, 'tampered', OUTPUT_DIR)
    else:
        print(f"Error: Could not find {TP_DIR}")
        
    print(f"\n✅ All done! Your PyTorch-ready dataset is at: {OUTPUT_DIR}")