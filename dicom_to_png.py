#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pydicom
import numpy as np
from PIL import Image
import concurrent.futures

def normalize_dicom_pixels(dicom_data):
    """Normalize DICOM pixel array to 8-bit range."""
    pixels = dicom_data.pixel_array
    
    # Normalize based on window center and width if available
    if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
        center = dicom_data.WindowCenter
        width = dicom_data.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        vmin = center - width // 2
        vmax = center + width // 2
        pixels = np.clip(pixels, vmin, vmax)
    
    # Normalize to 0-255 range
    if pixels.max() != pixels.min():
        pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255).astype(np.uint8)
    else:
        pixels = np.zeros_like(pixels, dtype=np.uint8)
    
    return pixels

def is_valid_dicom(file_path):
    """Check if a file is a valid DICOM file with pixel data."""
    try:
        dicom_data = pydicom.dcmread(file_path)
        return hasattr(dicom_data, 'pixel_array')
    except:
        return False

def convert_single_file(dicom_path, output_dir):
    """Convert a single DICOM file to PNG format."""
    try:
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(dicom_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.png")
        
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Convert to normalized pixel array
        pixels = normalize_dicom_pixels(dicom_data)
        
        # Create and save PNG
        image = Image.fromarray(pixels)
        image.save(output_path)
        
        if not os.path.exists(output_path):
            return {"filename": filename, "status": "error", "error": "Failed to save PNG file"}
            
        return {
            "filename": filename,
            "status": "success",
            "output_path": output_path
        }
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "error": str(e)
        }

def convert_folder(folder_path, max_workers=5):
    """Convert all DICOM files in a folder to PNG format."""
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return
    
    # Create png output directory
    output_dir = folder_path / "png"
    output_dir.mkdir(exist_ok=True)
    
    # Get all files in the folder
    all_files = list(folder_path.glob("*"))
    dicom_files = []
    
    print("Checking for valid DICOM files...")
    for file_path in tqdm(all_files, desc="Validating files"):
        if is_valid_dicom(str(file_path)):
            dicom_files.append(file_path)
    
    if not dicom_files:
        print(f"No valid DICOM files found in {folder_path}")
        return
    
    print(f"\nFound {len(dicom_files)} valid DICOM files in {folder_path}")
    results = []
    
    # Use ThreadPoolExecutor for parallel conversion
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for dicom_path in dicom_files:
            future = executor.submit(convert_single_file, str(dicom_path), str(output_dir))
            futures.append(future)
        
        # Display progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures), 
                         desc="Converting DICOM files"):
            results.append(future.result())
    
    # Print summary
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"\nConversion Summary:")
    print(f"  Total files: {len(results)}")
    print(f"  Successful conversions: {len(successful)}")
    print(f"  Failed conversions: {len(failed)}")
    
    if successful:
        print("\nSuccessfully converted files:")
        for s in successful:
            print(f"  {s['filename']}")
    
    if failed:
        print("\nFailed conversions:")
        for f in failed:
            print(f"  {f['filename']}: {f['error']}")
    
    if successful:
        print(f"\nConverted files saved to {output_dir}")
        # Verify the output files exist
        png_files = list(output_dir.glob("*.png"))
        print(f"Number of PNG files created: {len(png_files)}")

def main():
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG format")
    parser.add_argument("folder", help="Folder containing DICOM files")
    parser.add_argument("--workers", type=int, default=5, 
                       help="Maximum number of concurrent conversions")
    
    args = parser.parse_args()
    convert_folder(args.folder, args.workers)

if __name__ == "__main__":
    main() 