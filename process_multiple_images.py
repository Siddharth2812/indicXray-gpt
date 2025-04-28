from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import base64
from pathlib import Path
import asyncio
import csv
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def load_png_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

def process_single_image(img_path):
    try:
        img_base64 = load_png_as_base64(str(img_path))
        
        response = llm.invoke(input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe the image in detail, what is the issue that you can find in the image chest?",
                    },
                ],
            },
        ])
        
        return {
            'filename': img_path.name,
            'response': response.content,
            'status': 'success'
        }
    except Exception as e:
        return {
            'filename': img_path.name,
            'response': str(e),
            'status': 'error'
        }

def process_folder_images(folder_path, output_csv=None):
    # Get all PNG files in the folder
    image_files = list(Path(folder_path).glob("*.png"))
    
    if not image_files:
        print(f"No PNG images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output filename with timestamp if not provided
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"image_analysis_results_{timestamp}.csv"
    
    # Process images concurrently
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Use tqdm to show progress bar
        futures = [executor.submit(process_single_image, img_path) for img_path in image_files]
        for future in tqdm(futures, total=len(image_files), desc="Processing images"):
            results.append(future.result())
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} images")
    print(f"Failed to process: {error_count} images")

if __name__ == "__main__":
    # Example usage
    folder_path = "/Users/dog/Downloads/drive-download-20250427T052722Z-001/png"
    process_folder_images(folder_path) 