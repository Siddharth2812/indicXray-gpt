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
import concurrent.futures

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)

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
                        "text": """You are a radiologist.  
                                    Look at the chest X-ray image provided.  
                                    Based only on what you see in the image, generate a structured radiology report.

                                    Your response should include the following sections:

                                    1.⁠ ⁠View – Identify the view of the image if possible (e.g., Posteroanterior (PA), Anteroposterior (AP), Lateral). If uncertain, state "View not clearly identifiable."
                                    2.⁠ ⁠⁠Findings – Describe observations structured by organ systems:
                                    - Lung Fields  
                                    - Pleura  
                                    - Mediastinum  
                                    - Diaphragm  
                                    - Cardiac Silhouette  
                                    - Bones  
                                    If a region appears normal, write "No abnormality detected" for that region.
                                    3.⁠ ⁠Impression – Concise summary of the key findings. If the X-ray is normal, state: "No significant abnormality detected."
                                    4.⁠ ⁠Suggestions – (Optional) Mention further evaluation or follow-up only if it directly follows from visible findings (e.g., "Consider CT if mass is suspected"). Otherwise, write "No specific suggestion."

                                    Important rules:
                                    •⁠  ⁠Do not make any assumptions.
                                    •⁠  ⁠Do not include clinical history or demographics.
                                    •⁠  ⁠Do not describe any region not visible in the image.
                                    •⁠  ⁠Your report should be strictly image-based.
                                    •⁠  ⁠Your report should be in text format only not markdown.
                    """,
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
        output_csv = f"gpt_analysis_results.csv"
    
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

def process_images_from_csv(csv_file_path, images_folder_path, concurrent_workers=10, output_csv=None):
    """
    Process images from a CSV file concurrently, adding GPT analysis as a new column.
    Only processes images that exist in the specified folder.
    
    Args:
        csv_file_path: Path to CSV file containing image case IDs
        images_folder_path: Path to folder containing the images
        concurrent_workers: Number of images to process concurrently
        output_csv: Output CSV file path
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get all image files in the folder as a set for faster lookup
    image_files_set = set([f.name for f in Path(images_folder_path).glob("*.png")])
    print(f"Found {len(image_files_set)} images in the folder")
    
    # Create a new column for GPT responses
    if 'gpt_analysis' not in df.columns:
        df['gpt_analysis'] = None
    
    # Create output filename with timestamp if not provided
    if output_csv is None:
        output_csv = f"gpt_analysis_results.csv"
    
    # Filter to only include images that exist in the folder
    images_to_process = []
    for idx, row in df.iterrows():
        case_id = row['case ID']
        if case_id in image_files_set:
            images_to_process.append((idx, case_id))
        else:
            print(f"✗ Image not found: {case_id}")
    
    total_to_process = len(images_to_process)
    print(f"Found {total_to_process} images in CSV that exist in the folder")
    
    # Process images concurrently
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = {}
        for idx, case_id in images_to_process:
            img_path = Path(images_folder_path) / case_id
            future = executor.submit(process_single_image, img_path)
            futures[future] = (idx, case_id)
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_to_process, desc="Processing images"):
            idx, case_id = futures[future]
            try:
                result = future.result()
                df.at[idx, 'gpt_analysis'] = result['response']
                print(f"✓ Success: {case_id}")
            except Exception as e:
                print(f"✗ Error processing {case_id}: {str(e)}")
                df.at[idx, 'gpt_analysis'] = f"Error: {str(e)}"
            
            # Save intermediate results after each image is processed
            df.to_csv(output_csv, index=False)
    
    # Print summary
    success_count = sum(1 for result in df['gpt_analysis'] if result is not None and not str(result).startswith('Error:'))
    not_found_count = len(df) - total_to_process
    error_count = total_to_process - success_count
    
    print(f"\nProcessing complete!")
    print(f"Total images in CSV: {len(df)}")
    print(f"Successfully processed: {success_count} images")
    print(f"Failed to process: {error_count} images")
    print(f"Images not found in folder: {not_found_count}")
    print(f"Results saved to {output_csv}")
    
    return df

if __name__ == "__main__":
    # Example usage
    # folder_path = "/Users/dog/Downloads/drive-download-20250427T052722Z-001/png"
    # process_folder_images(folder_path) 
    
    # Use the new function to process images from CSV
    csv_file_path = "xrays_with_gt.csv"
    images_folder_path = "/Users/dog/Downloads/Pallavi CXR/pal abnormal xrays/50 Xrays from Pallavi/png"
    process_images_from_csv(csv_file_path, images_folder_path, concurrent_workers=10) 