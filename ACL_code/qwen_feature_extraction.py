#!/usr/bin/env python3
"""
Extract visual features from street view images using Qwen 2.5-VL
Based on the paper's semantic segmentation features
"""
import os
import torch
import json
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import multiprocessing as mp
from multiprocessing import Process, Manager
import time
from datetime import datetime
import logging
import sys
from tqdm import tqdm
import glob
from PIL import Image

# 19 visual features from the paper
VISUAL_FEATURES = [
    "tree", "road", "sky", "building", "sidewalk",
    "grass", "car", "wall", "fence", "floor",
    "earth", "plant", "signboard", "skyscraper", "ceiling",
    "rail", "water", "palm tree", "unknown"
]

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger"""
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Also output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_existing_data(json_path):
    """Load existing survey data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_image_path(road_id, gsv_dir="GSV_images", ai_dir="AI_images"):
    """Get the correct image path based on road ID"""
    if road_id.startswith('loc_') and '_p' in road_id:
        # AI-enhanced image
        base_loc = road_id.split('_p')[0]
        loc_num = base_loc.replace('loc_', '')
        image_path = os.path.join(ai_dir, loc_num, f"{road_id.replace('loc_', '')}.jpg")
    else:
        # Original GSV image
        loc_num = road_id.replace('loc_', '')
        image_path = os.path.join(gsv_dir, f"{loc_num}.png")
    
    return image_path

def extract_features_with_qwen(image_path, model, processor, logger):
    """Extract visual features using Qwen 2.5-VL"""
    try:
        # Load and preprocess image
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
        
        # Construct the prompt
        prompt = f"""You are an expert in urban scene analysis. Analyze this street view image and provide the percentage composition of the following visual elements. These percentages should sum to approximately 100%.

Visual elements to identify:
1. tree - Any type of trees, including foliage and branches
2. road - The paved road surface for vehicles
3. sky - The visible sky area
4. building - Building facades, walls of buildings
5. sidewalk - Pedestrian walkways, pavements
6. grass - Grass areas, lawns
7. car - Vehicles of any type
8. wall - Standalone walls, barriers (not building facades)
9. fence - Fences, railings, barriers
10. floor - Ground surfaces not categorized as road or sidewalk
11. earth - Exposed soil, dirt areas
12. plant - Vegetation other than trees and grass (bushes, flowers, etc.)
13. signboard - Signs, billboards, traffic signs
14. skyscraper - Tall buildings, high-rises
15. ceiling - Any overhead structures, tunnels, covered areas
16. rail - Rails, tracks, guardrails
17. water - Water bodies, fountains, pools
18. palm tree - Palm trees specifically
19. unknown - Any elements that don't fit the above categories

IMPORTANT INSTRUCTIONS:
- Provide ONLY the percentages as numbers (0-100)
- The total should be approximately 100%
- If an element is not present, use 0
- Be precise in distinguishing between similar categories (e.g., tree vs palm tree, building vs skyscraper)
- Consider the entire visible area of the image

Respond in this exact JSON format without any additional text:
{{
  "tree": <percentage>,
  "road": <percentage>,
  "sky": <percentage>,
  "building": <percentage>,
  "sidewalk": <percentage>,
  "grass": <percentage>,
  "car": <percentage>,
  "wall": <percentage>,
  "fence": <percentage>,
  "floor": <percentage>,
  "earth": <percentage>,
  "plant": <percentage>,
  "signboard": <percentage>,
  "skyscraper": <percentage>,
  "ceiling": <percentage>,
  "rail": <percentage>,
  "water": <percentage>,
  "palm_tree": <percentage>,
  "unknown": <percentage>
}}"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Prepare inference
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON response
        try:
            # Find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', output_text, re.DOTALL)
            if json_match:
                features_dict = json.loads(json_match.group())
                
                # Normalize feature names (replace underscores with spaces for palm tree)
                if "palm_tree" in features_dict:
                    features_dict["palm tree"] = features_dict.pop("palm_tree")
                
                # Ensure all features are present
                for feature in VISUAL_FEATURES:
                    if feature not in features_dict:
                        features_dict[feature] = 0.0
                
                # Normalize to ensure sum is 100%
                total = sum(features_dict.values())
                if total > 0:
                    for feature in features_dict:
                        features_dict[feature] = (features_dict[feature] / total) * 100
                
                return features_dict
            else:
                logger.error(f"No JSON found in response: {output_text}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Output text: {output_text}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None

def process_on_gpu(gpu_id, data_subset, output_dir, gpu_mapping):
    """Process a subset of data on a specific GPU"""
    # Set GPU
    actual_gpu_id = gpu_mapping[gpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(actual_gpu_id)
    
    # Setup logger
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(f'GPU_{gpu_id}', os.path.join(log_dir, f'gpu_{gpu_id}.log'))
    
    logger.info(f"Starting process on GPU {gpu_id} (actual GPU {actual_gpu_id}) for {len(data_subset)} users")
    
    # Load model
    logger.info("Loading Qwen2.5-VL model...")
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Process data
    processed_data = []
    
    for user_idx, user_data in enumerate(data_subset):
        user_copy = user_data.copy()
        logger.info(f"Processing user {user_idx + 1}/{len(data_subset)}: {user_copy.get('sessionId', 'unknown')}")
        
        # Process each rating
        if 'ratings' in user_copy and user_copy['ratings']:
            for rating_idx, rating in enumerate(user_copy['ratings']):
                road_id = rating['road']['id']
                logger.info(f"  Processing road {road_id} ({rating_idx + 1}/{len(user_copy['ratings'])})")
                
                # Get image path
                image_path = get_image_path(road_id)
                
                # Extract features
                features = extract_features_with_qwen(image_path, model, processor, logger)
                
                if features:
                    # Add visual features to rating
                    rating['visual_features'] = features
                    logger.info(f"    Successfully extracted features")
                else:
                    # Use zeros if extraction failed
                    rating['visual_features'] = {feature: 0.0 for feature in VISUAL_FEATURES}
                    logger.warning(f"    Failed to extract features, using zeros")
        
        processed_data.append(user_copy)
    
    # Save GPU-specific results
    gpu_output = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
    with open(gpu_output, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"GPU {gpu_id} completed. Results saved to {gpu_output}")

def merge_gpu_results(output_dir, num_gpus):
    """Merge results from all GPUs"""
    merged_data = []
    
    for gpu_id in range(num_gpus):
        gpu_file = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
        if os.path.exists(gpu_file):
            with open(gpu_file, 'r', encoding='utf-8') as f:
                gpu_data = json.load(f)
            merged_data.extend(gpu_data)
            print(f"  Loaded {len(gpu_data)} users from GPU {gpu_id}")
    
    return merged_data

def process_dataset(input_json, output_json, num_gpus=2, gpu_list=[0, 1]):
    """Process a dataset with visual feature extraction"""
    print(f"\nProcessing {input_json}...")
    
    # Load data
    data = load_existing_data(input_json)
    print(f"Loaded {len(data)} users")
    
    # Create output directory
    dataset_name = os.path.basename(input_json).replace('.json', '')
    output_dir = f"./visual_features_extraction_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing results
    if os.path.exists(output_json):
        response = input(f"\n{output_json} already exists. Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Skipping this dataset")
            return
    
    # Split data for GPUs
    data_per_gpu = len(data) // num_gpus
    gpu_assignments = []
    
    for i in range(num_gpus):
        start_idx = i * data_per_gpu
        if i == num_gpus - 1:
            gpu_data = data[start_idx:]
        else:
            gpu_data = data[start_idx:start_idx + data_per_gpu]
        gpu_assignments.append(gpu_data)
    
    # Create GPU mapping
    gpu_mapping = {i: gpu_list[i] for i in range(num_gpus)}
    
    print(f"\nGPU assignment:")
    for i, gpu_data in enumerate(gpu_assignments):
        print(f"  GPU {i} (actual GPU {gpu_mapping[i]}): {len(gpu_data)} users")
    
    # Start multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    processes = []
    start_time = time.time()
    
    print("\nStarting GPU processes...")
    for gpu_id, gpu_data in enumerate(gpu_assignments):
        p = Process(target=process_on_gpu, args=(gpu_id, gpu_data, output_dir, gpu_mapping))
        p.start()
        processes.append(p)
        print(f"  GPU {gpu_id} process started")
    
    # Wait for completion
    for p in processes:
        p.join()
    
    end_time = time.time()
    print(f"\nAll GPUs finished. Processing time: {end_time - start_time:.2f} seconds")
    
    # Merge results
    print("\nMerging GPU results...")
    merged_data = merge_gpu_results(output_dir, num_gpus)
    
    # Save final result
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved enriched data to {output_json}")
    
    # Statistics
    total_ratings = sum(len(user.get('ratings', [])) for user in merged_data)
    ratings_with_features = sum(
        1 for user in merged_data 
        for rating in user.get('ratings', []) 
        if 'visual_features' in rating
    )
    
    print(f"\nStatistics:")
    print(f"  Total users: {len(merged_data)}")
    print(f"  Total ratings: {total_ratings}")
    print(f"  Ratings with visual features: {ratings_with_features}")
    print(f"  Success rate: {ratings_with_features/total_ratings*100:.1f}%")

def main():
    # Configuration
    num_gpus = 2
    gpu_list = [0, 1]  # Actual GPU IDs to use
    
    print("Visual Feature Extraction using Qwen 2.5-VL")
    print(f"Using {num_gpus} GPUs: {gpu_list}")
    
    # Process training data
    process_dataset(
        input_json="split_train_data.json",
        output_json="split_train_data_with_visual_features.json",
        num_gpus=num_gpus,
        gpu_list=gpu_list
    )
    
    # Process test data
    process_dataset(
        input_json="split_test_data.json",
        output_json="split_test_data_with_visual_features.json",
        num_gpus=num_gpus,
        gpu_list=gpu_list
    )
    
    print("\nFeature extraction completed!")

if __name__ == "__main__":
    main()
