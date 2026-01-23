#!/usr/bin/env python3
"""
Standalone prediction script for Qwen-VL LoRA model
Tests both Type 1 and Type 2 prompts on test data
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import logging
from tqdm import tqdm
import re
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Persona descriptions
PERSONA_DESCRIPTIONS = {
    "Strong and Fearless": "Confident cyclists who are comfortable riding in traffic and on roads without dedicated bike infrastructure. You prefer direct routes and don't mind sharing the road with vehicles.",
    "Enthused and Confident": "Experienced cyclists who enjoy biking and are generally comfortable on roads, but prefer having some bike infrastructure when available. You can handle traffic but appreciate safer options.",
    "Comfortable but Cautious": "Casual cyclists who enjoy biking but strongly prefer protected bike lanes or quiet streets. You avoid heavy traffic and need to feel safe while riding.",
    "Interested but Concerned": "People who would like to bike more but are worried about safety. You need separated bike paths or very quiet neighborhood streets to feel comfortable riding."
}

def create_type1_prompt(persona, persona_desc):
    """Create Type 1 prompt: Analysis + Ratings (matching training format)"""
    prompt = f"""As a {persona} cyclist ({persona_desc}), analyze this street image for bikeability.

First, describe what you observe about:
1. Road infrastructure and bike facilities
2. Traffic conditions and safety concerns  
3. Comfort factors for your cycling profile

Then provide ratings (1-5 scale):
- Comfortable: How comfortable would you feel biking here?
- Safe: How safe would you feel biking here?
- Overall: Your overall bikeability rating

End with:
STRUCTURED OUTPUT:
Factors: [list key factors affecting your ratings]
Ratings: comfortable: X, safe: Y, overall: Z"""
    return prompt

def create_type2_prompt(persona, persona_desc):
    """Create Type 2 prompt: Factors + Ratings (matching training format)"""
    prompt = f"""As a {persona} cyclist ({persona_desc}), assess this street for bikeability.

Identify the most important factors affecting bikeability for someone with your cycling preferences, then rate the street.

Format your response as:
Factors: [list key factors]
Ratings: comfortable: X, safe: Y, overall: Z

Use a 1-5 scale for ratings."""
    return prompt

def get_image_path(road_id):
    """Get image path from road ID"""
    # Parse road_id to determine image type and location
    # Format: loc_123 (original) or loc_123_p1 (AI processed)
    
    if '_p' in road_id:
        # AI processed image: loc_123_p1 -> AI_images/123/123_p1.jpg
        parts = road_id.split('_')
        loc_num = parts[1]  # Extract '123' from 'loc_123_p1'
        filename = f"{loc_num}_{parts[2]}.jpg"  # '123_p1.jpg'
        return f"./AI_images/{loc_num}/{filename}"
    else:
        # Original image: loc_123 -> GSV_images/123.png
        loc_num = road_id.split('_')[1]  # Extract '123' from 'loc_123'
        return f"./GSV_images/{loc_num}.png"

def get_osm_description(road_id, locations_info, bike_lane_info):
    """Get OSM description for a road"""
    if road_id in locations_info:
        loc_info = locations_info[road_id]
        desc_parts = []
        
        # Add road type info
        if 'highway_type' in loc_info:
            desc_parts.append(f"Road type: {loc_info['highway_type']}")
        
        # Add bike lane info
        if road_id in bike_lane_info:
            desc_parts.append(f"Bike infrastructure: {bike_lane_info[road_id]}")
        
        # Add other relevant info
        if 'lanes' in loc_info:
            desc_parts.append(f"Number of lanes: {loc_info['lanes']}")
        
        if 'maxspeed' in loc_info:
            desc_parts.append(f"Speed limit: {loc_info['maxspeed']}")
        
        return "; ".join(desc_parts) if desc_parts else "No additional road information available"
    
    return "No additional road information available"

def parse_model_output(output_text, prompt_type):
    """Parse model output based on prompt type"""
    try:
        if prompt_type == 2:
            # Type 2: Factors and ratings
            factors_pattern = r'Factors:\s*\[(.*?)\]'
            ratings_pattern = r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)'
            
            factors_match = re.search(factors_pattern, output_text, re.IGNORECASE)
            ratings_match = re.search(ratings_pattern, output_text)
            
            factors = []
            if factors_match:
                factors_text = factors_match.group(1)
                factors = [f.strip() for f in factors_text.split(',')]
            
            ratings = {}
            if ratings_match:
                ratings = {
                    'comfortable': int(ratings_match.group(1)),
                    'safe': int(ratings_match.group(2)),
                    'overall': int(ratings_match.group(3))
                }
            
            return ratings, factors
        
        else:  # Type 1
            # Look for structured output section
            structured_pattern = r'STRUCTURED OUTPUT:\s*(.*?)$'
            structured_match = re.search(structured_pattern, output_text, re.DOTALL | re.IGNORECASE)
            
            if structured_match:
                structured_text = structured_match.group(1)
                
                # Extract factors and ratings from structured section
                factors_pattern = r'Factors:\s*\[(.*?)\]'
                ratings_pattern = r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)'
                
                factors_match = re.search(factors_pattern, structured_text, re.IGNORECASE)
                ratings_match = re.search(ratings_pattern, structured_text)
                
                factors = []
                if factors_match:
                    factors_text = factors_match.group(1)
                    factors = [f.strip() for f in factors_text.split(',')]
                
                ratings = {}
                if ratings_match:
                    ratings = {
                        'comfortable': int(ratings_match.group(1)),
                        'safe': int(ratings_match.group(2)),
                        'overall': int(ratings_match.group(3))
                    }
                
                return ratings, factors
            else:
                # If no structured output section, try to find ratings in the whole text
                ratings_pattern = r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)'
                ratings_match = re.search(ratings_pattern, output_text)
                
                ratings = {}
                if ratings_match:
                    ratings = {
                        'comfortable': int(ratings_match.group(1)),
                        'safe': int(ratings_match.group(2)),
                        'overall': int(ratings_match.group(3))
                    }
                
                return ratings, []
    
    except Exception as e:
        logger.error(f"Error parsing output for prompt type {prompt_type}: {e}")
        logger.debug(f"Output text was: {output_text[:200]}...")
    
    return {}, []

def load_model_and_processor(model_path):
    """Load the fine-tuned model and processor"""
    logger.info(f"Loading model from {model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Load base model
    base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, processor

def test_single_sample(model, processor, prompt_type=1):
    """Test model with a single sample to ensure it's working"""
    logger.info(f"Testing model with prompt type {prompt_type}...")
    
    # Create a dummy image
    test_image = Image.new('RGB', (448, 448), color='gray')
    
    # Create test prompt
    persona = "Comfortable but Cautious"
    persona_desc = PERSONA_DESCRIPTIONS[persona]
    
    if prompt_type == 1:
        prompt = create_type1_prompt(persona, persona_desc)
    else:  # Type 2
        prompt = create_type2_prompt(persona, persona_desc)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": test_image}
            ]
        }
    ]
    
    try:
        # Format for model
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
            padding=False,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
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
        
        logger.info(f"Test output: {output_text[:200]}...")
        logger.info("Model test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        logger.exception("Full traceback:")
        return False

def predict_on_test_data(model, processor, test_data_path, locations_info, bike_lane_info, prompt_type, output_suffix):
    """Make predictions on test data"""
    logger.info(f"\nMaking predictions with Type {prompt_type} prompts")
    logger.info(f"Loading test data from: {test_data_path}")
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test users")
    
    # Count total ratings first
    total_ratings = 0
    valid_users = []
    for user in test_data:
        persona = user.get('geller_classification', 'Unknown')
        if persona != 'Unknown' and persona in PERSONA_DESCRIPTIONS:
            valid_users.append(user)
            total_ratings += len(user.get('ratings', []))
    
    logger.info(f"Total ratings to process: {total_ratings} from {len(valid_users)} valid users")
    
    predictions = []
    processed_ratings = 0
    skipped_ratings = 0
    
    # Create progress bar for ratings
    pbar = tqdm(total=total_ratings, desc=f"Type {prompt_type} predictions")
    
    # Process each test user
    for user_idx, user in enumerate(valid_users):
        session_id = user.get('sessionId', f'user_{user_idx}')
        persona = user.get('geller_classification')
        persona_desc = PERSONA_DESCRIPTIONS[persona]
        ratings = user.get('ratings', [])
        
        for rating in ratings:
            try:
                road_id = rating['road']['id']
                image_path = get_image_path(road_id)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    skipped_ratings += 1
                    pbar.update(1)
                    continue
                
                # Load and preprocess image
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((448, 448), Image.Resampling.LANCZOS)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}")
                    skipped_ratings += 1
                    pbar.update(1)
                    continue
                
                # Get OSM description
                osm_desc = get_osm_description(road_id, locations_info, bike_lane_info)
                
                # Create prompt
                if prompt_type == 1:
                    prompt = create_type1_prompt(persona, persona_desc)
                elif prompt_type == 2:
                    prompt = create_type2_prompt(persona, persona_desc)
                
                # Add OSM info if available
                if osm_desc != "No additional road information available":
                    prompt += f"\n\nAdditional road information: {osm_desc}"
                
                # Prepare messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image}
                        ]
                    }
                ]
                
                # Format for model
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
                    padding=False,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=True
                ).to(model.device)
                
                # Generate
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
                
                # Parse output
                pred_ratings, pred_factors = parse_model_output(output_text, prompt_type)
                
                # Store prediction
                prediction = {
                    'session_id': session_id,
                    'persona': persona,
                    'road_id': road_id,
                    'prompt_type': prompt_type,
                    'ground_truth': {
                        'ratings': rating['ratings'],
                        'indicators': rating.get('indicators', [])
                    },
                    'predictions': {
                        'ratings': pred_ratings,
                        'factors': pred_factors
                    },
                    'raw_output': output_text
                }
                
                predictions.append(prediction)
                processed_ratings += 1
                
                # Save predictions incrementally every 10 predictions
                if len(predictions) % 10 == 0:
                    temp_output_file = f"qwen_vl_predictions_type{prompt_type}_{output_suffix}_temp.json"
                    with open(temp_output_file, 'w') as f:
                        json.dump(predictions, f, indent=2)
                    logger.debug(f"Saved {len(predictions)} predictions to temporary file")
                
            except Exception as e:
                logger.error(f"Error processing road {road_id} for user {session_id}: {str(e)}")
                skipped_ratings += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Log progress every 10 ratings
            if processed_ratings % 10 == 0 and processed_ratings > 0:
                pbar.set_postfix({
                    'processed': processed_ratings,
                    'skipped': skipped_ratings,
                    'success_rate': f'{processed_ratings/(processed_ratings+skipped_ratings)*100:.1f}%'
                })
    
    pbar.close()
    
    # Save final predictions
    output_file = f"qwen_vl_predictions_type{prompt_type}_{output_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Remove temporary file if it exists
    temp_output_file = f"qwen_vl_predictions_type{prompt_type}_{output_suffix}_temp.json"
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
        logger.debug("Removed temporary file")
    
    logger.info(f"\nPrediction summary:")
    logger.info(f"  Total ratings: {total_ratings}")
    logger.info(f"  Processed ratings: {processed_ratings}")
    logger.info(f"  Skipped ratings: {skipped_ratings}")
    logger.info(f"  Success rate: {processed_ratings/(processed_ratings+skipped_ratings)*100:.1f}%")
    logger.info(f"  Saved to: {output_file}")
    
    return predictions

def main():
    # Configuration
    model_path = "./qwen_vl_lora_output/final_model"
    test_data_path = "split_test_data.json"
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    # Check if image directories exist
    if not os.path.exists("GSV_images"):
        logger.error("GSV_images directory not found!")
        return
    if not os.path.exists("AI_images"):
        logger.error("AI_images directory not found!")
        return
    
    logger.info("Found image directories:")
    logger.info(f"  GSV_images: {len(os.listdir('GSV_images'))} files")
    logger.info(f"  AI_images: {len(os.listdir('AI_images'))} directories")
    
    # Load model and processor
    model, processor = load_model_and_processor(model_path)
    
    # Test model first
    logger.info("\n" + "="*60)
    logger.info("Testing model functionality")
    logger.info("="*60)
    
    for test_type in [1, 2]:  # Only test Type 1 and Type 2
        if not test_single_sample(model, processor, prompt_type=test_type):
            logger.error(f"Model test failed for Type {test_type}. Exiting.")
            return
    
    # Load OSM data
    logger.info("\nLoading OSM data...")
    
    # Load locations info
    if os.path.exists('locations_info.json'):
        with open('locations_info.json', 'r') as f:
            locations_data = json.load(f)
        locations_info = {loc['id']: loc for loc in locations_data.get('locations', [])}
        logger.info(f"Loaded {len(locations_info)} location entries")
    else:
        logger.warning("locations_info.json not found")
        locations_info = {}
    
    # Load bike lane info
    if os.path.exists('bike_lane_info.csv'):
        bike_lane_df = pd.read_csv('bike_lane_info.csv')
        bike_lane_info = {}
        for _, row in bike_lane_df.iterrows():
            bike_lane_info[f"loc_{row['index']}"] = row['bike_lane_type']
        logger.info(f"Loaded {len(bike_lane_info)} bike lane entries")
    else:
        logger.warning("bike_lane_info.csv not found")
        bike_lane_info = {}
    
    # Get timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test Type 1 and Type 2 prompts
    for prompt_type in [1, 2]:  # Only Type 1 and Type 2
        logger.info("\n" + "="*60)
        logger.info(f"PREDICTIONS WITH TYPE {prompt_type} PROMPTS")
        if prompt_type == 1:
            logger.info("(Analysis + Ratings)")
        else:  # Type 2
            logger.info("(Factors + Ratings)")
        logger.info("="*60)
        
        predictions = predict_on_test_data(
            model, processor, test_data_path, 
            locations_info, bike_lane_info, 
            prompt_type=prompt_type, output_suffix=timestamp
        )
    
    logger.info("\n" + "="*60)
    logger.info("PREDICTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()