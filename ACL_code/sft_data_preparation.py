#!/usr/bin/env python3
"""
Prepare SFT training data for Qwen VLM LoRA fine-tuning
Generate three types of training data with proper persona balance
SIMPLIFIED VERSION - Concise 50-100 word GPT-4o outputs
FIXED VERSION - Handles empty factors by having GPT-4o select appropriate ones
"""
import json
import os
import openai
from openai import OpenAI
import random
import time
from collections import defaultdict
from datetime import datetime
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import multiprocessing as mp
from multiprocessing import Process, Manager
import re
import base64
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Data type proportions
TYPE1_RATIO = 0.30  # 30% for complete reasoning
TYPE2_RATIO = 0.70  # 70% for structured analysis
TYPE3_RATIO = 0.50  # 50% for direct rating

# Complete list of possible indicators
ALL_INDICATORS = [
   
]

# Persona descriptions
PERSONA_DESCRIPTIONS = {
    "Strong and Fearless": {
        "brief": "comfortable with all types of cycling infrastructure, showing little preference difference between protected and unprotected facilities",
        "detailed": "Confident cyclists who are comfortable riding in traffic and on roads without dedicated bike infrastructure. You prefer direct routes and don't mind sharing the road with vehicles."
    },
    "Enthused and Confident": {
        "brief": "regular cyclists who prefer bike lanes but will ride in mixed traffic when necessary",
        "detailed": "Experienced cyclists who enjoy biking and are generally comfortable on roads, but prefer having some bike infrastructure when available. You can handle traffic but appreciate safer options."
    },
    "No Way No How": {
        "brief": "non-cyclists who are not interested in cycling or find it too dangerous regardless of infrastructure",
        "detailed": "People who do not ride bicycles and have no interest in cycling. You find cycling too dangerous or impractical regardless of the infrastructure available."
    },
    "Interested but Concerned": {
        "brief": "would cycle more if separated from traffic; requires protected infrastructure to feel safe",
        "detailed": "People who would like to bike more but are worried about safety. You need separated bike paths or very quiet neighborhood streets to feel comfortable riding."
    }
}

# Common rating scale description
RATING_SCALE_DESC = """Rate the following on a scale of 1-5:
- Comfortable: How comfortable would you feel cycling here?
- Safe: How safe would you perceive this road segment?
- Overall: Your overall willingness to cycle on this road"""

def load_data_files():
    """Load all necessary data files"""
    logger.info("Loading data files...")
    
    # Load training data
    with open('split_train_data.json', 'r') as f:
        train_data = json.load(f)
    
    # Load OSM data
    with open('locations_info.json', 'r') as f:
        locations_data = json.load(f)
    locations_info = {loc['id']: loc for loc in locations_data['locations']}
    
    # Load bike lane info
    bike_lane_df = pd.read_csv('bike_lane_info.csv')
    bike_lane_info = {}
    for _, row in bike_lane_df.iterrows():
        bike_lane_info[f"loc_{row['index']}"] = row['bike_lane_type']
    
    logger.info(f"Loaded {len(train_data)} training users")
    
    return train_data, locations_info, bike_lane_info

def get_image_path(road_id):
    """Get the correct image path based on road ID"""
    if road_id.startswith('loc_') and '_p' in road_id:
        # AI-enhanced image
        base_loc = road_id.split('_p')[0]
        loc_num = base_loc.replace('loc_', '')
        image_path = os.path.join('AI_images', loc_num, f"{road_id.replace('loc_', '')}.jpg")
    else:
        # Original GSV image
        loc_num = road_id.replace('loc_', '')
        image_path = os.path.join('GSV_images', f"{loc_num}.png")
    
    return image_path

def get_osm_description(road_id, locations_info, bike_lane_info):
    """Get human-readable OSM description"""
    desc_parts = []
    
    # Extract base location ID for OSM lookup
    base_road_id = road_id.split('_p')[0] if '_p' in road_id else road_id
    
    if base_road_id in locations_info:
        loc = locations_info[base_road_id]
        
        # Highway type (road type)
        highway_type = loc.get('highway_type', 'unknown')
        if highway_type and highway_type != 'unknown':
            # Convert OSM highway types to readable format
            road_type_map = {
                'primary': 'primary road',
                'secondary': 'secondary road',
                'tertiary': 'tertiary road',
                'residential': 'residential street',
                'trunk': 'trunk road',
                'motorway': 'motorway',
                'unclassified': 'local road'
            }
            readable_type = road_type_map.get(highway_type, highway_type)
            desc_parts.append(f"Road type: {readable_type}")
        
        # Speed limit
        maxspeed = loc.get('maxspeed')
        if maxspeed and maxspeed != 'unknown':
            desc_parts.append(f"Speed limit: {maxspeed}")
        
        # Lanes
        lanes = loc.get('lanes')
        if lanes and lanes != 'unknown':
            try:
                lanes_num = int(lanes)
                desc_parts.append(f"Number of lanes: {lanes_num}")
            except:
                pass
    
    # Bike lane type
    if base_road_id in bike_lane_info:
        bike_lane_type = bike_lane_info[base_road_id]
        if bike_lane_type and bike_lane_type != 'none':
            desc_parts.append(f"Bike infrastructure: {bike_lane_type}")
    
    return "; ".join(desc_parts) if desc_parts else "No additional road information available"

def encode_image(image_path):
    """Encode image to base64 with preprocessing to match training"""
    # Load and preprocess image to match training
    image = Image.open(image_path).convert('RGB')
    image = image.resize((448, 448), Image.Resampling.LANCZOS)
    
    # Convert to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf-8')

def extract_factors_from_text(text, ratings):
    """Try to extract relevant factors from the reasoning text"""
    text_lower = text.lower()
    extracted = []
    
    # Keywords mapping to factors
    keyword_to_factor = {
        'traffic': ['auto_volume', 'auto_speed'],
        'speed': ['auto_speed'],
        'fast': ['auto_speed'],
        'busy': ['auto_volume'],
        'bike lane': ['bike_lane_quality'],
        'protected': ['separation_moving'],
        'separated': ['separation_moving'],
        'parked cars': ['separation_parked'],
        'parking': ['separation_parked'],
        'wide': ['road_width'],
        'narrow': ['road_width'],
        'pavement': ['pavement_condition'],
        'surface': ['surface_quality'],
        'visibility': ['visibility'],
        'intersection': ['intersection_complexity'],
        'hill': ['hill'],
        'construction': ['construction']
    }
    
    for keyword, factors in keyword_to_factor.items():
        if keyword in text_lower:
            extracted.extend(factors)
    
    # Remove duplicates and limit to 3
    extracted = list(dict.fromkeys(extracted))[:3]
    
    return extracted if extracted else get_fallback_indicators(ratings)

def get_fallback_indicators(ratings):
    """Get fallback indicators based on ratings"""
    avg_rating = (ratings['comfortable'] + ratings['safe'] + ratings['overall']) / 3
    
    if avg_rating >= 4:
        return ["bike_lane_quality", "separation_moving", "road_width"]
    elif avg_rating >= 3:
        return ["auto_volume", "shoulder_width", "pavement_condition"]
    else:
        return ["auto_speed", "auto_volume", "separation_moving"]

def generate_type1_reasoning(client, image_path, osm_desc, persona, persona_desc, ground_truth):
    """Generate Type 1 reasoning based on ground truth indicators and ratings"""
    
    # Check if indicators are empty
    has_indicators = ground_truth['indicators'] and len(ground_truth['indicators']) > 0
    
    # Convert indicators to natural language - complete list
    factor_descriptions = {
        'auto_speed': 'traffic speed',
        'auto_volume': 'traffic volume',
        'bike_lane_quality': 'bike lane quality',
        'bridge': 'bridge crossing',
        'bus_lane_conflict': 'bus lane conflicts',
        'construction': 'construction zones',
        'debris': 'road debris',
        'double_parking': 'double-parked vehicles',
        'driveways': 'driveway crossings',
        'hill': 'steep hills',
        'intersection_complexity': 'complex intersections',
        'merge_area': 'merge areas',
        'multiple_lanes': 'multiple traffic lanes',
        'navigation_difficulty': 'navigation complexity',
        'one_way_street': 'one-way street configuration',
        'pavement_condition': 'pavement condition',
        'separation_moving': 'separation from moving traffic',
        'separation_parked': 'separation from parked cars',
        'shoulder_width': 'shoulder width',
        'single_lane': 'single lane road',
        'surrounding_environment': 'surrounding environment',
        'traffic_signals': 'traffic signals',
        'tunnel': 'tunnel passage',
        'turn_lane': 'turn lanes',
        'visibility': 'visibility conditions',
        'green_bike_lane': 'green painted bike lanes',
        'road_width': 'road width',
        'surface_quality': 'road surface quality'
    }
    
    if has_indicators:
        # Original case: user selected indicators
        natural_factors = [factor_descriptions.get(ind, ind) for ind in ground_truth['indicators']]
        factors_to_mention = ', '.join(natural_factors)
        indicators_to_use = ground_truth['indicators']
    else:
        # Empty indicators case: need GPT-4o to identify factors
        factors_to_mention = None
        indicators_to_use = None
    
    # Unified prompt structure
    base_prompt = f"""You are helping generate training data. A '{persona}' cyclist has evaluated this street with the following results:

PERSONA: {persona} - {persona_desc['detailed']}

THEIR ASSESSMENT:
- Comfort rating: {ground_truth['ratings']['comfortable']}/5
- Safety rating: {ground_truth['ratings']['safe']}/5  
- Overall rating: {ground_truth['ratings']['overall']}/5

STREET INFORMATION:
{osm_desc}"""

    if has_indicators:
        # Add the identified factors
        prompt = base_prompt + f"""
- Key factors they identified: {factors_to_mention}

TASK: Generate a BRIEF (50-100 words) first-person assessment that explains why this '{persona}' cyclist gave these ratings based on the factors they identified. The reasoning should naturally lead to their ratings and mention their key concerns.

Do NOT include structured output - just the reasoning paragraph."""
    else:
        # Ask to identify and use factors
        prompt = base_prompt + f"""

TASK: Looking at this street image and considering the ratings given, generate a BRIEF (50-100 words) first-person assessment that explains why this '{persona}' cyclist gave these ratings. 

In your reasoning, naturally mention 2-3 key factors from this list that best explain the ratings:
{json.dumps(ALL_INDICATORS, indent=2)}

At the end of your response, add a line:
KEY FACTORS: [factor1, factor2, factor3]

Use the exact factor names from the list (e.g., "auto_speed", "bike_lane_quality")."""

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        reasoning_text = response.choices[0].message.content
        
        # Process response to extract factors if needed
        if not has_indicators:
            # Extract the factors from the response
            factors_match = re.search(r'KEY FACTORS:\s*\[(.*?)\]', reasoning_text, re.IGNORECASE | re.DOTALL)
            if factors_match:
                factors_str = factors_match.group(1)
                indicators_to_use = [f.strip().strip('"\'') for f in factors_str.split(',')]
                # Remove the KEY FACTORS line from reasoning
                reasoning_text = re.sub(r'\n*KEY FACTORS:.*?$', '', reasoning_text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE).strip()
            else:
                # Try alternative parsing
                # Look for factors mentioned in brackets anywhere
                bracket_match = re.search(r'\[(.*?)\]', reasoning_text)
                if bracket_match and any(ind in bracket_match.group(1) for ind in ALL_INDICATORS):
                    factors_str = bracket_match.group(1)
                    indicators_to_use = [f.strip().strip('"\'') for f in factors_str.split(',') if f.strip().strip('"\'') in ALL_INDICATORS]
                    # Clean up the reasoning text
                    reasoning_text = reasoning_text.replace(bracket_match.group(0), '').strip()
                
                if not indicators_to_use or len(indicators_to_use) == 0:
                    # Fallback: try to intelligently extract from the text
                    indicators_to_use = extract_factors_from_text(reasoning_text, ground_truth['ratings'])
        
        # Ensure we have valid indicators
        if not indicators_to_use or len(indicators_to_use) == 0:
            # Ultimate fallback based on ratings
            indicators_to_use = get_fallback_indicators(ground_truth['ratings'])
        
        # Construct the complete response
        indicators_str = ", ".join(indicators_to_use)
        complete_response = f"""{reasoning_text}

STRUCTURED OUTPUT:
Factors: [{indicators_str}]
Ratings: comfortable: {ground_truth['ratings']['comfortable']}, safe: {ground_truth['ratings']['safe']}, overall: {ground_truth['ratings']['overall']}"""
        
        return complete_response
        
    except Exception as e:
        logger.error(f"Error generating Type 1 reasoning: {e}")
        return None

def parse_type1_output(output_text):
    """Parse the structured output from Type 1 reasoning"""
    try:
        # Find the structured output section
        structured_match = re.search(r'STRUCTURED OUTPUT:(.*?)$', output_text, re.DOTALL | re.IGNORECASE)
        if not structured_match:
            return None, None
        
        structured_text = structured_match.group(1)
        
        # Extract factors
        factors_match = re.search(r'Factors:\s*\[(.*?)\]', structured_text, re.IGNORECASE)
        if factors_match:
            factors = [f.strip() for f in factors_match.group(1).split(',')]
        else:
            factors = []
        
        # Extract ratings
        ratings = {}
        ratings_match = re.search(r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)', structured_text)
        if ratings_match:
            ratings = {
                'comfortable': int(ratings_match.group(1)),
                'safe': int(ratings_match.group(2)),
                'overall': int(ratings_match.group(3))
            }
        
        return factors, ratings
        
    except Exception as e:
        logger.error(f"Error parsing Type 1 output: {e}")
        return None, None

def create_type1_prompt(persona, persona_desc):
    """Create Type 1 complete reasoning prompt"""
    return f"""As a {persona} cyclist ({persona_desc['detailed']}), analyze this street image for bikeability.

Provide a brief assessment covering:
- Key observations about the street
- Factors affecting your cycling experience
- Your comfort and safety evaluation

{RATING_SCALE_DESC}

End with:
STRUCTURED OUTPUT:
Factors: [list specific factors like "protected bike lane", "heavy traffic", "narrow road"]
Ratings: comfortable: X, safe: Y, overall: Z"""

def create_type2_prompt(persona, persona_desc):
    """Create Type 2 structured analysis prompt"""
    return f"""As a {persona} cyclist ({persona_desc['brief']}), assess this street for bikeability.

Identify the most important factors affecting bikeability for someone with your cycling preferences, then rate the street.

Format your response as:
Factors: [list key factors]
Ratings: comfortable: X, safe: Y, overall: Z

Use a 1-5 scale for ratings."""

def create_type3_prompt(persona, persona_desc):
    """Create Type 3 direct rating prompt"""
    return f"""As a {persona} cyclist ({persona_desc['brief']}), rate this street's bikeability.

Provide ratings (1-5 scale):
Ratings: comfortable: X, safe: Y, overall: Z"""

def prepare_sample_for_sft(sample_data, sample_type, type1_reasoning=None):
    """Prepare a single sample in SFT format"""
    image_path = sample_data['image_path']
    persona = sample_data['persona']
    persona_desc = PERSONA_DESCRIPTIONS[persona]
    
    # Create prompts based on type
    if sample_type == 1:
        prompt = create_type1_prompt(persona, persona_desc)
        response = type1_reasoning if type1_reasoning else "Error: No reasoning generated"
        
    elif sample_type == 2:
        prompt = create_type2_prompt(persona, persona_desc)
        # For Type 2, check if we have indicators or need to use GPT-4o generated ones
        if sample_data.get('gpt4o_indicators'):
            # Use GPT-4o generated indicators
            indicators_str = ", ".join(sample_data['gpt4o_indicators'])
        else:
            # Use original indicators
            indicators_str = ", ".join(sample_data['ground_truth']['indicators']) if sample_data['ground_truth']['indicators'] else "general road conditions"
        ratings = sample_data['ground_truth']['ratings']
        response = f"Factors: [{indicators_str}]\nRatings: comfortable: {ratings['comfortable']}, safe: {ratings['safe']}, overall: {ratings['overall']}"
        
    else:  # Type 3
        prompt = create_type3_prompt(persona, persona_desc)
        ratings = sample_data['ground_truth']['ratings']
        response = f"Ratings: comfortable: {ratings['comfortable']}, safe: {ratings['safe']}, overall: {ratings['overall']}"
    
    # Add OSM information to prompt if available
    osm_desc = sample_data['osm_description']
    if osm_desc != "No additional road information available":
        prompt += f"\n\nAdditional road information: {osm_desc}"
    
    # Create SFT format
    sft_sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image_path}
                ]
            },
            {
                "role": "assistant",
                "content": response
            }
        ],
        "metadata": {
            "sample_type": sample_type,
            "persona": persona,
            "road_id": sample_data['road_id'],
            "session_id": sample_data['session_id']
        }
    }
    
    return sft_sample

def process_persona_batch(persona, persona_samples, client, output_dir):
    """Process all samples for a specific persona"""
    logger.info(f"Processing {persona}: {len(persona_samples)} samples")
    
    # Calculate sample sizes for each type
    n_samples = len(persona_samples)
    n_type1 = int(n_samples * TYPE1_RATIO)
    n_type2 = int(n_samples * TYPE2_RATIO)
    n_type3 = n_samples  # Use all samples for Type 3 since overlap is allowed
    
    logger.info(f"  Type 1: {n_type1} samples")
    logger.info(f"  Type 2: {n_type2} samples") 
    logger.info(f"  Type 3: {n_type3} samples")
    
    # Randomly shuffle and select samples
    random.shuffle(persona_samples)
    
    type1_samples = persona_samples[:n_type1]
    type2_samples = persona_samples[:n_type2]
    type3_samples = persona_samples  # All samples for type 3
    
    sft_samples = []
    
    # Process Type 1 samples (with GPT-4o generation)
    logger.info(f"  Generating Type 1 reasoning for {persona}...")
    for sample in tqdm(type1_samples, desc=f"{persona} Type 1"):
        # Generate reasoning with GPT-4o
        reasoning = generate_type1_reasoning(
            client,
            sample['image_path'],
            sample['osm_description'],
            persona,
            PERSONA_DESCRIPTIONS[persona],
            sample['ground_truth']
        )
        
        if reasoning:
            # Parse to verify structured output
            factors, ratings = parse_type1_output(reasoning)
            if factors is not None and ratings is not None:
                # If original indicators were empty, store GPT-4o generated factors
                if not sample['ground_truth']['indicators']:
                    sample['gpt4o_indicators'] = factors
                sft_sample = prepare_sample_for_sft(sample, 1, reasoning)
                sft_samples.append(sft_sample)
            else:
                logger.warning(f"Failed to parse Type 1 output for {sample['road_id']}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Process Type 2 samples (structured analysis)
    logger.info(f"  Processing Type 2 samples for {persona}...")
    for sample in type2_samples:
        # Check if this sample was processed in Type 1 and has GPT-4o indicators
        if not sample['ground_truth']['indicators']:
            # Find if we have GPT-4o generated indicators for this sample
            for type1_sample in type1_samples:
                if type1_sample['road_id'] == sample['road_id'] and 'gpt4o_indicators' in type1_sample:
                    sample['gpt4o_indicators'] = type1_sample['gpt4o_indicators']
                    break
        
        sft_sample = prepare_sample_for_sft(sample, 2)
        sft_samples.append(sft_sample)
    
    # Process Type 3 samples (direct rating)
    logger.info(f"  Processing Type 3 samples for {persona}...")
    for sample in type3_samples:
        sft_sample = prepare_sample_for_sft(sample, 3)
        sft_samples.append(sft_sample)
    
    # Save persona-specific results
    persona_output = os.path.join(output_dir, f"sft_{persona.lower().replace(' ', '_')}.json")
    with open(persona_output, 'w') as f:
        json.dump(sft_samples, f, indent=2)
    
    logger.info(f"  Saved {len(sft_samples)} samples for {persona}")
    
    return sft_samples

def process_and_save_persona(persona, samples, api_key, base_url, output_dir):
    """Helper function for processing persona in separate process"""
    # Setup client for this process
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Process the persona
    process_persona_batch(persona, samples, client, output_dir)

def main():
    # Create output directory
    output_dir = "./sft_training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup OpenAI client for main process
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Load data
    train_data, locations_info, bike_lane_info = load_data_files()
    
    # Organize samples by persona
    persona_samples = defaultdict(list)
    
    logger.info("Organizing samples by persona...")
    empty_indicators_count = 0
    for user in train_data:
        persona = user.get('geller_classification', 'Unknown')
        if persona == 'Unknown' or persona not in PERSONA_DESCRIPTIONS:
            continue
        
        for rating in user.get('ratings', []):
            road_id = rating['road']['id']
            
            # Check if image exists
            image_path = get_image_path(road_id)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Check if indicators are empty
            if not rating.get('indicators', []):
                empty_indicators_count += 1
            
            # Prepare sample data
            sample = {
                'road_id': road_id,
                'session_id': user['sessionId'],
                'persona': persona,
                'image_path': image_path,
                'osm_description': get_osm_description(road_id, locations_info, bike_lane_info),
                'ground_truth': {
                    'ratings': rating['ratings'],
                    'indicators': rating.get('indicators', [])
                }
            }
            
            persona_samples[persona].append(sample)
    
    # Display distribution
    logger.info("\nPersona distribution:")
    total_samples = 0
    for persona, samples in persona_samples.items():
        logger.info(f"  {persona}: {len(samples)} samples")
        total_samples += len(samples)
    logger.info(f"  Total: {total_samples} samples")
    logger.info(f"  Samples with empty indicators: {empty_indicators_count} ({empty_indicators_count/total_samples*100:.1f}%)")
    
    # Separate "Interested but Concerned" from others
    interested_but_concerned_samples = None
    other_personas = []
    
    for persona, samples in persona_samples.items():
        if persona == "Interested but Concerned":
            interested_but_concerned_samples = (persona, samples)
        else:
            other_personas.append((persona, samples))
    
    all_sft_samples = []
    
    # Process personas concurrently
    logger.info("\n" + "="*60)
    logger.info("Starting concurrent processing...")
    logger.info("="*60)
    
    if interested_but_concerned_samples:
        # Start process for "Interested but Concerned"
        logger.info("\nStarting separate process for 'Interested but Concerned'...")
        ibc_process = Process(
            target=lambda: process_and_save_persona(
                interested_but_concerned_samples[0],
                interested_but_concerned_samples[1],
                API_KEY,
                BASE_URL,
                output_dir
            )
        )
        ibc_process.start()
    
    # Process other personas in main process
    logger.info("\nProcessing other personas in main process...")
    other_results = []
    for persona, samples in other_personas:
        logger.info(f"\nProcessing {persona}...")
        persona_sft_samples = process_persona_batch(persona, samples, client, output_dir)
        other_results.extend(persona_sft_samples)
        all_sft_samples.extend(persona_sft_samples)
    
    # Wait for "Interested but Concerned" process to complete
    if interested_but_concerned_samples:
        logger.info("\nWaiting for 'Interested but Concerned' process to complete...")
        ibc_process.join()
        
        # Load results from the separate process
        ibc_output_file = os.path.join(output_dir, "sft_interested_but_concerned.json")
        if os.path.exists(ibc_output_file):
            with open(ibc_output_file, 'r') as f:
                ibc_samples = json.load(f)
            all_sft_samples.extend(ibc_samples)
            logger.info(f"Loaded {len(ibc_samples)} samples from 'Interested but Concerned' process")
    
    # Shuffle all samples
    random.shuffle(all_sft_samples)
    
    # Save complete training set
    output_file = os.path.join(output_dir, "sft_training_complete.json")
    with open(output_file, 'w') as f:
        json.dump(all_sft_samples, f, indent=2)
    
    # Generate statistics
    stats = {
        "total_samples": len(all_sft_samples),
        "by_type": {
            "type1": sum(1 for s in all_sft_samples if s['metadata']['sample_type'] == 1),
            "type2": sum(1 for s in all_sft_samples if s['metadata']['sample_type'] == 2),
            "type3": sum(1 for s in all_sft_samples if s['metadata']['sample_type'] == 3)
        },
        "by_persona": {}
    }
    
    for persona in PERSONA_DESCRIPTIONS.keys():
        persona_count = sum(1 for s in all_sft_samples if s['metadata']['persona'] == persona)
        stats["by_persona"][persona] = persona_count
    
    # Save statistics
    stats_file = os.path.join(output_dir, "sft_training_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SFT TRAINING DATA GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples generated: {stats['total_samples']}")
    logger.info("\nBy type:")
    for t, count in stats['by_type'].items():
        logger.info(f"  {t}: {count} samples ({count/stats['total_samples']*100:.1f}%)")
    logger.info("\nBy persona:")
    for p, count in stats['by_persona'].items():
        logger.info(f"  {p}: {count} samples")
    logger.info(f"\nOutput files:")
    logger.info(f"  - {output_file}")
    logger.info(f"  - {stats_file}")

if __name__ == "__main__":
    main()