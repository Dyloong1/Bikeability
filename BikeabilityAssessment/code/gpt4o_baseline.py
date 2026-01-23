import json
import os
import base64
import openai
from openai import OpenAI
import pandas as pd
from collections import defaultdict
import time
from tqdm import tqdm
from PIL import Image
import io

# Persona descriptions
PERSONA_DESCRIPTIONS = {
    "Strong and Fearless": "Comfortable with all types of cycling infrastructure, showing little preference difference between protected and unprotected facilities",
    "Enthused and Confident": "Regular cyclists who prefer bike lanes but will ride in mixed traffic when necessary",
    "Interested but Concerned": "Would cycle more if separated from traffic; requires protected infrastructure to feel safe",
    "No Way No How": "Unlikely to cycle regardless of infrastructure improvements"
}

class GPT4oBikeabilityBaseline:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", model="gpt-4o", temperature=0.3):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.predictions_cache = defaultdict(dict)  # Cache to avoid duplicate predictions
        
    def encode_image(self, image_path, max_size=(768, 768), quality=85):
        """Encode and resize image to base64 for API"""
        # Open and resize image
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Resize if larger than max_size
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            # Encode to base64
            return base64.b64encode(buffer.read()).decode('utf-8')
    
    def load_osm_data(self, locations_file, bike_lane_file):
        """Load OSM data from files"""
        # Load locations info
        with open(locations_file, 'r') as f:
            locations_data = json.load(f)
        
        # Create a dictionary for quick lookup
        self.locations_info = {}
        for loc in locations_data['locations']:
            self.locations_info[loc['id']] = loc
        
        # Load bike lane info
        bike_lane_df = pd.read_csv(bike_lane_file)
        self.bike_lane_info = {}
        for _, row in bike_lane_df.iterrows():
            self.bike_lane_info[f"loc_{row['index']}"] = row['bike_lane_type']
    
    def get_osm_attributes(self, location_id):
        """Get OSM attributes for a location, excluding null values"""
        if location_id not in self.locations_info:
            return {}
        
        loc_data = self.locations_info[location_id]
        attributes = {}
        
        # Add non-null attributes
        osm_fields = ['roadType', 'surface', 'maxspeed', 'hasCycleway', 'lanes', 'width']
        for field in osm_fields:
            if field in loc_data and loc_data[field] not in [None, 'unknown', 'nan']:
                attributes[field] = loc_data[field]
        
        # Add bike lane type if available
        if location_id in self.bike_lane_info:
            attributes['bikeLaneType'] = self.bike_lane_info[location_id]
        
        return attributes
    
    def create_prompt(self, persona, osm_attributes):
        """Create the prompt for GPT-4o"""
        persona_desc = PERSONA_DESCRIPTIONS.get(persona, "Unknown persona type")
        
        # Format OSM attributes
        osm_text = "OSM Attributes:\n"
        if osm_attributes:
            for key, value in osm_attributes.items():
                osm_text += f"- {key}: {value}\n"
        else:
            osm_text += "- No additional OSM data available\n"
        
        prompt = """You are an expert in urban cycling infrastructure assessment. Analyze the provided street view image and assess its bikeability from the perspective of a specific cyclist persona.

Cyclist Persona: {persona}
Persona Description: {persona_desc}

{osm_text}

Task: Perform a bikeability assessment following these steps:

1. First, analyze the street environment shown in the image and the OSM attributes. Describe:
   - Road characteristics (width, lanes, traffic volume appearance)
   - Cycling infrastructure present (if any)
   - Safety concerns visible
   - Environmental factors affecting cycling

2. Consider how a "{persona}" cyclist would perceive this environment based on their characteristics. Think about:
   - What factors would most influence their comfort level?
   - What specific concerns would they have?
   - How would the infrastructure match their needs?

3. Provide ratings on a scale of 1-4 (1=worst, 4=best) for:
   - Comfortable: How comfortable would this persona feel cycling here?
   - Safe: How safe would they perceive this road segment?
   - Overall: Their overall willingness to cycle on this road

4. After your analysis, provide a JSON output with:
   - "influencing_factors": list of key factors affecting the ratings (e.g., "high_traffic", "no_bike_lane", "wide_road", "protected_lane", etc.)
   - "ratings": object with "comfortable", "safe", and "overall" scores (1-4)

Please think step by step and provide detailed reasoning before the final JSON output.

End your response with the JSON output in this exact format:
```json
{{
  "influencing_factors": ["factor1", "factor2", ...],
  "ratings": {{
    "comfortable": X,
    "safe": X,
    "overall": X
  }}
}}
```""".format(persona=persona, persona_desc=persona_desc, osm_text=osm_text)
        
        return prompt
    
    def predict_single(self, image_path, location_id, persona):
        """Make prediction for a single image"""
        # Check cache first
        cache_key = f"{location_id}_{persona}"
        if cache_key in self.predictions_cache:
            return self.predictions_cache[cache_key]
        
        # Get OSM attributes
        osm_attributes = self.get_osm_attributes(location_id)
        
        # Create prompt
        prompt = self.create_prompt(persona, osm_attributes)
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Call GPT-4o
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.temperature,
                max_tokens=1500
            )
            
            # Extract response
            full_response = response.choices[0].message.content
            
            # Extract JSON from response
            json_start = full_response.rfind('```json')
            json_end = full_response.rfind('```')
            if json_start != -1 and json_end > json_start:
                json_str = full_response[json_start+7:json_end].strip()
                result = json.loads(json_str)
            else:
                # Try to find JSON directly
                import re
                json_match = re.search(r'\{[^{}]*"influencing_factors"[^{}]*\}', full_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not extract JSON from response")
            
            # Add full response for logging
            result['full_response'] = full_response
            result['location_id'] = location_id
            result['persona'] = persona
            
            # Cache the result
            self.predictions_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Error processing {location_id} for {persona}: {str(e)}")
            return None
    
    def process_test_data(self, test_data_file, gsv_images_dir, ai_images_dir, output_file):
        """Process all test data and generate predictions"""
        # Load test data
        with open(test_data_file, 'r') as f:
            test_data = json.load(f)
        
        # First, collect all unique location-persona combinations
        unique_predictions = []
        processed = set()
        
        print("Collecting unique location-persona combinations...")
        for user_data in test_data:
            persona = user_data.get('geller_classification')
            if not persona:
                continue
            
            ratings = user_data.get('ratings', [])
            
            for rating in ratings:
                road_id = rating['road']['id']
                pred_key = f"{road_id}_{persona}"
                
                if pred_key not in processed:
                    processed.add(pred_key)
                    unique_predictions.append({
                        'road_id': road_id,
                        'persona': persona,
                        'rating': rating,
                        'user_session': user_data['sessionId']
                    })
        
        print(f"Found {len(unique_predictions)} unique location-persona combinations to process")
        
        results = []
        
        # Process each unique prediction
        for pred_data in tqdm(unique_predictions, desc="Processing predictions"):
            road_id = pred_data['road_id']
            persona = pred_data['persona']
            rating = pred_data['rating']
            
            # Determine image path
            if road_id.startswith('loc_') and '_p' in road_id:
                # AI-enhanced image
                base_loc = road_id.split('_p')[0]
                loc_num = base_loc.replace('loc_', '')
                image_path = os.path.join(ai_images_dir, loc_num, f"{road_id.replace('loc_', '')}.jpg")
            else:
                # Original GSV image
                loc_num = road_id.replace('loc_', '')
                image_path = os.path.join(gsv_images_dir, f"{loc_num}.png")
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"\nWarning: Image not found: {image_path}")
                continue
            
            # Get prediction
            prediction = self.predict_single(image_path, road_id, persona)
            
            if prediction:
                # Add ground truth for comparison
                prediction['ground_truth'] = {
                    'comfortable': rating['ratings']['comfortable'],
                    'safe': rating['ratings']['safe'],
                    'overall': rating['ratings']['overall']
                }
                prediction['ground_truth_indicators'] = rating.get('indicators', [])
                prediction['user_session'] = pred_data['user_session']
                
                results.append(prediction)
            
            # Rate limiting
            time.sleep(0.5)  # Adjust based on your API limits
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessed {len(results)} unique location-persona combinations")
        print(f"Results saved to: {output_file}")
        
        return results
    
    def evaluate_results(self, results_file):
        """Evaluate prediction results"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Calculate metrics
        metrics = {
            'comfortable': {'errors': [], 'exact_matches': 0},
            'safe': {'errors': [], 'exact_matches': 0},
            'overall': {'errors': [], 'exact_matches': 0}
        }
        
        persona_metrics = defaultdict(lambda: {
            'count': 0,
            'comfortable_mae': 0,
            'safe_mae': 0,
            'overall_mae': 0
        })
        
        for result in results:
            pred_ratings = result['ratings']
            true_ratings = result['ground_truth']
            persona = result['persona']
            
            persona_metrics[persona]['count'] += 1
            
            for metric in ['comfortable', 'safe', 'overall']:
                error = abs(pred_ratings[metric] - true_ratings[metric])
                metrics[metric]['errors'].append(error)
                
                if error == 0:
                    metrics[metric]['exact_matches'] += 1
                
                persona_metrics[persona][f'{metric}_mae'] += error
        
        # Calculate overall metrics
        print("=" * 60)
        print("GPT-4O ZERO-SHOT BASELINE EVALUATION")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print("-" * 40)
        for metric in ['comfortable', 'safe', 'overall']:
            errors = metrics[metric]['errors']
            mae = sum(errors) / len(errors)
            exact_match_rate = metrics[metric]['exact_matches'] / len(errors)
            
            print(f"{metric.capitalize():>12}: MAE={mae:.3f}, Exact Match={exact_match_rate:.1%}")
        
        print("\nMetrics by Persona:")
        print("-" * 40)
        for persona, stats in sorted(persona_metrics.items()):
            count = stats['count']
            print(f"\n{persona} (n={count}):")
            for metric in ['comfortable', 'safe', 'overall']:
                mae = stats[f'{metric}_mae'] / count
                print(f"  {metric:>10}: MAE={mae:.3f}")
        
        return metrics, persona_metrics


# Main execution
if __name__ == "__main__":
    # Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Initialize baseline model
    baseline = GPT4oBikeabilityBaseline(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="gpt-4o"  # Using GPT-4o model
    )
    
    # Load OSM data
    baseline.load_osm_data('locations_info.json', 'bike_lane_info.csv')
    
    # Process test data
    results = baseline.process_test_data(
        test_data_file='split_test_data.json',
        gsv_images_dir='GSV_images',
        ai_images_dir='AI_images',
        output_file='gpt4o_baseline_predictions.json'
    )
    
    # Evaluate results
    if os.path.exists('gpt4o_baseline_predictions.json'):
        baseline.evaluate_results('gpt4o_baseline_predictions.json')