#!/usr/bin/env python3
"""
Fine-tune Qwen-VL with LoRA using progressive training strategy

"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from qwen_vl_utils import process_vision_info
import logging
from datetime import datetime
from tqdm import tqdm
import random
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
import re
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

class BikeabilityDataset(Dataset):
    """Custom dataset for bikeability assessment"""
    
    def __init__(self, data_path, processor, max_length=2048, data_subset=None, image_size=(448, 448)):
        """
        Args:
            data_path: Path to the SFT format data file
            processor: Qwen processor
            max_length: Maximum sequence length
            data_subset: Optional list of sample types to include (e.g., [1, 2, 3])
            image_size: Target image size (width, height)
        """
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter by sample type if specified
        if data_subset is not None:
            self.data = [
                sample for sample in self.data 
                if sample['metadata']['sample_type'] in data_subset
            ]
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        # Count sample types
        type_counts = defaultdict(int)
        for sample in self.data:
            type_counts[sample['metadata']['sample_type']] += 1
        
        logger.info("Sample distribution:")
        for t, count in sorted(type_counts.items()):
            logger.info(f"  Type {t}: {count} samples")
    
    def preprocess_image(self, image_path):
        """Preprocess image to standard size"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open and convert to RGB
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to process image at {image_path}: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample['messages']
        
        # Get sample info for better error reporting
        sample_info = f"Sample {idx}, Type {sample['metadata']['sample_type']}, Road ID: {sample['metadata']['road_id']}"
        
        try:
            # Process images in messages
            processed_messages = []
            for msg in messages:
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    new_content = []
                    for item in msg['content']:
                        if item['type'] == 'image':
                            # Load and preprocess image
                            image = self.preprocess_image(item['image'])
                            new_content.append({
                                'type': 'image',
                                'image': image  # Pass PIL Image object
                            })
                        else:
                            new_content.append(item)
                    processed_messages.append({
                        'role': msg['role'],
                        'content': new_content
                    })
                else:
                    processed_messages.append(msg)
            
            # Format messages for Qwen
            text = self.processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Process vision inputs
            image_inputs, video_inputs = process_vision_info(processed_messages)
            
            # Tokenize
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            )
            
            # Flatten batch dimension
            inputs = {k: v[0] if v.shape[0] == 1 else v for k, v in inputs.items()}
            
            # Add metadata
            inputs['metadata'] = sample['metadata']
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error processing {sample_info}")
            raise RuntimeError(f"Failed to process {sample_info}: {str(e)}") from e

class ProgressiveTrainingStrategy:
    """Implement progressive training with changing data composition"""
    
    def __init__(self, base_data_path, processor, max_length=2048, image_size=(448, 448)):
        self.base_data_path = base_data_path
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load all data to understand distribution
        with open(base_data_path, 'r') as f:
            all_data = json.load(f)
        
        # Organize by type
        self.data_by_type = defaultdict(list)
        for sample in all_data:
            sample_type = sample['metadata']['sample_type']
            self.data_by_type[sample_type].append(sample)
        
        self.total_samples = sum(len(samples) for samples in self.data_by_type.values())
        
        logger.info("Total data distribution:")
        logger.info(f"  Total samples: {self.total_samples}")
        for t, samples in self.data_by_type.items():
            logger.info(f"  Type {t}: {len(samples)} samples ({len(samples)/self.total_samples*100:.1f}%)")
    
    def create_epoch_dataset(self, epoch, strategy='progressive'):
        """Create dataset with different composition for each epoch"""
        
        if strategy == 'progressive':
            if epoch == 0:
                # First epoch: Balanced mix (20% Type 1, 50% Type 2, 30% Type 3)
                type_ratios = {1: 0.2, 2: 0.5, 3: 0.3}
                logger.info("Epoch 0: Balanced mix (20% Type 1, 50% Type 2, 30% Type 3)")
            else:
                # Second epoch: Type 1 dominant (60% Type 1, 30% Type 2, 10% Type 3)
                type_ratios = {1: 0.6, 2: 0.3, 3: 0.1}
                logger.info("Epoch 1+: Type 1 dominant (60% Type 1, 30% Type 2, 10% Type 3)")
        else:
            # Fixed ratio
            type_ratios = {1: 0.3, 2: 0.5, 3: 0.2}
        
        # Sample data according to ratios
        epoch_data = []
        
        # Use all available data
        total_target = self.total_samples
        logger.info(f"Using all {total_target} samples for training")
        
        for sample_type, ratio in type_ratios.items():
            target_count = int(total_target * ratio)
            available = self.data_by_type[sample_type]
            
            if len(available) >= target_count:
                # Sample without replacement
                sampled = random.sample(available, target_count)
            else:
                # Use all available and repeat some to reach target
                sampled = available.copy()
                if target_count > len(available):
                    # Repeat sampling to reach target count
                    additional_needed = target_count - len(available)
                    additional = random.choices(available, k=additional_needed)
                    sampled.extend(additional)
            
            epoch_data.extend(sampled)
            logger.info(f"  Sampled {len(sampled)} Type {sample_type} samples (target: {target_count})")
        
        # Shuffle epoch data
        random.shuffle(epoch_data)
        
        # Create temporary file for this epoch
        epoch_file = f"temp_epoch_{epoch}_data.json"
        with open(epoch_file, 'w') as f:
            json.dump(epoch_data, f)
        
        # Create dataset
        dataset = BikeabilityDataset(epoch_file, self.processor, self.max_length, image_size=self.image_size)
        
        # Clean up temp file
        os.remove(epoch_file)
        
        logger.info(f"Total epoch samples: {len(epoch_data)}")
        
        return dataset

def setup_model_and_lora(model_path, lora_config):
    """Setup model with LoRA"""
    logger.info("Loading Qwen2.5-VL model...")
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Disable cache for gradient checkpointing
    )
    
    # Enable gradient checkpointing first
    model.gradient_checkpointing_enable()
    
    # Add LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable input gradients after adding LoRA
    model.enable_input_require_grads()
    
    # Set model to training mode
    model.train()
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def custom_data_collator(processor):
    """Create custom data collator that handles metadata and Qwen2.5-VL format"""
    
    def collator(features):
        # Extract metadata before collating
        metadata = [f.pop('metadata', None) for f in features]
        
        # Since images are now preprocessed to same size, we can use standard collation
        batch = {}
        
        # Process each key
        for key in features[0].keys():
            if key == 'input_ids':
                # Pad input_ids
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=151643  # Qwen2.5's pad token id
                )
            elif key == 'attention_mask':
                # Pad attention_mask
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=0
                )
            elif key == 'labels':
                # Pad labels
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [f[key] for f in features],
                    batch_first=True,
                    padding_value=-100
                )
            elif isinstance(features[0][key], torch.Tensor):
                # For tensor types, stack them
                batch[key] = torch.stack([f[key] for f in features])
            else:
                # For non-tensor types, just collect them
                batch[key] = [f[key] for f in features]
        
        # Add labels for language modeling if not present
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids'].clone()
            # Set padding tokens to -100 in labels
            if 'attention_mask' in batch:
                batch['labels'][batch['attention_mask'] == 0] = -100
        
        # Add metadata back
        batch['metadata'] = metadata
        
        return batch
    
    return collator

def train_model(model, processor, train_dataset_creator, output_dir, num_epochs=2):
    """Train model with progressive strategy"""
    
    # Calculate steps per epoch for proper logging
    total_samples = train_dataset_creator.total_samples
    batch_size = 4
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = total_samples // effective_batch_size
    
    logger.info(f"Training configuration:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Estimated steps per epoch: {steps_per_epoch}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # We'll manually control epochs
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,  # Increased for larger dataset
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,  # Save less frequently for larger dataset
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Recommended setting
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        label_names=[]  # Explicitly set empty label_names to suppress warning
    )
    
    # Custom data collator
    data_collator = custom_data_collator(processor)
    
    # Train for multiple epochs with different data compositions
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch}")
        logger.info(f"{'='*60}")
        
        # Create dataset for this epoch
        train_dataset = train_dataset_creator.create_epoch_dataset(epoch)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,  # Use processing_class instead of tokenizer
            data_collator=data_collator
        )
        
        # Train for one epoch
        trainer.train()
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        trainer.save_model(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)  # Save processor too
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    return model

def parse_model_output(output_text, expected_type):
    """Parse model output based on expected type"""
    try:
        if expected_type == 3:
            # Type 3: Direct ratings
            match = re.search(r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)', output_text)
            if match:
                return {
                    'comfortable': int(match.group(1)),
                    'safe': int(match.group(2)),
                    'overall': int(match.group(3))
                }, []
        
        elif expected_type == 2:
            # Type 2: Factors and ratings
            factors_match = re.search(r'Factors:\s*\[(.*?)\]', output_text, re.IGNORECASE)
            ratings_match = re.search(r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)', output_text)
            
            factors = []
            if factors_match:
                factors = [f.strip() for f in factors_match.group(1).split(',')]
            
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
            structured_match = re.search(r'STRUCTURED OUTPUT:(.*?)$', output_text, re.DOTALL | re.IGNORECASE)
            if structured_match:
                structured_text = structured_match.group(1)
                
                # Extract factors and ratings
                factors_match = re.search(r'Factors:\s*\[(.*?)\]', structured_text, re.IGNORECASE)
                ratings_match = re.search(r'comfortable:\s*(\d),\s*safe:\s*(\d),\s*overall:\s*(\d)', structured_text)
                
                factors = []
                if factors_match:
                    factors = [f.strip() for f in factors_match.group(1).split(',')]
                
                ratings = {}
                if ratings_match:
                    ratings = {
                        'comfortable': int(ratings_match.group(1)),
                        'safe': int(ratings_match.group(2)),
                        'overall': int(ratings_match.group(3))
                    }
                
                return ratings, factors
    
    except Exception as e:
        logger.error(f"Error parsing output: {e}")
    
    return {}, []

def predict_on_test_set(model, processor, test_data_path, locations_info, bike_lane_info, output_dir):
    """Make predictions on test set"""
    logger.info(f"\nMaking predictions on test set: {test_data_path}")
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Import functions from data preparation
    from sft_data_preparation import (
        get_image_path, 
        get_osm_description,
        create_type1_prompt,
        create_type2_prompt,
        create_type3_prompt,
        PERSONA_DESCRIPTIONS
    )
    
    predictions = []
    model.eval()
    
    # Process each test sample
    for user_idx, user in enumerate(tqdm(test_data, desc="Processing test users")):
        persona = user.get('geller_classification', 'Unknown')
        if persona == 'Unknown':
            continue
        
        persona_desc = PERSONA_DESCRIPTIONS[persona]
        
        for rating in user.get('ratings', []):
            road_id = rating['road']['id']
            image_path = get_image_path(road_id)
            osm_desc = get_osm_description(road_id, locations_info, bike_lane_info)
            
            # Preprocess image
            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((448, 448), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                continue
            
            # Test all three types of prompts
            for prompt_type in [1, 2, 3]:
                # Create prompt based on type
                if prompt_type == 1:
                    prompt = create_type1_prompt(persona, persona_desc)
                elif prompt_type == 2:
                    prompt = create_type2_prompt(persona, persona_desc)
                else:
                    prompt = create_type3_prompt(persona, persona_desc)
                
                # Add OSM info
                if osm_desc != "No additional road information available":
                    prompt += f"\n\nAdditional road information: {osm_desc}"
                
                # Prepare input
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
                    padding=True,
                    return_tensors="pt",
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
                    'session_id': user['sessionId'],
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
    
    # Save predictions
    output_file = os.path.join(output_dir, 'qwen_vl_test_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Saved {len(predictions)} predictions to {output_file}")
    
    # Calculate metrics
    evaluate_predictions(predictions, output_dir)
    
    return predictions

def evaluate_predictions(predictions, output_dir):
    """Evaluate prediction results"""
    # Group by prompt type
    results_by_type = defaultdict(list)
    for pred in predictions:
        results_by_type[pred['prompt_type']].append(pred)
    
    evaluation_results = {}
    
    for prompt_type, type_preds in results_by_type.items():
        logger.info(f"\nEvaluating Type {prompt_type} ({len(type_preds)} predictions)")
        
        # Extract valid predictions
        y_true = []
        y_pred = []
        
        for pred in type_preds:
            if pred['predictions']['ratings']:
                for metric in ['comfortable', 'safe', 'overall']:
                    if metric in pred['predictions']['ratings']:
                        y_true.append(pred['ground_truth']['ratings'][metric])
                        y_pred.append(pred['predictions']['ratings'][metric])
        
        if y_true:
            # Calculate MAE
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate exact match rate
            exact_matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            exact_match_rate = exact_matches / len(y_true)
            
            # Calculate ±1 accuracy
            within_one = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= 1)
            within_one_rate = within_one / len(y_true)
            
            evaluation_results[f'type_{prompt_type}'] = {
                'mae': mae,
                'exact_match_rate': exact_match_rate,
                'within_one_rate': within_one_rate,
                'n_predictions': len(y_true)
            }
            
            logger.info(f"  MAE: {mae:.3f}")
            logger.info(f"  Exact match rate: {exact_match_rate:.1%}")
            logger.info(f"  ±1 accuracy: {within_one_rate:.1%}")
    
    # Save evaluation results
    eval_file = os.path.join(output_dir, 'qwen_vl_evaluation_results.json')
    with open(eval_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return evaluation_results

def main():
    # Configuration
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    train_data_path = "./sft_training_data/sft_training_complete.json"
    test_data_path = "split_test_data.json"
    output_dir = "./qwen_vl_lora_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # LoRA configuration - adjusted for Qwen2.5-VL
    lora_config = LoraConfig(
        r=16,  # Reduced rank for stability
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Only attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Setup model
    model = setup_model_and_lora(model_path, lora_config)
    
    # Create progressive training strategy
    training_strategy = ProgressiveTrainingStrategy(train_data_path, processor)
    
    # Train model
    logger.info("\nStarting progressive training...")
    trained_model = train_model(
        model, 
        processor, 
        training_strategy, 
        output_dir, 
        num_epochs=2
    )
    
    # Save final model
    final_model_dir = os.path.join(output_dir, "final_model")
    trained_model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info(f"Saved final model to {final_model_dir}")
    
    # Load OSM data for predictions
    with open('locations_info.json', 'r') as f:
        locations_data = json.load(f)
    locations_info = {loc['id']: loc for loc in locations_data['locations']}
    
    import pandas as pd
    bike_lane_df = pd.read_csv('bike_lane_info.csv')
    bike_lane_info = {}
    for _, row in bike_lane_df.iterrows():
        bike_lane_info[f"loc_{row['index']}"] = row['bike_lane_type']
    
    # Make predictions on test set
    logger.info("\nMaking predictions on test set...")
    predictions = predict_on_test_set(
        trained_model,
        processor,
        test_data_path,
        locations_info,
        bike_lane_info,
        output_dir
    )
    
    logger.info("\nTraining and evaluation completed!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()