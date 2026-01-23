#!/usr/bin/env python3
"""
Updated ablation study for Qwen-VL training
Run different data type combinations on different GPUs

"""
import os
import sys
import argparse
import subprocess
from multiprocessing import Process
import time
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_on_gpu(gpu_id, experiment_name, data_types, progressive=False, num_epochs=2):
    """Run training on specific GPU with given data types"""
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Create unique output directory
        output_dir = f"./qwen_vl_lora_output_{experiment_name}"
        
        # Create config file for this experiment
        config = {
            "experiment_name": experiment_name,
            "gpu_id": gpu_id,
            "data_types": data_types,
            "progressive": progressive,
            "output_dir": output_dir,
            "num_epochs": num_epochs
        }
        
        config_file = f"config_{experiment_name}_{gpu_id}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training script with specific configuration
        cmd = [
            sys.executable,
            "train_ablation_single.py",
            "--config", config_file,
            "--gpu-id", str(gpu_id),
            "--experiment-name", experiment_name,
            "--output-dir", output_dir,
            "--num-epochs", str(num_epochs)
        ]
        
        if data_types:
            cmd.extend(["--data-types"] + [str(t) for t in data_types])
        
        if progressive:
            cmd.append("--progressive")
        
        logger.info(f"GPU {gpu_id}: Starting {experiment_name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Create log file for this experiment
        log_file = f"log_{experiment_name}.txt"
        with open(log_file, 'w') as log:
            # Run training
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        
        if result.returncode == 0:
            logger.info(f"GPU {gpu_id}: {experiment_name} completed successfully")
        else:
            logger.error(f"GPU {gpu_id}: {experiment_name} failed with return code {result.returncode}")
        
        # Clean up config file
        if os.path.exists(config_file):
            os.remove(config_file)
            
    except Exception as e:
        logger.error(f"GPU {gpu_id}: {experiment_name} encountered error: {str(e)}")

def check_gpu_availability():
    """Check if GPUs are available"""
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        logger.info(f"Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return num_gpus
    except Exception as e:
        logger.error(f"Error checking GPUs: {e}")
        return 0

def main():
    """Run all ablation experiments in parallel"""
    
    # Check if we should use the NWNH-enhanced dataset
    training_data_file = "./sft_training_data/sft_training_complete_with_nwnh.json"
    if not os.path.exists(training_data_file):
        logger.warning(f"NWNH-enhanced training data not found: {training_data_file}")
        logger.warning("Using original training data instead")
        training_data_file = "./sft_training_data/sft_training_complete.json"
    else:
        logger.info(f"Using NWNH-enhanced training data: {training_data_file}")
    
    # Check GPU availability
    num_gpus = check_gpu_availability()
    if num_gpus < 4:
        logger.warning(f"Only {num_gpus} GPU(s) found. Some experiments may need to be run sequentially.")
    
    experiments = [
        # GPU 0: Full model with progressive training (baseline)
        {
            "gpu_id": 0,
            "name": "full_progressive",
            "data_types": [1, 2, 3],
            "progressive": True,
            "description": "Full model with all data types and progressive training (Epoch 0: balanced, Epoch 1: Type 1 dominant)"
        },
        # GPU 1: Only Type 2 (factors + ratings)
        {
            "gpu_id": 1,
            "name": "only_type2",
            "data_types": [2],
            "progressive": False,
            "description": "Only Type 2: Factors + Ratings"
        },
        # GPU 2: Type 1 + Type 2 (reasoning + factors, no direct ratings)
        {
            "gpu_id": 2,
            "name": "type1_type2",
            "data_types": [1, 2],
            "progressive": True,  # Progressive for multiple types
            "description": "Type 1 + Type 2: Reasoning + Factors with progressive training"
        },
        # GPU 3: Full model with fixed ratio training (non-progressive)
        {
            "gpu_id": 3,
            "name": "full_fixed",
            "data_types": [1, 2, 3],
            "progressive": False,
            "description": "Full model with all data types and fixed ratio training (constant mix across epochs)"
        }
    ]
    
    # Additional experiments for future runs
    additional_experiments = [
        # Type 2 + Type 3 (no reasoning)
        {
            "gpu_id": 0,
            "name": "type2_type3",
            "data_types": [2, 3],
            "progressive": False,
            "description": "Type 2 + Type 3: Factors + Direct Ratings (no reasoning)"
        },
        # Only Type 1 (reasoning only)
        {
            "gpu_id": 1,
            "name": "only_type1",
            "data_types": [1],
            "progressive": False,
            "description": "Only Type 1: Reasoning with structured output"
        }
    ]
    
    # Print experiment plan
    print("="*60)
    print("UPDATED ABLATION STUDY EXPERIMENTS")
    print("="*60)
    print(f"Training data: {training_data_file}")
    print("\nMain experiments:")
    for exp in experiments:
        print(f"\nGPU {exp['gpu_id']}: {exp['name']}")
        print(f"  Description: {exp['description']}")
        print(f"  Data types: {exp['data_types']}")
        print(f"  Progressive: {exp['progressive']}")
    
    print("\n" + "="*60)
    print("\nKey differences from previous run:")
    print("1. Using NWNH-enhanced training data (if available)")
    print("2. Replaced 'only_type3' with 'full_fixed' (non-progressive)")
    print("3. 'full_fixed' uses constant data mix across epochs")
    print("   - Type 1: 30%, Type 2: 50%, Type 3: 20% (fixed)")
    print("4. 'full_progressive' still uses progressive strategy")
    print("   - Epoch 0: 20% Type 1, 50% Type 2, 30% Type 3")
    print("   - Epoch 1: 60% Type 1, 30% Type 2, 10% Type 3")
    print("\n" + "="*60)
    
    # Ask for confirmation
    response = input("\nProceed with experiments? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Create processes for each experiment
    processes = []
    start_time = time.time()
    
    # Run experiments
    for exp in experiments[:num_gpus]:  # Only run as many as we have GPUs
        p = Process(
            target=run_training_on_gpu,
            args=(
                exp["gpu_id"],
                exp["name"],
                exp["data_types"],
                exp["progressive"]
            )
        )
        processes.append(p)
        p.start()
        time.sleep(5)  # Small delay to avoid file conflicts
    
    # Wait for all processes to complete
    logger.info("\nExperiments started. Waiting for completion...")
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"Experiment {i} completed")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"\nAll experiments completed in {elapsed_time/60:.1f} minutes")
    
    # Summarize results
    print("\n" + "="*60)
    print("EXPERIMENT OUTPUTS")
    print("="*60)
    
    for exp in experiments:
        output_dir = f"./qwen_vl_lora_output_{exp['name']}"
        if os.path.exists(output_dir):
            print(f"\n{exp['name']}:")
            print(f"  Output directory: {output_dir}")
            
            # Check for final model
            final_model_dir = os.path.join(output_dir, "final_model")
            if os.path.exists(final_model_dir):
                print(f"  ✓ Final model saved")
            
            # Check for checkpoints
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                print(f"  ✓ {len(checkpoints)} checkpoint(s) saved")
            
            # Check for experiment config
            config_file = os.path.join(output_dir, "experiment_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"  Data types: {config.get('data_types', 'N/A')}")
                print(f"  Progressive: {config.get('progressive', 'N/A')}")
            
            # Check for training logs
            log_file = f"log_{exp['name']}.txt"
            if os.path.exists(log_file):
                print(f"  ✓ Training log: {log_file}")
    
    # Create summary report
    summary = {
        "experiments": experiments,
        "total_time_minutes": elapsed_time / 60,
        "num_gpus_used": num_gpus,
        "training_data": training_data_file,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Updated experiments with NWNH data and replaced only_type3 with full_fixed"
    }
    
    with open("ablation_study_summary_updated.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to ablation_study_summary_updated.json")

if __name__ == "__main__":
    main()