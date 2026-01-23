# Bikeability

## Project Structure

### data/
Survey data files:
- `bike_lane_info.csv` - Bike lane information
- `indicator_pool.json` - Indicator pool for evaluation
- `locations_info.json` - Location information
- `ratings_anonymous.json` - Anonymous rating data
- `surveys_anonymous.json` - Anonymous survey data

### code/
Code for Qwen VL training and baseline experiments:
- `train_qwen_vl.py` - Qwen VL model training script
- `qwen_vl_predict.py` - Qwen VL prediction
- `qwen_feature_extraction.py` - Feature extraction using Qwen
- `sft_data_preparation.py` - SFT data preparation
- `ablation_training_script.py` - Ablation study training
- `gpt4o_baseline.py` - GPT-4o baseline
- `gpt4o_eval.py` - GPT-4o evaluation
- `gpt4o_eval_language.py` - GPT-4o language evaluation
- `extract_gpt4o_factors.py` - Extract factors from GPT-4o
- `rf_baseline_training.py` - Random Forest baseline training