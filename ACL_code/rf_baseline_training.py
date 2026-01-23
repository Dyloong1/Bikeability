#!/usr/bin/env python3
"""
Random Forest Baseline Training
Two approaches:
1. Predict only 3 ratings (comfortable, safe, overall)
2. Predict 3 ratings + 40 indicators (43 outputs total)
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib
import os
from datetime import datetime
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Define persona mappings
PERSONA_MAPPING = {
    "Strong and Fearless": 0,
    "Enthused and Confident": 1,
    "Interested but Concerned": 2,
    "No Way No How": 3
}

# Define bike lane type mappings
BIKE_LANE_MAPPING = {
    "striped": 0,
    "protected": 1,
    "buffered": 2,
    "roadway": 3,
    "sharrows": 4,
    "shared": 5,
    "cycle_track": 6,
    "separated": 7
}

# Visual features from the paper
VISUAL_FEATURES = [
    "tree", "road", "sky", "building", "sidewalk",
    "grass", "car", "wall", "fence", "floor",
    "earth", "plant", "signboard", "skyscraper", "ceiling",
    "rail", "water", "palm tree", "unknown"
]

def load_data(train_path, test_path):
    """Load training and test data"""
    logger.info(f"Loading training data from {train_path}")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    logger.info(f"Loading test data from {test_path}")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    return train_data, test_data

def load_osm_data(locations_file, bike_lane_file):
    """Load OSM and bike lane data"""
    # Load locations info
    with open(locations_file, 'r') as f:
        locations_data = json.load(f)
    
    locations_info = {}
    for loc in locations_data['locations']:
        locations_info[loc['id']] = loc
    
    # Load bike lane info
    bike_lane_df = pd.read_csv(bike_lane_file)
    bike_lane_info = {}
    for _, row in bike_lane_df.iterrows():
        bike_lane_info[f"loc_{row['index']}"] = row['bike_lane_type']
    
    return locations_info, bike_lane_info

def load_indicator_pool(indicator_file):
    """Load indicator pool"""
    with open(indicator_file, 'r') as f:
        indicator_data = json.load(f)
    return indicator_data['indicators']

def extract_features(rating, persona, locations_info, bike_lane_info):
    """Extract all features for a single rating"""
    features = {}
    road_id = rating['road']['id']
    
    # 1. Visual features (19 features)
    visual_features = rating.get('visual_features', {})
    for vf in VISUAL_FEATURES:
        features[f'visual_{vf.replace(" ", "_")}'] = visual_features.get(vf, 0.0)
    
    # 2. Persona feature (1 feature)
    features['persona'] = PERSONA_MAPPING.get(persona, -1)
    
    # 3. OSM features
    if road_id in locations_info:
        loc_data = locations_info[road_id]
        
        # Road type (categorical -> numeric)
        road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service']
        road_type = loc_data.get('roadType', 'unknown')
        features['road_type'] = road_types.index(road_type) if road_type in road_types else -1
        
        # Numeric features
        features['maxspeed'] = loc_data.get('maxspeed', 25) if loc_data.get('maxspeed') != 'unknown' else 25
        features['lanes'] = int(loc_data.get('lanes', 1)) if loc_data.get('lanes') not in ['unknown', None] else 1
        features['has_cycleway'] = 1 if loc_data.get('hasCycleway', False) else 0
        
        # Surface type
        surfaces = ['asphalt', 'concrete', 'paved', 'unpaved', 'gravel']
        surface = loc_data.get('surface', 'unknown')
        features['surface'] = surfaces.index(surface) if surface in surfaces else -1
    else:
        # Default values if location not found
        features['road_type'] = -1
        features['maxspeed'] = 25
        features['lanes'] = 1
        features['has_cycleway'] = 0
        features['surface'] = -1
    
    # 4. Bike lane type (1 feature)
    if road_id in bike_lane_info:
        features['bike_lane_type'] = BIKE_LANE_MAPPING.get(bike_lane_info[road_id], -1)
    else:
        features['bike_lane_type'] = -1
    
    return features

def prepare_data_approach1(data, locations_info, bike_lane_info):
    """Prepare data for Approach 1: Only predict 3 ratings"""
    X = []
    y = []
    
    for user in data:
        persona = user.get('geller_classification', 'Unknown')
        
        for rating in user.get('ratings', []):
            # Extract features
            features = extract_features(rating, persona, locations_info, bike_lane_info)
            
            # Extract targets (3 ratings)
            targets = [
                rating['ratings']['comfortable'],
                rating['ratings']['safe'],
                rating['ratings']['overall']
            ]
            
            X.append(list(features.values()))
            y.append(targets)
    
    feature_names = list(features.keys())
    return np.array(X), np.array(y), feature_names

def prepare_data_approach2(data, locations_info, bike_lane_info, indicators):
    """Prepare data for Approach 2: Predict 3 ratings + 40 indicators"""
    X = []
    y_ratings = []
    y_indicators = []
    
    for user in data:
        persona = user.get('geller_classification', 'Unknown')
        
        for rating in user.get('ratings', []):
            # Extract features
            features = extract_features(rating, persona, locations_info, bike_lane_info)
            
            # Extract rating targets (3 ratings)
            rating_targets = [
                rating['ratings']['comfortable'],
                rating['ratings']['safe'],
                rating['ratings']['overall']
            ]
            
            # Extract indicator targets (40 binary values)
            present_indicators = set(rating.get('indicators', []))
            indicator_targets = [1 if ind in present_indicators else 0 for ind in indicators]
            
            X.append(list(features.values()))
            y_ratings.append(rating_targets)
            y_indicators.append(indicator_targets)
    
    feature_names = list(features.keys())
    return np.array(X), np.array(y_ratings), np.array(y_indicators), feature_names

class Approach1Model:
    """Model for Approach 1: Predict only ratings"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train, feature_names):
        """Train the model"""
        self.feature_names = feature_names
        
        logger.info("Training Approach 1 model (3 ratings only)...")
        
        # Grid search for hyperparameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        self.model = MultiOutputRegressor(base_model)
        
        # Note: MultiOutputRegressor doesn't support grid search directly
        # So we'll use default parameters for simplicity
        self.model.fit(X_train, y_train)
        
        logger.info("Approach 1 model training completed")
        
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        
        results = {}
        rating_names = ['comfortable', 'safe', 'overall']
        
        for i, name in enumerate(rating_names):
            mae = mean_absolute_error(y_test[:, i], predictions[:, i])
            results[f'{name}_mae'] = mae
            
            # Calculate exact match rate
            exact_matches = np.sum(np.round(predictions[:, i]) == y_test[:, i])
            results[f'{name}_exact_match_rate'] = exact_matches / len(y_test)
        
        # Overall MAE
        results['overall_mae'] = mean_absolute_error(y_test, predictions)
        
        return results, predictions

class Approach2Model:
    """Model for Approach 2: Predict ratings + indicators"""
    
    def __init__(self):
        self.rating_model = None
        self.indicator_model = None
        self.feature_names = None
        
    def train(self, X_train, y_ratings_train, y_indicators_train, feature_names):
        """Train both models"""
        self.feature_names = feature_names
        
        logger.info("Training Approach 2 models (3 ratings + 40 indicators)...")
        
        # Train rating model (regression)
        logger.info("  Training rating model...")
        base_rating_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.rating_model = MultiOutputRegressor(base_rating_model)
        self.rating_model.fit(X_train, y_ratings_train)
        
        # Train indicator model (classification)
        logger.info("  Training indicator model...")
        base_indicator_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.indicator_model = MultiOutputClassifier(base_indicator_model)
        self.indicator_model.fit(X_train, y_indicators_train)
        
        logger.info("Approach 2 model training completed")
        
    def predict(self, X_test):
        """Make predictions for both ratings and indicators"""
        rating_predictions = self.rating_model.predict(X_test)
        indicator_predictions = self.indicator_model.predict(X_test)
        return rating_predictions, indicator_predictions
    
    def evaluate(self, X_test, y_ratings_test, y_indicators_test, indicators):
        """Evaluate both models"""
        rating_predictions, indicator_predictions = self.predict(X_test)
        
        results = {}
        
        # Evaluate ratings
        rating_names = ['comfortable', 'safe', 'overall']
        for i, name in enumerate(rating_names):
            mae = mean_absolute_error(y_ratings_test[:, i], rating_predictions[:, i])
            results[f'{name}_mae'] = mae
            
            exact_matches = np.sum(np.round(rating_predictions[:, i]) == y_ratings_test[:, i])
            results[f'{name}_exact_match_rate'] = exact_matches / len(y_ratings_test)
        
        results['ratings_overall_mae'] = mean_absolute_error(y_ratings_test, rating_predictions)
        
        # Evaluate indicators (top 10 most frequent)
        indicator_freq = np.sum(y_indicators_test, axis=0)
        top_10_indices = np.argsort(indicator_freq)[-10:][::-1]
        
        results['top_10_indicators'] = {}
        for idx in top_10_indices:
            indicator_name = indicators[idx]
            accuracy = accuracy_score(y_indicators_test[:, idx], indicator_predictions[:, idx])
            results['top_10_indicators'][indicator_name] = {
                'accuracy': accuracy,
                'frequency': int(indicator_freq[idx])
            }
        
        # Overall indicator accuracy
        results['indicators_overall_accuracy'] = accuracy_score(
            y_indicators_test.flatten(), 
            indicator_predictions.flatten()
        )
        
        return results, rating_predictions, indicator_predictions

def save_predictions(test_data, predictions_approach1, predictions_approach2, indicators, output_dir):
    """Save prediction results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results
    results = []
    idx = 0
    
    for user in test_data:
        persona = user.get('geller_classification', 'Unknown')
        session_id = user.get('sessionId', 'unknown')
        
        for rating in user.get('ratings', []):
            result = {
                'session_id': session_id,
                'persona': persona,
                'road_id': rating['road']['id'],
                'ground_truth': {
                    'comfortable': rating['ratings']['comfortable'],
                    'safe': rating['ratings']['safe'],
                    'overall': rating['ratings']['overall'],
                    'indicators': rating.get('indicators', [])
                },
                'approach1_predictions': {
                    'comfortable': float(predictions_approach1[0][idx, 0]),
                    'safe': float(predictions_approach1[0][idx, 1]),
                    'overall': float(predictions_approach1[0][idx, 2])
                },
                'approach2_predictions': {
                    'ratings': {
                        'comfortable': float(predictions_approach2[0][idx, 0]),
                        'safe': float(predictions_approach2[0][idx, 1]),
                        'overall': float(predictions_approach2[0][idx, 2])
                    },
                    'indicators': {
                        indicators[i]: bool(predictions_approach2[1][idx, i])
                        for i in range(len(indicators))
                        if predictions_approach2[1][idx, i] == 1
                    }
                }
            }
            results.append(result)
            idx += 1
    
    # Save detailed results
    output_file = os.path.join(output_dir, 'rf_baseline_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Predictions saved to {output_file}")
    
    return results

def main():
    # Configuration
    train_path = "split_train_data_with_visual_features.json"
    test_path = "split_test_data_with_visual_features.json"
    locations_file = "locations_info.json"
    bike_lane_file = "bike_lane_info.csv"
    indicator_file = "indicator_pool.json"
    output_dir = "./rf_baseline_results"
    
    # Load data
    logger.info("Loading data...")
    train_data, test_data = load_data(train_path, test_path)
    locations_info, bike_lane_info = load_osm_data(locations_file, bike_lane_file)
    indicators = load_indicator_pool(indicator_file)
    
    logger.info(f"Training data: {len(train_data)} users")
    logger.info(f"Test data: {len(test_data)} users")
    
    # Approach 1: Predict only 3 ratings
    logger.info("\n" + "="*60)
    logger.info("APPROACH 1: Predicting 3 ratings only")
    logger.info("="*60)
    
    X_train_1, y_train_1, feature_names_1 = prepare_data_approach1(
        train_data, locations_info, bike_lane_info
    )
    X_test_1, y_test_1, _ = prepare_data_approach1(
        test_data, locations_info, bike_lane_info
    )
    
    logger.info(f"Training samples: {X_train_1.shape[0]}")
    logger.info(f"Features: {X_train_1.shape[1]}")
    logger.info(f"Targets: {y_train_1.shape[1]} ratings")
    
    model1 = Approach1Model()
    model1.train(X_train_1, y_train_1, feature_names_1)
    
    # Evaluate Approach 1
    results1, predictions1 = model1.evaluate(X_test_1, y_test_1)
    
    logger.info("\nApproach 1 Results:")
    for metric, value in results1.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # Save model
    model1_path = os.path.join(output_dir, 'approach1_model.pkl')
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model1, model1_path)
    logger.info(f"Model saved to {model1_path}")
    
    # Approach 2: Predict 3 ratings + 40 indicators
    logger.info("\n" + "="*60)
    logger.info("APPROACH 2: Predicting 3 ratings + 40 indicators")
    logger.info("="*60)
    
    X_train_2, y_ratings_train_2, y_indicators_train_2, feature_names_2 = prepare_data_approach2(
        train_data, locations_info, bike_lane_info, indicators
    )
    X_test_2, y_ratings_test_2, y_indicators_test_2, _ = prepare_data_approach2(
        test_data, locations_info, bike_lane_info, indicators
    )
    
    logger.info(f"Training samples: {X_train_2.shape[0]}")
    logger.info(f"Features: {X_train_2.shape[1]}")
    logger.info(f"Rating targets: {y_ratings_train_2.shape[1]}")
    logger.info(f"Indicator targets: {y_indicators_train_2.shape[1]}")
    
    model2 = Approach2Model()
    model2.train(X_train_2, y_ratings_train_2, y_indicators_train_2, feature_names_2)
    
    # Evaluate Approach 2
    results2, rating_preds2, indicator_preds2 = model2.evaluate(
        X_test_2, y_ratings_test_2, y_indicators_test_2, indicators
    )
    
    logger.info("\nApproach 2 Results:")
    logger.info("Rating predictions:")
    for metric, value in results2.items():
        if 'indicator' not in metric and 'top_10' not in metric:
            logger.info(f"  {metric}: {value:.3f}")
    
    logger.info("\nIndicator predictions:")
    logger.info(f"  Overall accuracy: {results2['indicators_overall_accuracy']:.3f}")
    logger.info("  Top 10 most frequent indicators:")
    for ind_name, ind_results in results2['top_10_indicators'].items():
        logger.info(f"    {ind_name}: accuracy={ind_results['accuracy']:.3f}, frequency={ind_results['frequency']}")
    
    # Save model
    model2_path = os.path.join(output_dir, 'approach2_model.pkl')
    joblib.dump(model2, model2_path)
    logger.info(f"Model saved to {model2_path}")
    
    # Save predictions
    logger.info("\nSaving predictions...")
    predictions_data = save_predictions(
        test_data,
        (predictions1, None),
        (rating_preds2, indicator_preds2),
        indicators,
        output_dir
    )
    
    # Save evaluation summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'approach1_results': results1,
        'approach2_results': results2,
        'feature_names': feature_names_1,
        'indicators': indicators,
        'train_size': len(train_data),
        'test_size': len(test_data)
    }
    
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation summary saved to {summary_path}")
    
    # Print final comparison
    logger.info("\n" + "="*60)
    logger.info("FINAL COMPARISON")
    logger.info("="*60)
    logger.info("\nApproach 1 (3 ratings only):")
    logger.info(f"  Overall MAE: {results1['overall_mae']:.3f}")
    logger.info(f"  Comfortable exact match: {results1['comfortable_exact_match_rate']:.1%}")
    logger.info(f"  Safe exact match: {results1['safe_exact_match_rate']:.1%}")
    logger.info(f"  Overall exact match: {results1['overall_exact_match_rate']:.1%}")
    
    logger.info("\nApproach 2 (3 ratings + 40 indicators):")
    logger.info(f"  Ratings Overall MAE: {results2['ratings_overall_mae']:.3f}")
    logger.info(f"  Comfortable exact match: {results2['comfortable_exact_match_rate']:.1%}")
    logger.info(f"  Safe exact match: {results2['safe_exact_match_rate']:.1%}")
    logger.info(f"  Overall exact match: {results2['overall_exact_match_rate']:.1%}")
    logger.info(f"  Indicators accuracy: {results2['indicators_overall_accuracy']:.1%}")
    
    logger.info("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
