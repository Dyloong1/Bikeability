import json
from collections import Counter

def extract_gpt4o_factors_to_json(predictions_file='gpt4o_baseline_predictions.json'):
    """Extract all GPT-4o influencing factors and save to JSON files"""
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Extract all factors
    all_factors_list = []
    factor_counter = Counter()
    
    # Collect all predictions with their factors
    all_predictions_data = []
    
    for i, pred in enumerate(predictions):
        # Get the influencing factors
        factors = pred.get('influencing_factors', [])
        
        # Count each factor
        for factor in factors:
            factor_counter[factor] += 1
            all_factors_list.append(factor)
        
        # Store prediction data
        pred_data = {
            'prediction_id': i,
            'location_id': pred.get('location_id', ''),
            'persona': pred.get('persona', ''),
            'influencing_factors': factors,
            'ground_truth_indicators': pred.get('ground_truth_indicators', []),
            'predicted_ratings': pred.get('ratings', {}),
            'ground_truth_ratings': pred.get('ground_truth', {})
        }
        all_predictions_data.append(pred_data)
    
    # Create simple factors list with counts
    factors_simple = {
        'total_predictions': len(predictions),
        'total_factor_occurrences': len(all_factors_list),
        'unique_factors_count': len(factor_counter),
        'factors_by_frequency': []
    }
    
    # Add factors sorted by frequency
    for factor, count in factor_counter.most_common():
        factors_simple['factors_by_frequency'].append({
            'factor': factor,
            'count': count,
            'percentage': round(count / len(predictions) * 100, 1)
        })
    
    # Create detailed output with examples
    factors_detailed = {
        'summary': {
            'total_predictions': len(predictions),
            'unique_factors': len(factor_counter),
            'total_factor_occurrences': len(all_factors_list)
        },
        'all_unique_factors': sorted(factor_counter.keys()),
        'factor_details': {},
        'all_predictions': all_predictions_data
    }
    
    # Add examples for each factor
    for factor in factor_counter:
        examples = []
        for pred in all_predictions_data:
            if factor in pred['influencing_factors'] and len(examples) < 3:
                examples.append({
                    'location': pred['location_id'],
                    'persona': pred['persona'],
                    'co_occurring_factors': [f for f in pred['influencing_factors'] if f != factor],
                    'ground_truth': pred['ground_truth_indicators']
                })
        
        factors_detailed['factor_details'][factor] = {
            'count': factor_counter[factor],
            'percentage': round(factor_counter[factor] / len(predictions) * 100, 1),
            'examples': examples
        }
    
    # Save simple version
    with open('gpt4o_factors_simple.json', 'w', encoding='utf-8') as f:
        json.dump(factors_simple, f, indent=2, ensure_ascii=False)
    print("Saved simple factors list to: gpt4o_factors_simple.json")
    
    # Save detailed version
    with open('gpt4o_factors_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(factors_detailed, f, indent=2, ensure_ascii=False)
    print("Saved detailed factors data to: gpt4o_factors_detailed.json")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Total predictions: {len(predictions)}")
    print(f"- Unique factors found: {len(factor_counter)}")
    print(f"- Total factor occurrences: {len(all_factors_list)}")
    print(f"\nTop 20 most common factors:")
    print("-" * 60)
    for factor, count in factor_counter.most_common(20):
        percentage = count / len(predictions) * 100
        print(f"{factor:<40} {count:>5} ({percentage:>5.1f}%)")
    
    return factors_simple, factors_detailed

if __name__ == "__main__":
    extract_gpt4o_factors_to_json()