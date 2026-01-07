import json
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

class GPT4oSemanticEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize evaluator with semantic similarity model"""
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}
        
    def get_embedding(self, text):
        """Get embedding for a text, with caching"""
        if text not in self.embeddings_cache:
            self.embeddings_cache[text] = self.model.encode([text])[0]
        return self.embeddings_cache[text]
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_embedding(text1.lower().strip())
        emb2 = self.get_embedding(text2.lower().strip())
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def find_best_matches(self, predicted_factors, ground_truth_indicators, threshold=0.7):
        """Find best semantic matches between predicted and ground truth"""
        matches = []
        matched_truth = set()
        matched_pred = set()
        
        # Create similarity matrix
        similarities = []
        for pred in predicted_factors:
            row = []
            for truth in ground_truth_indicators:
                sim = self.semantic_similarity(pred, truth)
                row.append(sim)
            similarities.append(row)
        
        # Convert to numpy array for easier manipulation
        sim_matrix = np.array(similarities)
        
        # Find best matches (greedy approach)
        while True:
            if sim_matrix.size == 0:
                break
                
            # Find maximum similarity
            max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            max_sim = sim_matrix[max_idx]
            
            if max_sim < threshold:
                break
                
            pred_idx, truth_idx = max_idx
            pred_factor = predicted_factors[pred_idx]
            truth_indicator = ground_truth_indicators[truth_idx]
            
            if pred_factor not in matched_pred and truth_indicator not in matched_truth:
                matches.append({
                    'predicted': pred_factor,
                    'ground_truth': truth_indicator,
                    'similarity': max_sim
                })
                matched_pred.add(pred_factor)
                matched_truth.add(truth_indicator)
            
            # Remove this pair from consideration
            sim_matrix[pred_idx, truth_idx] = -1
        
        return matches, matched_pred, matched_truth
    
    def calculate_semantic_metrics(self, predicted_factors, ground_truth_indicators, threshold=0.7):
        """Calculate metrics based on semantic similarity"""
        if not predicted_factors or not ground_truth_indicators:
            # Return zero metrics if either is empty
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'avg_similarity': 0.0,
                'matches': [],
                'tp': 0,
                'fp': len(predicted_factors) if predicted_factors else 0,
                'fn': len(ground_truth_indicators) if ground_truth_indicators else 0
            }
        
        # Calculate all pairwise similarities
        all_similarities = []
        for pred in predicted_factors:
            for truth in ground_truth_indicators:
                sim = self.semantic_similarity(pred, truth)
                all_similarities.append({
                    'pred': pred,
                    'truth': truth,
                    'similarity': sim
                })
        
        # Sort by similarity descending
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top K similarities where K = number of ground truth indicators
        # This represents the best possible alignment
        k = len(ground_truth_indicators)
        top_k_similarities = all_similarities[:k]
        
        # Calculate average similarity of top K pairs
        avg_top_k_similarity = np.mean([s['similarity'] for s in top_k_similarities])
        
        # Now find actual matches using threshold
        matches, matched_pred, matched_truth = self.find_best_matches(
            predicted_factors, ground_truth_indicators, threshold
        )
        
        # Calculate metrics
        tp = len(matches)
        fp = len(predicted_factors) - tp
        fn = len(ground_truth_indicators) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_similarity': avg_top_k_similarity,  # Now this is the average of top K
            'matches': matches,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_predictions(self, predictions_file, similarity_thresholds=[0.5, 0.6, 0.7, 0.8]):
        """Main evaluation function with multiple similarity thresholds"""
        # Load predictions
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        print(f"Loaded {len(predictions)} predictions")
        print(f"Using semantic similarity matching with thresholds: {similarity_thresholds}")
        print("Note: AvgSim now represents the average similarity of top-K pairs where K = |ground truth indicators|")
        
        # Group predictions by persona
        persona_groups = defaultdict(list)
        for pred in predictions:
            persona = pred.get('persona', 'Unknown')
            persona_groups[persona].append(pred)
        
        # Debug: Print available personas
        print(f"\nFound personas in data: {list(persona_groups.keys())}")
        print(f"Sample counts by persona:")
        for persona, preds in persona_groups.items():
            print(f"  {persona}: {len(preds)} samples")
        
        # Evaluate for each threshold
        results_by_threshold = {}
        
        for threshold in similarity_thresholds:
            print(f"\n{'='*80}")
            print(f"Evaluating with similarity threshold: {threshold}")
            print(f"{'='*80}")
            
            # Calculate metrics for each persona
            all_results = {}
            
            for persona, persona_predictions in persona_groups.items():
                # Calculate rating metrics (same for all thresholds)
                rating_results = self.calculate_rating_metrics_for_persona(persona_predictions)
                
                # Calculate factor identification metrics with semantic similarity
                factor_results = self.calculate_factor_metrics_for_persona(
                    persona_predictions, threshold
                )
                
                all_results[persona] = {
                    'n': len(persona_predictions),
                    'rating_metrics': rating_results,
                    'factor_metrics': factor_results
                }
            
            # Calculate average across all personas
            avg_results = self.calculate_average_metrics(all_results)
            all_results['Average'] = avg_results
            
            results_by_threshold[threshold] = all_results
            
            # Print results for this threshold
            self.print_results_table(all_results, threshold)
        
        # Print comparison across thresholds
        self.print_threshold_comparison(results_by_threshold)
        
        # Save detailed results
        self.save_detailed_results(results_by_threshold)
        
        return results_by_threshold
    
    def calculate_rating_metrics_for_persona(self, predictions):
        """Calculate rating metrics for a specific persona"""
        metrics = {dim: {'errors': [], 'predictions': [], 'ground_truth': []} 
                  for dim in ['comfortable', 'safe', 'overall']}
        
        for pred in predictions:
            pred_ratings = pred['ratings']
            true_ratings = pred['ground_truth']
            
            for dim in ['comfortable', 'safe', 'overall']:
                pred_val = pred_ratings[dim]
                true_val = true_ratings[dim]
                error = abs(pred_val - true_val)
                
                metrics[dim]['errors'].append(error)
                metrics[dim]['predictions'].append(pred_val)
                metrics[dim]['ground_truth'].append(true_val)
        
        # Calculate metrics for each dimension
        results = {}
        for dim in ['comfortable', 'safe', 'overall']:
            errors = np.array(metrics[dim]['errors'])
            preds = np.array(metrics[dim]['predictions'])
            truth = np.array(metrics[dim]['ground_truth'])
            
            mae = np.mean(errors) if len(errors) > 0 else 0
            exact_match = np.mean(errors == 0) if len(errors) > 0 else 0
            within_one = np.mean(errors <= 1) if len(errors) > 0 else 0
            
            if len(preds) > 1 and np.std(preds) > 0 and np.std(truth) > 0:
                pearson_corr, _ = pearsonr(preds, truth)
            else:
                pearson_corr = 0
            
            results[dim] = {
                'mae': mae,
                'exact_match': exact_match,
                'within_one': within_one,
                'pearson': pearson_corr
            }
        
        return results
    
    def calculate_factor_metrics_for_persona(self, predictions, threshold):
        """Calculate factor identification metrics using semantic similarity"""
        all_semantic_results = []
        match_examples = []
        skipped_count = 0
        
        for pred in predictions:
            pred_factors = pred.get('influencing_factors', [])
            true_indicators = pred.get('ground_truth_indicators', [])
            
            # Skip if no ground truth
            if not true_indicators:
                skipped_count += 1
                continue
            
            # Calculate semantic metrics
            semantic_result = self.calculate_semantic_metrics(
                pred_factors, true_indicators, threshold
            )
            all_semantic_results.append(semantic_result)
            
            # Collect match examples
            if semantic_result['matches'] and len(match_examples) < 5:
                match_examples.extend(semantic_result['matches'][:2])
        
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} samples without ground truth indicators")
        
        # Check if we have any results
        if not all_semantic_results:
            print(f"  No valid samples for {predictions[0].get('persona', 'Unknown')}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'avg_similarity': 0.0,
                'total_matches': 0
            }
        
        # Aggregate metrics
        total_tp = sum(r['tp'] for r in all_semantic_results)
        total_fp = sum(r['fp'] for r in all_semantic_results)
        total_fn = sum(r['fn'] for r in all_semantic_results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_similarities = [r['avg_similarity'] for r in all_semantic_results if r['avg_similarity'] > 0]
        overall_avg_similarity = np.mean(avg_similarities) if avg_similarities else 0
        
        # Print some match examples
        if match_examples:
            print(f"\nExample matches for {predictions[0].get('persona', 'Unknown')}:")
            for match in match_examples[:5]:
                print(f"  '{match['predicted']}' → '{match['ground_truth']}' (sim: {match['similarity']:.3f})")
        
        # Show the average similarity calculation
        if all_semantic_results:
            valid_sims = [r['avg_similarity'] for r in all_semantic_results if r['avg_similarity'] > 0]
            if valid_sims:
                print(f"  Average top-K similarity: {overall_avg_similarity:.3f} (based on {len(valid_sims)} valid predictions)")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_similarity': overall_avg_similarity,
            'total_matches': total_tp
        }
    
    def calculate_average_metrics(self, all_results):
        """Calculate average metrics across all personas"""
        # Initialize storage for averages
        rating_dims = ['comfortable', 'safe', 'overall']
        rating_metrics = ['mae', 'exact_match', 'within_one', 'pearson']
        factor_metrics = ['precision', 'recall', 'f1', 'avg_similarity']
        
        # Calculate total samples
        total_samples = sum(results['n'] for persona, results in all_results.items())
        
        # Initialize weighted sums
        weighted_rating_sums = {dim: {metric: 0 for metric in rating_metrics} for dim in rating_dims}
        weighted_factor_sums = {metric: 0 for metric in factor_metrics}
        
        # Calculate weighted sums
        for persona, results in all_results.items():
            weight = results['n'] / total_samples
            
            # Rating metrics
            for dim in rating_dims:
                for metric in rating_metrics:
                    weighted_rating_sums[dim][metric] += results['rating_metrics'][dim][metric] * weight
            
            # Factor metrics
            for metric in factor_metrics:
                weighted_factor_sums[metric] += results['factor_metrics'][metric] * weight
        
        # Construct average results
        avg_results = {
            'n': total_samples,
            'rating_metrics': {dim: weighted_rating_sums[dim] for dim in rating_dims},
            'factor_metrics': weighted_factor_sums
        }
        
        return avg_results
    
    def print_results_table(self, all_results, threshold):
        """Print results in table format"""
        print(f"\nRESULTS WITH SIMILARITY THRESHOLD = {threshold}")
        print("="*120)
        
        # Header
        print(f"{'Method':<30} {'Persona':<25} {'Rating Prediction':<40} {'Factor Identification':<40}")
        print(f"{'':<30} {'':<25} {'MAE↓':>8} {'EM↑':>8} {'W1↑':>8} {'Pear↑':>8} {'Prec↑':>8} {'Rec↑':>8} {'F1↑':>8} {'AvgSim↑':>8}")
        print("-"*120)
        
        # Define persona order
        persona_order = ['Strong and Fearless', 'Enthused and Confident', 'Interested but Concerned', 'No Way No How', 'Average']
        
        # Print results for each persona
        for persona_display in persona_order:
            if persona_display in all_results:
                results = all_results[persona_display]
                # Calculate average MAE across dimensions
                avg_mae = np.mean([results['rating_metrics'][dim]['mae'] for dim in ['comfortable', 'safe', 'overall']])
                avg_em = np.mean([results['rating_metrics'][dim]['exact_match'] for dim in ['comfortable', 'safe', 'overall']])
                avg_w1 = np.mean([results['rating_metrics'][dim]['within_one'] for dim in ['comfortable', 'safe', 'overall']])
                avg_pearson = np.mean([results['rating_metrics'][dim]['pearson'] for dim in ['comfortable', 'safe', 'overall']])
                
                # Factor metrics
                prec = results['factor_metrics']['precision']
                rec = results['factor_metrics']['recall']
                f1 = results['factor_metrics']['f1']
                avg_sim = results['factor_metrics']['avg_similarity']
                
                # Print row
                method_name = f"GPT-4o Semantic (t={threshold})" if persona_display == persona_order[0] else ""
                print(f"{method_name:<30} {persona_display:<25} {avg_mae:>8.2f} {avg_em:>8.2f} {avg_w1:>8.2f} {avg_pearson:>8.2f} "
                      f"{prec:>8.2f} {rec:>8.2f} {f1:>8.2f} {avg_sim:>8.2f}")
    
    def print_threshold_comparison(self, results_by_threshold):
        """Print comparison across different thresholds"""
        print("\n" + "="*100)
        print("THRESHOLD COMPARISON (Average across all personas)")
        print("="*100)
        print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Avg Top-K Sim':>15}")
        print("-"*100)
        
        for threshold in sorted(results_by_threshold.keys()):
            avg_results = results_by_threshold[threshold]['Average']
            prec = avg_results['factor_metrics']['precision']
            rec = avg_results['factor_metrics']['recall']
            f1 = avg_results['factor_metrics']['f1']
            avg_sim = avg_results['factor_metrics']['avg_similarity']
            
            print(f"{threshold:>10.1f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {avg_sim:>15.3f}")
    
    def save_detailed_results(self, results_by_threshold):
        """Save detailed results to JSON file"""
        # Convert numpy floats to Python floats for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results_by_threshold)
        
        with open('gpt4o_semantic_evaluation_results_topk.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nDetailed results saved to: gpt4o_semantic_evaluation_results_topk.json")


# Usage
if __name__ == "__main__":
    print("Running semantic similarity evaluation with top-K similarity metric...")
    print("Note: First run will download the sentence transformer model (~80MB)")
    
    evaluator = GPT4oSemanticEvaluator()
    
    # Test with multiple thresholds
    results = evaluator.evaluate_predictions(
        'gpt4o_baseline_predictions.json',
        similarity_thresholds=[0.5, 0.6, 0.7, 0.8]
    )