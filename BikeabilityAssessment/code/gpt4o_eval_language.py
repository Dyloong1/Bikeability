import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class GPT4oEvaluator:
    def __init__(self):
        # Initialize BERT model for BERT-Score calculation
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
    def clean_response(self, text: str) -> str:
        """Clean the full_response text by removing markdown artifacts and extra formatting"""
        # Remove markdown code blocks
        text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove markdown headers but keep the content
        text = re.sub(r'#{1,6}\s*', '', text)
        
        # Clean up bullet points
        text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def evaluate_factual_accuracy(self, response: dict, ground_truth: dict) -> float:
        """
        Evaluate factual accuracy by comparing extracted factors and ratings
        with ground truth indicators and ratings
        """
        score = 0.0
        total_checks = 0
        
        # Check if ratings match ground truth (with some tolerance)
        if 'ratings' in response and 'ground_truth' in response:
            pred_ratings = response['ratings']
            gt_ratings = response['ground_truth']
            
            for key in ['comfortable', 'safe', 'overall']:
                if key in pred_ratings and key in gt_ratings:
                    # Allow ±1 difference in ratings
                    diff = abs(pred_ratings[key] - gt_ratings[key])
                    if diff <= 1:
                        score += (1 - diff * 0.5)  # Full score for exact match, 0.5 for diff=1
                    total_checks += 1
        
        # Check if identified factors align with ground truth indicators
        if 'influencing_factors' in response and 'ground_truth_indicators' in response:
            factors = set(response['influencing_factors'])
            indicators = set(response['ground_truth_indicators'])
            
            # Map indicators to expected factors
            indicator_factor_map = {
                'bike_lane_quality': ['protected_lane', 'bike_lane', 'no_bike_lane'],
                'separation_moving': ['protected_lane', 'mixed_traffic'],
                'separation_parked': ['protected_lane', 'parked_cars'],
                'auto_volume': ['high_traffic', 'moderate_traffic', 'low_traffic'],
                'surrounding_environment': ['urban_environment', 'trees', 'greenery']
            }
            
            relevant_factors = set()
            for indicator in indicators:
                if indicator in indicator_factor_map:
                    relevant_factors.update(indicator_factor_map[indicator])
            
            if relevant_factors:
                overlap = len(factors.intersection(relevant_factors))
                score += overlap / len(relevant_factors)
                total_checks += 1
        
        return score / total_checks if total_checks > 0 else 0.0
    
    def evaluate_logical_coherence(self, full_response: str) -> float:
        """
        Evaluate logical coherence by checking structure and flow of reasoning
        """
        cleaned_response = self.clean_response(full_response)
        score = 0.0
        
        # Check for logical structure markers
        structure_markers = [
            'analysis', 'therefore', 'because', 'due to', 'as a result',
            'however', 'although', 'considering', 'based on', 'given that'
        ]
        
        marker_count = sum(1 for marker in structure_markers 
                          if marker.lower() in cleaned_response.lower())
        structure_score = min(marker_count / 5, 1.0)  # Normalize to max 1.0
        
        # Check for step-by-step reasoning
        numbered_steps = len(re.findall(r'\d+\.\s+\w+', cleaned_response))
        step_score = min(numbered_steps / 3, 1.0)  # Expect at least 3 steps
        
        # Check for conclusions that follow from premises
        has_ratings_justification = bool(re.search(
            r'(rating|score|overall|comfortable|safe).*?(because|due to|based on|considering)',
            cleaned_response, re.IGNORECASE
        ))
        
        # Combine scores
        score = (structure_score * 0.4 + step_score * 0.3 + 
                (0.3 if has_ratings_justification else 0.0))
        
        return score
    
    def evaluate_persona_consistency(self, response: dict) -> float:
        """
        Evaluate if the response is consistent with the persona characteristics
        """
        persona = response.get('persona', '').lower()
        factors = response.get('influencing_factors', [])
        ratings = response.get('ratings', {})
        
        score = 1.0  # Start with perfect score
        
        if 'enthused and confident' in persona:
            # This persona should be more tolerant of mixed traffic
            if 'no_bike_lane' in factors and ratings.get('overall', 0) == 1:
                score *= 0.8  # Slight penalty for being too conservative
            if 'protected_lane' in factors and ratings.get('overall', 0) < 3:
                score *= 0.7  # Should rate protected lanes highly
                
        elif 'interested but concerned' in persona:
            # This persona should be more cautious
            if 'high_traffic' in factors and ratings.get('safe', 0) > 3:
                score *= 0.7  # Should be concerned about high traffic
            if 'no_bike_lane' in factors and ratings.get('overall', 0) > 3:
                score *= 0.8  # Should prefer bike lanes
                
        # Add more persona-specific rules as needed
        
        return score
    
    def calculate_bert_score(self, text1: str, text2: str) -> float:
        """
        Calculate BERT-based similarity score between two texts
        """
        # Tokenize and get embeddings
        with torch.no_grad():
            tokens1 = self.tokenizer(text1, return_tensors='pt', 
                                    truncation=True, max_length=512, padding=True)
            tokens2 = self.tokenizer(text2, return_tensors='pt', 
                                    truncation=True, max_length=512, padding=True)
            
            outputs1 = self.model(**tokens1)
            outputs2 = self.model(**tokens2)
            
            # Use CLS token embeddings
            embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
            embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
        return similarity
    
    def evaluate_sample(self, sample: dict, reference_samples: List[dict] = None) -> Dict[str, float]:
        """
        Evaluate a single sample across all metrics
        """
        results = {
            'factual_accuracy': self.evaluate_factual_accuracy(sample, sample),
            'logical_coherence': self.evaluate_logical_coherence(sample.get('full_response', '')),
            'persona_consistency': self.evaluate_persona_consistency(sample)
        }
        
        # Calculate BERT-Score against reference samples if provided
        if reference_samples and 'full_response' in sample:
            bert_scores = []
            sample_response = self.clean_response(sample['full_response'])
            
            for ref in reference_samples[:3]:  # Compare with up to 3 reference samples
                if 'full_response' in ref and ref.get('persona') == sample.get('persona'):
                    ref_response = self.clean_response(ref['full_response'])
                    if ref_response and sample_response:
                        bert_scores.append(self.calculate_bert_score(sample_response, ref_response))
            
            results['bert_score'] = np.mean(bert_scores) if bert_scores else 0.0
        else:
            results['bert_score'] = 0.0
            
        return results
    
    def evaluate_dataset(self, json_file: str, reference_file: str = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate entire dataset and group results by persona
        """
        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load reference data if provided
        reference_data = []
        if reference_file:
            with open(reference_file, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
        
        # Group by persona
        persona_results = defaultdict(list)
        
        # Evaluate each sample
        for sample in data:
            persona = sample.get('persona', 'Unknown')
            
            # Find reference samples for the same location/persona
            ref_samples = [r for r in reference_data 
                          if r.get('location_id') == sample.get('location_id')]
            
            metrics = self.evaluate_sample(sample, ref_samples or reference_data)
            persona_results[persona].append(metrics)
        
        # Calculate averages by persona
        persona_averages = {}
        for persona, results in persona_results.items():
            if results:
                avg_metrics = {
                    'factual_accuracy': np.mean([r['factual_accuracy'] for r in results]),
                    'logical_coherence': np.mean([r['logical_coherence'] for r in results]),
                    'persona_consistency': np.mean([r['persona_consistency'] for r in results]),
                    'bert_score': np.mean([r['bert_score'] for r in results]),
                    'sample_count': len(results)
                }
                persona_averages[persona] = avg_metrics
        
        return persona_averages

def main():
    """
    Main function to run the evaluation
    """
    print("GPT-4o Baseline Evaluation Script")
    print("=" * 50)
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = GPT4oEvaluator()
    
    # File paths
    predictions_file = 'gpt4o_baseline_predictions.json'
    reference_file = 'gpt4o_baseline_predictions.json'  # Optional: use the provided samples as reference
    
    try:
        # Run evaluation
        print(f"\nEvaluating {predictions_file}...")
        results = evaluator.evaluate_dataset(predictions_file, reference_file)
        
        # Print results
        print("\nEvaluation Results by Persona:")
        print("=" * 50)
        
        for persona, metrics in results.items():
            print(f"\n{persona}:")
            print(f"  Samples: {metrics['sample_count']}")
            print(f"  Factual Accuracy: {metrics['factual_accuracy']:.3f}")
            print(f"  Logical Coherence: {metrics['logical_coherence']:.3f}")
            print(f"  Persona Consistency: {metrics['persona_consistency']:.3f}")
            print(f"  BERT-Score: {metrics['bert_score']:.3f}")
        
        # Calculate overall averages
        print("\nOverall Averages:")
        print("=" * 50)
        total_samples = sum(m['sample_count'] for m in results.values())
        
        # Weighted averages
        overall_metrics = {
            'factual_accuracy': sum(m['factual_accuracy'] * m['sample_count'] 
                                  for m in results.values()) / total_samples,
            'logical_coherence': sum(m['logical_coherence'] * m['sample_count'] 
                                   for m in results.values()) / total_samples,
            'persona_consistency': sum(m['persona_consistency'] * m['sample_count'] 
                                     for m in results.values()) / total_samples,
            'bert_score': sum(m['bert_score'] * m['sample_count'] 
                            for m in results.values()) / total_samples
        }
        
        print(f"Total Samples: {total_samples}")
        print(f"Factual Accuracy (Acc.): {overall_metrics['factual_accuracy']:.3f}")
        print(f"Logical Coherence (Coh.): {overall_metrics['logical_coherence']:.3f}")
        print(f"Persona Consistency (Cons.): {overall_metrics['persona_consistency']:.3f}")
        print(f"BERT-Score (BERT-S): {overall_metrics['bert_score']:.3f}")
        
        # Save results to file
        output_file = 'evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'persona_results': results,
                'overall_metrics': overall_metrics
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"\nError: Could not find file - {e}")
        print("Please ensure 'gpt4o_baseline_predictions.json' exists in the same directory.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
