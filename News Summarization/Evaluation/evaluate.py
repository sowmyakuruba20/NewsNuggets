# Imports necessary for the evaluation metrics
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import os
import pickle

class TextEvaluator:
    def __init__(self):
        # Initialize Rouge once to use it for multiple evaluations
        self.rouge = Rouge()

    def calculate_rouge_bleu(self, reference, generated):
        # Calculate ROUGE scores
        scores = self.rouge.get_scores(generated, reference)[0]
        rouge_1 = scores['rouge-1']
        rouge_2 = scores['rouge-2']
        rouge_l = scores['rouge-l']

        # Tokenization for BLEU calculation
        reference_tokens = [word_tokenize(reference.lower())]
        generated_tokens = word_tokenize(generated.lower())

        # Calculate BLEU score
        bleu_score = sentence_bleu(reference_tokens, generated_tokens)

        # Returning results as a dictionary
        return {
            'rouge_1_f': rouge_1['f'],
            'rouge_1_precision': rouge_1['p'],
            'rouge_1_recall': rouge_1['r'],
            'rouge_2_f': rouge_2['f'],
            'rouge_2_precision': rouge_2['p'],
            'rouge_2_recall': rouge_2['r'],
            'rouge_l_f': rouge_l['f'],
            'rouge_l_precision': rouge_l['p'],
            'rouge_l_recall': rouge_l['r'],
            'bleu_score': bleu_score
        }
    
    def save_model(self, model, file_path):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

# # Example usage
# if __name__ == "__main__":
#     evaluator = TextEvaluator()
#     reference_text = "The quick brown fox jumps over the lazy dog."
#     generated_text = "A quick brown fox jumps over the lazy dog."
#     results = evaluator.calculate_rouge_bleu(reference_text, generated_text)
#     print(results)
