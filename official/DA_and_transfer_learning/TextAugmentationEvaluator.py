import numpy as np
import math
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from llama_cpp import Llama


# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class TextAugmentationEvaluator:
    """
    Comprehensive evaluation framework for text augmentation techniques
    based on the taxonomy from "Evaluation Metrics for Text Data Augmentation in NLP"
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the evaluator with a pre-trained model for various metrics

        Args:
            model_name: Name of the pre-trained model for evaluation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.smoothing_function = SmoothingFunction().method1

        # Initialize model if needed
        self._initialize_model()

    # def _initialize_model(self):
    #     """Initialize the model and tokenizer for evaluation"""
    #     try:
    #         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #         self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    #     except Exception as e:
    #         print(f"Warning: Could not load model {self.model_name}: {e}")
    #         print("Some metrics requiring model will be unavailable")

    def _initialize_model(self):
        """Initialize tokenizer and model"""
        try:
            if "gpt" in self.model_name.lower():
                from transformers import AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            else:
                from transformers import AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")

    # 1. MACHINE TRANSLATION QUALITY METRICS
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score between reference and candidate texts

        Args:
            reference: Original text
            candidate: Augmented text

        Returns:
            BLEU score (0-1)
        """
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())

        return sentence_bleu([ref_tokens], cand_tokens,
                             smoothing_function=self.smoothing_function)

    def calculate_macro_bleu(self, references: List[str], candidates: List[str]) -> float:
        """
        Calculate macro BLEU score for multiple text pairs

        Args:
            references: List of original texts
            candidates: List of augmented texts

        Returns:
            Macro BLEU score
        """
        if len(references) != len(candidates):
            raise ValueError("References and candidates must have the same length")

        bleu_scores = []
        for ref, cand in zip(references, candidates):
            bleu_scores.append(self.calculate_bleu_score(ref, cand))

        return np.mean(bleu_scores)

    # 2. TEXT GENERATION QUALITY METRICS
    def calculate_novelty(self, generated_text: str, corpus: List[str]) -> float:
        """
        Calculate novelty score using Jaccard similarity

        Args:
            generated_text: Generated/augmented text
            corpus: Original corpus

        Returns:
            Novelty score (0-1, higher = more novel)
        """
        gen_tokens = set(word_tokenize(generated_text.lower()))
        max_similarity = 0

        for text in corpus:
            corpus_tokens = set(word_tokenize(text.lower()))
            if len(gen_tokens.union(corpus_tokens)) > 0:
                jaccard_sim = len(gen_tokens.intersection(corpus_tokens)) / len(gen_tokens.union(corpus_tokens))
                max_similarity = max(max_similarity, jaccard_sim)

        return 1 - max_similarity

    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate diversity metrics: Self-BLEU, UTR, TTR, RWORDS

        Args:
            texts: List of generated/augmented texts

        Returns:
            Dictionary with diversity metrics
        """
        # Self-BLEU
        self_bleu_scores = []
        for i, text in enumerate(texts):
            references = [texts[j] for j in range(len(texts)) if j != i]
            if references:
                ref_tokens = [word_tokenize(ref.lower()) for ref in references]
                cand_tokens = word_tokenize(text.lower())
                self_bleu = sentence_bleu(ref_tokens, cand_tokens,
                                          smoothing_function=self.smoothing_function)
                self_bleu_scores.append(self_bleu)

        # Unique Trigrams Ratio (UTR)
        all_trigrams = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            all_trigrams.extend(trigrams)

        utr = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0

        # Type-Token Ratio (TTR)
        all_tokens = []
        for text in texts:
            all_tokens.extend(word_tokenize(text.lower()))

        ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

        # Rare Words (RWORDS) - simplified version
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        rare_threshold = 0.01 * total_tokens  # Words appearing less than 1% of total
        rare_words = sum(1 for count in token_counts.values() if count < rare_threshold)
        rwords = rare_words / len(token_counts) if token_counts else 0

        return {
            'self_bleu': np.mean(self_bleu_scores) if self_bleu_scores else 0,
            'utr': utr,
            'ttr': ttr,
            'rwords': rwords
        }

    def _initialize_model(self):
        """Initialize model and tokenizer (Hugging Face or local GGUF)."""
        try:
            if self.model_name.endswith(".gguf"):
                print(f"Loading local GGUF model: {self.model_name}")
                self.model = Llama(model_path=self.model_name, n_ctx=2048, n_threads=8, logits_all=True)
                self.tokenizer = None  # not needed for llama_cpp
            elif "gpt" in self.model_name.lower():
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            else:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")

    def calculate_perplexity(self, texts: List[str]) -> float:
        """Compute average perplexity. Works with HF or GGUF models."""
        if not self.model:
            print("Model not loaded.")
            return float('nan')

        total_logprob, total_tokens = 0.0, 0

        # CASE 1: HuggingFace-style causal LM (GPT, LLaMA HF, itp.)
        if hasattr(self.model, 'lm_head'):
            self.model.eval()
            with torch.no_grad():
                for text in texts:
                    if not text.strip():
                        continue
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    labels = inputs['input_ids'].clone()
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss.item()
                    total_logprob += loss * inputs['input_ids'].size(1)
                    total_tokens += inputs['input_ids'].size(1)

            return math.exp(total_logprob / total_tokens) if total_tokens > 0 else float('nan')

        # CASE 2: Local GGUF (llama_cpp)
        print("Calculating perplexity using llama_cpp...")
        for text in texts:
            if not isinstance(text, str) or len(text.strip()) < 5:
                continue
            try:
                # if native perplexity exists and works
                if hasattr(self.model, "perplexity"):
                    output = self.model.perplexity(text)
                    if isinstance(output, dict) and output.get("perplexity") is not None:
                        total_logprob += output["perplexity"] * len(text.split())
                        total_tokens += len(text.split())
                        continue  # skip to next text if worked

                # fallback manual perplexity calculation
                tokens = self.model.tokenize(text.encode("utf-8"))
                if len(tokens) < 2:
                    continue
                logits = self.model.eval(tokens[:-1])
                if logits is None or getattr(logits, "logits", None) is None:
                    print("Warning: logits unavailable, skipping text")
                    continue

                # compute pseudo-probabilities
                arr = np.array(logits["logits"])
                last_token_logits = arr[-1]
                probs = np.exp(last_token_logits - np.max(last_token_logits))
                probs /= np.sum(probs)
                avg_logprob = np.mean(np.log(probs + 1e-9))
                ppl = math.exp(-avg_logprob)

                total_logprob += ppl * len(tokens)
                total_tokens += len(tokens)
            except Exception as e:
                print(f"Error processing text: {e}")
                continue

        if total_tokens == 0:
            return float('nan')
        return total_logprob / total_tokens




    def calculate_spelling_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate spelling-related metrics (simplified version)

        Args:
            texts: List of texts to evaluate

        Returns:
            Dictionary with spelling metrics
        """
        # Simplified spelling check - count words with non-alphabetic characters
        # In practice, you'd use a proper spell checker library

        total_words = 0
        misspelled_words = 0
        total_chars = 0
        misspelled_chars = 0

        for text in texts:
            words = word_tokenize(text)
            total_words += len(words)

            for word in words:
                # Simple heuristic: words with numbers or special chars might be misspelled
                if not word.isalpha() and word.isalnum():
                    misspelled_words += 1
                    misspelled_chars += len(word)
                total_chars += len(word)

        return {
            'spell_words': misspelled_words / total_words if total_words > 0 else 0,
            'spell_chars': misspelled_chars / total_chars if total_chars > 0 else 0
        }

    # 3. CHARACTER N-GRAM MATCHES
    def calculate_chrf_score(self, reference: str, candidate: str, n: int = 6) -> float:
        """
        Calculate Character n-gram F-score (CHRF)

        Args:
            reference: Reference text
            candidate: Candidate text
            n: Maximum n-gram length

        Returns:
            CHRF score
        """
        def get_char_ngrams(text: str, n: int) -> List[str]:
            """Extract character n-grams from text"""
            ngrams = []
            for i in range(1, n + 1):
                for j in range(len(text) - i + 1):
                    ngrams.append(text[j:j + i])
            return ngrams

        ref_ngrams = Counter(get_char_ngrams(reference, n))
        cand_ngrams = Counter(get_char_ngrams(candidate, n))

        # Calculate precision and recall
        common_ngrams = ref_ngrams & cand_ngrams
        common_count = sum(common_ngrams.values())

        if sum(cand_ngrams.values()) == 0:
            precision = 0
        else:
            precision = common_count / sum(cand_ngrams.values())

        if sum(ref_ngrams.values()) == 0:
            recall = 0
        else:
            recall = common_count / sum(ref_ngrams.values())

        # F-score
        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    # 4. PREDICTION QUALITY FOR CLASSIFICATION
    def calculate_classification_metrics(self, y_true: List[int], y_pred: List[int],
                                         y_prob: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            Dictionary with classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'jaccard': jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        }

        if y_prob is not None and len(set(y_true)) == 2:  # Binary classification
            metrics['auc'] = roc_auc_score(y_true, y_prob)

        return metrics

    # 5. DATASET RELATIONSHIP METRICS
    def calculate_dataset_similarity(self, embeddings1: np.ndarray,
                                     embeddings2: np.ndarray) -> Dict[str, float]:
        """
        Calculate similarity between two datasets using embeddings

        Args:
            embeddings1: Embeddings from first dataset
            embeddings2: Embeddings from second dataset

        Returns:
            Dictionary with similarity metrics
        """
        # Average embeddings for each dataset
        avg_emb1 = np.mean(embeddings1, axis=0)
        avg_emb2 = np.mean(embeddings2, axis=0)

        # Cosine similarity
        cos_sim = cosine_similarity([avg_emb1], [avg_emb2])[0][0]

        # Spearman correlation (element-wise)
        spearman_corr, _ = spearmanr(avg_emb1, avg_emb2)

        return {
            'cosine_similarity': cos_sim,
            'spearman_correlation': spearman_corr
        }

    # 6. COMPREHENSIVE EVALUATION
    def evaluate_augmentation(self, original_texts: List[str],
                              augmented_texts: List[str],
                              labels: Optional[List[int]] = None,
                              model_predictions: Optional[Dict] = None) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive evaluation of text augmentation techniques

        Args:
            original_texts: List of original texts
            augmented_texts: List of augmented texts
            labels: Optional labels for classification evaluation
            model_predictions: Optional model predictions for classification metrics

        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}

        # 1. Translation Quality (BLEU)
        results['bleu_score'] = self.calculate_macro_bleu(original_texts, augmented_texts)

        # 2. Text Generation Quality
        results['diversity_metrics'] = self.calculate_diversity_metrics(augmented_texts)

        # Calculate novelty for each augmented text
        novelty_scores = []
        for aug_text in augmented_texts:
            novelty = self.calculate_novelty(aug_text, original_texts)
            novelty_scores.append(novelty)
        results['novelty_score'] = np.mean(novelty_scores)

        # Perplexity
        results['perplexity'] = self.calculate_perplexity(augmented_texts)

        # Spelling metrics
        results['spelling_metrics'] = self.calculate_spelling_metrics(augmented_texts)

        # 3. Character n-gram matches (CHRF)
        chrf_scores = []
        for orig, aug in zip(original_texts, augmented_texts):
            chrf_scores.append(self.calculate_chrf_score(orig, aug))
        results['chrf_score'] = np.mean(chrf_scores)

        # 4. Classification metrics (if provided)
        if labels is not None and model_predictions is not None:
            if 'predictions' in model_predictions:
                results['classification_metrics'] = self.calculate_classification_metrics(
                    labels, model_predictions['predictions'],
                    model_predictions.get('probabilities')
                )

        return results

    def print_evaluation_report(self, results: Dict) -> None:
        """
        Print a formatted evaluation report

        Args:
            results: Results dictionary from evaluate_augmentation
        """
        print("=" * 60)
        print("TEXT AUGMENTATION EVALUATION REPORT")
        print("=" * 60)

        print(f"\n1. TRANSLATION QUALITY")
        print(f"   BLEU Score: {results.get('bleu_score', 0):.4f}")

        print(f"\n2. TEXT GENERATION QUALITY")
        print(f"   Novelty Score: {results.get('novelty_score', 0):.4f}")

        if 'diversity_metrics' in results:
            div_metrics = results['diversity_metrics']
            print(f"   Diversity Metrics:")
            print(f"     Self-BLEU: {div_metrics.get('self_bleu', 0):.4f}")
            print(f"     UTR (Unique Trigrams Ratio): {div_metrics.get('utr', 0):.4f}")
            print(f"     TTR (Type-Token Ratio): {div_metrics.get('ttr', 0):.4f}")
            print(f"     RWORDS (Rare Words): {div_metrics.get('rwords', 0):.4f}")

        print(f"   Perplexity: {results.get('perplexity', 0):.4f}")

        if 'spelling_metrics' in results:
            spell_metrics = results['spelling_metrics']
            print(f"   Spelling Metrics:")
            print(f"     Misspelled Words Ratio: {spell_metrics.get('spell_words', 0):.4f}")
            print(f"     Misspelled Chars Ratio: {spell_metrics.get('spell_chars', 0):.4f}")

        print(f"\n3. CHARACTER N-GRAM MATCHES")
        print(f"   CHRF Score: {results.get('chrf_score', 0):.4f}")

        if 'classification_metrics' in results:
            print(f"\n4. CLASSIFICATION METRICS")
            clf_metrics = results['classification_metrics']
            print(f"   Accuracy: {clf_metrics.get('accuracy', 0):.4f}")
            print(f"   Precision (Macro): {clf_metrics.get('precision_macro', 0):.4f}")
            print(f"   Recall (Macro): {clf_metrics.get('recall_macro', 0):.4f}")
            print(f"   F1-Score (Macro): {clf_metrics.get('f1_macro', 0):.4f}")
            print(f"   Jaccard Score: {clf_metrics.get('jaccard', 0):.4f}")
            if 'auc' in clf_metrics:
                print(f"   AUC: {clf_metrics.get('auc', 0):.4f}")

        print("=" * 60)

# Example usage
def example_usage():
    """Example of how to use the TextAugmentationEvaluator"""

    # Initialize evaluator
    evaluator = TextAugmentationEvaluator(model_name="gpt2")
    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    #evaluator = TextAugmentationEvaluator(model_name=model_name)


# Example data
    ESCONV_PATH = '/data/esconv_both_parts.csv'
    MEISD_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/filtered_negative_MEISD_intensity_max_first_25_conv.csv'

    AUGMENTED_FILES = {
    'enhanced_classical_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/balanced datasets/esconv_enhanced_classical_augmentation_70percent_balanced.xlsx',
    'enhanced_mixed_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/balanced datasets/esconv_enhanced_mixed_augmentation_70percent_balanced.xlsx',
    'enhanced_llm_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/balanced datasets/esconv_enhanced_llm_augmentation_70percent_balanced.xlsx',
    'enhanced_nlp_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/balanced datasets/esconv_enhanced_nlp_augmentation_70percent_balanced.xlsx',
    'enhanced_nlp_llm_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/balanced datasets/esconv_enhanced_llm_nlp_augmentation_70percent_balanced.xlsx'
    }
    original_texts = ESCONV_PATH
    results_all = {}

    for name, path in AUGMENTED_FILES.items():
        print(f"\n=== Evaluating augmentation: {name} ===")
        df = pd.read_excel(path)
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content', 'conversation'])]
        if not text_cols:
            raise ValueError(f"No text column found in {name}")
        aug_texts = df[text_cols[0]].dropna().astype(str).tolist()

        # dopasuj długości
        min_len = min(len(original_texts), len(aug_texts))
        results = evaluator.evaluate_augmentation(original_texts[:min_len], aug_texts[:min_len])

        evaluator.print_evaluation_report(results)

    # results = evaluator.evaluate_augmentation(original_texts, AUGMENTED_FILES)
    #
    # # Print report
    # evaluator.print_evaluation_report(results)
    #
    # return results

if __name__ == "__main__":
    example_usage()