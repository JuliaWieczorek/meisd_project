"""
Analyzer jakości augmentacji danych ESConv-MEISD
Analizuje jakość przeprowadzonych augmentacji przy użyciu istniejących metod
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Importowanie klas z oryginalnego kodu
from data_augmentation import EnhancedESConvProcessor, EnhancedMEISDDataAugmenter

from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, jaccard_score
from scipy.stats import spearmanr
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import wordnet as wn

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("VADER not available. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

try:
    from nrclex import NRCLex
    NRCLEX_AVAILABLE = True
except ImportError:
    print("NRCLex not available. Install with: pip install NRCLex")
    NRCLEX_AVAILABLE = False


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


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

    def _initialize_model(self):
        """Initialize the model and tokenizer for evaluation"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")
            print("Some metrics requiring model will be unavailable")

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
        ref_tokens = simple_tokenize(reference.lower())
        cand_tokens = simple_tokenize(candidate.lower())

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
        gen_tokens = set(simple_tokenize(generated_text.lower()))
        max_similarity = 0

        for text in corpus:
            corpus_tokens = set(simple_tokenize(text.lower()))
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
                ref_tokens = [simple_tokenize(ref.lower()) for ref in references]
                cand_tokens = simple_tokenize(text.lower())
                self_bleu = sentence_bleu(ref_tokens, cand_tokens,
                                          smoothing_function=self.smoothing_function)
                self_bleu_scores.append(self_bleu)

        # Unique Trigrams Ratio (UTR)
        all_trigrams = []
        for text in texts:
            tokens = simple_tokenize(text.lower())
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            all_trigrams.extend(trigrams)

        utr = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0

        # Type-Token Ratio (TTR)
        all_tokens = []
        for text in texts:
            all_tokens.extend(simple_tokenize(text.lower()))

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

    def calculate_perplexity(self, texts: List[str]) -> float:
        """
        Calculate perplexity using a language model

        Args:
            texts: List of texts to evaluate

        Returns:
            Average perplexity score
        """
        if not self.model or not self.tokenizer:
            print("Model not available for perplexity calculation")
            return 0.0

        total_loss = 0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

                # For models without labels, we need to shift inputs for language modeling
                if hasattr(self.model, 'transformer'):  # For GPT-like models
                    labels = inputs['input_ids'].clone()
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                else:
                    # For BERT-like models, we'll skip perplexity calculation
                    continue

                total_loss += loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)

        if total_tokens == 0:
            return 0.0

        return math.exp(total_loss / total_tokens)

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
            words = simple_tokenize(text)
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

class AugmentationQualityAnalyzer:
    """
    Analyzer jakości augmentacji danych
    """

    def __init__(self, esconv_path, meisd_path):
        self.esconv_path = esconv_path
        self.meisd_path = meisd_path
        self.esconv_processor = None
        self.meisd_augmenter = None
        self.original_esconv = None
        self.original_meisd = None
        self.augmented_datasets = {}
        self.analysis_results = {}

    def setup(self):
        """Inicjalizacja analizatora"""
        print("=== Inicjalizacja analizatora jakości augmentacji ===")

        # Setup ESConv processor
        self.esconv_processor = EnhancedESConvProcessor(self.esconv_path)
        self.original_esconv = self.esconv_processor.load_data()
        classification_data = self.esconv_processor.prepare_for_classification()

        # Setup MEISD augmenter (potrzebne do metod analizy)
        self.meisd_augmenter = EnhancedMEISDDataAugmenter(
            self.meisd_path, self.esconv_processor
        )
        self.meisd_augmenter.setup()

        return classification_data

    def load_augmented_datasets(self, file_paths):
        """Wczytanie zaugmentowanych datassetów"""
        print("\n=== Wczytywanie zaugmentowanych datassetów ===")

        for name, path in file_paths.items():
            if os.path.exists(path):
                try:
                    self.augmented_datasets[name] = pd.read_excel(path)
                    print(f"Wczytano {name}: {len(self.augmented_datasets[name])} próbek")
                except Exception as e:
                    print(f"Błąd wczytywania {name}: {e}")
            else:
                print(f"Plik {name} nie istnieje: {path}")

    def analyze_basic_statistics(self):
        """Analiza podstawowych statystyk"""
        print("\n=== Analiza podstawowych statystyk ===")

        stats_results = {}

        for dataset_name, dataset in self.augmented_datasets.items():
            print(f"\n--- Analiza {dataset_name} ---")

            # Podstawowe statystyki
            total_samples = len(dataset)
            class_distribution = dataset['label'].value_counts().to_dict()
            intensity_distribution = dataset['intensity'].value_counts().to_dict()

            # Statystyki tekstu
            text_lengths = [len(str(text).split()) for text in dataset['conversation']]
            avg_length = np.mean(text_lengths)
            std_length = np.std(text_lengths)

            # Analiza unikalności
            unique_texts = len(dataset['conversation'].unique())
            uniqueness_ratio = unique_texts / total_samples

            stats_results[dataset_name] = {
                'total_samples': total_samples,
                'class_distribution': class_distribution,
                'intensity_distribution': intensity_distribution,
                'avg_text_length': avg_length,
                'std_text_length': std_length,
                'unique_texts': unique_texts,
                'uniqueness_ratio': uniqueness_ratio,
                'balance_ratio': min(class_distribution.values()) / max(class_distribution.values())
            }

            print(f"  Liczba próbek: {total_samples}")
            print(f"  Rozkład klas: {class_distribution}")
            print(f"  Średnia długość tekstu: {avg_length:.1f} ± {std_length:.1f} słów")
            print(f"  Unikalność tekstów: {uniqueness_ratio:.3f}")
            print(f"  Balans klas: {stats_results[dataset_name]['balance_ratio']:.3f}")

        self.analysis_results['basic_stats'] = stats_results
        return stats_results

    def analyze_transformation_quality(self):
        """Analiza jakości transformacji używając istniejących metod"""
        print("\n=== Analiza jakości transformacji ===")

        quality_results = {}

        for dataset_name, dataset in self.augmented_datasets.items():
            print(f"\n--- Analiza jakości transformacji {dataset_name} ---")

            # Identyfikacja zaugmentowanych próbek (te które nie pochodzą z oryginalnego ESConv)
            original_classification = self.esconv_processor.prepare_for_classification()
            original_texts = set(original_classification['Utterances'].str.lower().str.strip())

            augmented_samples = dataset[
                ~dataset['conversation'].str.lower().str.strip().isin(original_texts)
            ]

            print(f"  Znaleziono {len(augmented_samples)} zaugmentowanych próbek")

            if len(augmented_samples) == 0:
                print("  Brak zaugmentowanych próbek do analizy")
                continue

            # Analiza jakości dla każdej próbki
            quality_scores = []

            for _, row in tqdm(augmented_samples.iterrows(),
                               total=len(augmented_samples),
                               desc=f"Analizowanie jakości {dataset_name}"):

                # Znajdź najbardziej podobną oryginalną próbkę MEISD
                original_sample = self._find_most_similar_meisd_sample(row['conversation'])

                if original_sample is not None:
                    # Użyj istniejącej metody oceny jakości
                    quality = self.meisd_augmenter._calculate_transformation_quality(
                        original_sample, row['conversation'], row['intensity']
                    )
                    quality_scores.append(quality)

            if quality_scores:
                avg_quality = np.mean(quality_scores)
                std_quality = np.std(quality_scores)

                quality_results[dataset_name] = {
                    'avg_quality': avg_quality,
                    'std_quality': std_quality,
                    'quality_scores': quality_scores,
                    'num_analyzed': len(quality_scores)
                }

                print(f"  Średnia jakość transformacji: {avg_quality:.3f} ± {std_quality:.3f}")
                print(f"  Zakres jakości: {min(quality_scores):.3f} - {max(quality_scores):.3f}")

        self.analysis_results['transformation_quality'] = quality_results
        return quality_results

    def _find_most_similar_meisd_sample(self, transformed_text):
        """Znajdź najbardziej podobną oryginalną próbkę MEISD"""
        try:
            if self.meisd_augmenter.meisd_data is None:
                return None

            # Prosty heurystyk: znajdź próbkę o podobnej długości
            target_length = len(transformed_text.split())
            meisd_texts = self.meisd_augmenter.meisd_data['conversation'].astype(str)

            # Znajdź próbkę o najbardziej podobnej długości
            length_diffs = [abs(len(text.split()) - target_length) for text in meisd_texts]
            best_idx = np.argmin(length_diffs)

            return meisd_texts.iloc[best_idx]

        except Exception as e:
            print(f"Błąd w znajdowaniu podobnej próbki: {e}")
            return None

    def analyze_style_consistency(self):
        """Analiza konsystencji stylu z ESConv"""
        print("\n=== Analiza konsystencji stylu z ESConv ===")

        style_results = {}

        # Pobierz wzorce stylu ESConv
        esconv_patterns = self.esconv_processor.style_patterns

        for dataset_name, dataset in self.augmented_datasets.items():
            print(f"\n--- Analiza stylu {dataset_name} ---")

            style_scores = {'low': [], 'high': []}

            for intensity in ['low', 'high']:
                intensity_samples = dataset[dataset['intensity'] == intensity]

                if len(intensity_samples) == 0:
                    continue

                # Analiza wzorców dla danej intensywności
                target_patterns = esconv_patterns.get(intensity, {})

                for _, row in intensity_samples.iterrows():
                    text = str(row['conversation'])
                    score = self._calculate_style_consistency(text, target_patterns)
                    style_scores[intensity].append(score)

            # Oblicz średnie wyniki
            avg_scores = {}
            for intensity in ['low', 'high']:
                if style_scores[intensity]:
                    avg_scores[intensity] = np.mean(style_scores[intensity])
                else:
                    avg_scores[intensity] = 0

            style_results[dataset_name] = {
                'avg_style_scores': avg_scores,
                'style_scores': style_scores
            }

            print(f"  Konsystencja stylu 'low': {avg_scores['low']:.3f}")
            print(f"  Konsystencja stylu 'high': {avg_scores['high']:.3f}")

        self.analysis_results['style_consistency'] = style_results
        return style_results

    def _calculate_style_consistency(self, text, target_patterns):
        """Oblicz konsystencję stylu z wzorcami ESConv"""
        if not target_patterns:
            return 0.5

        score = 0.0
        num_metrics = 0

        # Sprawdź długość tekstu
        text_length = len(text.split())
        target_length = target_patterns.get('avg_length', 50)
        if target_length > 0:
            length_score = 1.0 - min(abs(text_length - target_length) / target_length, 1.0)
            score += length_score
            num_metrics += 1

        # Sprawdź starter sentences
        starters = target_patterns.get('sentence_starters', [])
        if starters:
            first_word = text.split()[0].lower() if text.split() else ''
            starter_score = 1.0 if first_word in starters else 0.0
            score += starter_score
            num_metrics += 1

        # Sprawdź zaimki osobowe
        personal_pronouns = target_patterns.get('personal_pronouns', [])
        if personal_pronouns:
            text_lower = text.lower()
            pronoun_matches = sum(1 for pronoun, _ in personal_pronouns
                                  if pronoun in text_lower)
            pronoun_score = min(pronoun_matches / 3.0, 1.0)
            score += pronoun_score
            num_metrics += 1

        return score / num_metrics if num_metrics > 0 else 0.5

    def analyze_nlp_metrics(self):
        """Analiza metryk NLP (BLEU, CHRF, Novelty, Perplexity, itd.)"""
        print("\n=== Analiza metryk NLP (BLEU, CHRF, Novelty, Perplexity, itd.) ===")

        evaluator = TextAugmentationEvaluator()
        nlp_results = {}

        for dataset_name, dataset in self.augmented_datasets.items():
            print(f"\n--- NLP ocena: {dataset_name} ---")

            # Pobierz oryginalne teksty ESConv
            original_classification = self.esconv_processor.prepare_for_classification()
            original_set = set(original_classification['Utterances'].str.lower().str.strip())

            # Wyciągnij tylko zaugmentowane próbki
            augmented_only = dataset[
                ~dataset['conversation'].str.lower().str.strip().isin(original_set)
            ]

            if augmented_only.empty:
                print(f"  Brak zaugmentowanych przykładów — pomijanie {dataset_name}")
                continue

            augmented_texts = augmented_only['conversation'].astype(str).tolist()

        # UWAGA: Tu można dopasować oryginalne teksty — ale tymczasowo robimy dummy-copy
        # Jeśli masz kolumnę 'source_text', zamień poniższą linię na:
        # original_texts = augmented_only['source_text'].astype(str).tolist()
            if 'source_text' in augmented_only.columns:
                original_texts = augmented_only['source_text'].astype(str).tolist()
            else:
                original_texts = augmented_texts.copy()


            labels = augmented_only['label'].tolist() if 'label' in augmented_only.columns else None
            predictions = augmented_only['predicted'] if 'predicted' in augmented_only.columns else None
            probs = augmented_only['probability'] if 'probability' in augmented_only.columns else None

            if predictions is not None and not predictions.isnull().all():
                model_predictions = {
                    'predictions': predictions.tolist(),
                    'probabilities': probs.tolist() if probs is not None and not probs.isnull().all() else None
                }
                print(f"Brak predykcji — pomijam metryki klasyfikacji dla {dataset_name}")
            else:
                model_predictions = None


            result = evaluator.evaluate_augmentation(
                original_texts=original_texts,
                augmented_texts=augmented_texts,
                labels=labels,
                model_predictions=model_predictions
            )

            evaluator.print_evaluation_report(result)
            nlp_results[dataset_name] = result

        self.analysis_results['nlp_metrics'] = nlp_results
        return nlp_results

    def analyze_emotion_intensity_consistency(self):
        """Analiza zgodności intensywności emocji pomiędzy source_text i conversation"""
        print("\n=== Analiza intensywności emocji (SENTDIFF, SENTSTD) ===")

        from augmentation_quality_analyzer import EnhancedEmotionIntensityEvaluator
        evaluator = EnhancedEmotionIntensityEvaluator()

        results = {}

        for dataset_name, dataset in self.augmented_datasets.items():
            print(f"\n--- Analiza {dataset_name} ---")

            if 'source_text' not in dataset.columns:
                print(f"  Pominięto {dataset_name} (brak kolumny 'source_text')")
                continue

            original_texts = dataset['source_text'].astype(str).tolist()
            augmented_texts = dataset['conversation'].astype(str).tolist()
            original_labels = dataset['intensity'].astype(str).tolist()

            consistency = evaluator.calculate_emotion_intensity_consistency_advanced(
                original_texts, augmented_texts, original_labels
            )

            results[dataset_name] = consistency

            # Zapisz jako osobny plik CSV
            os.makedirs("analysis_results", exist_ok=True)
            df = pd.DataFrame([consistency])
            df.insert(0, 'dataset', dataset_name)
            df.to_csv(f"analysis_results/emotion_consistency_{dataset_name}.csv", index=False)

        # Zbiorczy plik dla wszystkich datasetów
        combined_df = pd.DataFrame([
            {'dataset': name, **metrics}
            for name, metrics in results.items()
        ])
        combined_df.to_csv("analysis_results/emotion_consistency_summary.csv", index=False)
        print("Zapisano zbiorczy plik: analysis_results/emotion_consistency_summary.csv")

        self.analysis_results['emotion_intensity_consistency'] = results
        return results



    def compare_datasets(self):
        """Porównanie różnych metod augmentacji"""
        print("\n=== Porównanie różnych metod augmentacji ===")

        comparison_results = {}

        # Porównaj podstawowe metryki
        methods = list(self.augmented_datasets.keys())

        comparison_data = {
            'method': [],
            'total_samples': [],
            'balance_ratio': [],
            'uniqueness_ratio': [],
            'avg_text_length': [],
            'avg_quality': [],
            'style_consistency_low': [],
            'style_consistency_high': [],
            'bleu_score': [],
            'chrf_score': [],
            'novelty_score': [],
            'perplexity': []
        }

        for method in methods:
            comparison_data['method'].append(method)

            # Podstawowe statystyki
            basic_stats = self.analysis_results.get('basic_stats', {}).get(method, {})
            comparison_data['total_samples'].append(basic_stats.get('total_samples', 0))
            comparison_data['balance_ratio'].append(basic_stats.get('balance_ratio', 0))
            comparison_data['uniqueness_ratio'].append(basic_stats.get('uniqueness_ratio', 0))
            comparison_data['avg_text_length'].append(basic_stats.get('avg_text_length', 0))

            # Jakość transformacji
            quality_stats = self.analysis_results.get('transformation_quality', {}).get(method, {})
            comparison_data['avg_quality'].append(quality_stats.get('avg_quality', 0))

            # Konsystencja stylu
            style_stats = self.analysis_results.get('style_consistency', {}).get(method, {})
            style_scores = style_stats.get('avg_style_scores', {})
            comparison_data['style_consistency_low'].append(style_scores.get('low', 0))
            comparison_data['style_consistency_high'].append(style_scores.get('high', 0))

            nlp_metrics = self.analysis_results.get('nlp_metrics', {}).get(method, {})
            comparison_data['bleu_score'].append(nlp_metrics.get('bleu_score', 0))
            comparison_data['chrf_score'].append(nlp_metrics.get('chrf_score', 0))
            comparison_data['novelty_score'].append(nlp_metrics.get('novelty_score', 0))
            comparison_data['perplexity'].append(nlp_metrics.get('perplexity', 0))


        comparison_df = pd.DataFrame(comparison_data)
        comparison_results['comparison_table'] = comparison_df

        print("\n--- Tabela porównawcza ---")
        print(comparison_df.to_string(index=False))

        # Ranking metod
        ranking_scores = []
        for i, method in enumerate(methods):
            # Oblicz łączny wynik (można dostosować wagi)
            total_score = (
                    comparison_df.iloc[i]['balance_ratio'] * 0.1 +
                    comparison_df.iloc[i]['uniqueness_ratio'] * 0.1 +
                    comparison_df.iloc[i]['avg_quality'] * 0.2 +
                    comparison_df.iloc[i]['style_consistency_low'] * 0.1 +
                    comparison_df.iloc[i]['style_consistency_high'] * 0.1 +
                    comparison_df.iloc[i]['bleu_score'] * 0.15 +
                    comparison_df.iloc[i]['chrf_score'] * 0.1 +
                    comparison_df.iloc[i]['novelty_score'] * 0.1 +
                    (1 / (comparison_df.iloc[i]['perplexity'] + 1e-6)) * 0.05
            )
            ranking_scores.append((method, total_score))

        ranking_scores.sort(key=lambda x: x[1], reverse=True)
        comparison_results['ranking'] = ranking_scores

        print("\n--- Ranking metod augmentacji ---")
        for i, (method, score) in enumerate(ranking_scores):
            print(f"{i+1}. {method}: {score:.3f}")

        self.analysis_results['comparison'] = comparison_results

        ranking = comparison_results['ranking']
        ranking_df = pd.DataFrame(ranking, columns=['method', 'score'])
        ranking_df.to_csv('analysis_results/final_ranking.csv', index=False)
        print("Ranking zapisany do: analysis_results/final_ranking.csv")

        return comparison_results

    def generate_visualizations(self, output_dir='analysis_results'):
        """Generowanie wizualizacji"""
        print(f"\n=== Generowanie wizualizacji ===")

        os.makedirs(output_dir, exist_ok=True)

        num_datasets = len(self.augmented_datasets)
        cols = 2
        rows = (num_datasets + 1) // cols


# 1. Rozkład długości tekstów
        plt.figure(figsize=(6 * cols, 4 * rows))

        for i, (dataset_name, dataset) in enumerate(self.augmented_datasets.items()):
            text_lengths = [len(str(text).split()) for text in dataset['conversation']]
            plt.subplot(rows, cols, i + 1)
            plt.hist(text_lengths, bins=30, alpha=0.7, label=dataset_name)
            plt.xlabel('Długość tekstu (słowa)')
            plt.ylabel('Częstość')
            plt.title(f'Rozkład długości tekstów - {dataset_name}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/text_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Rozkład klas
        plt.figure(figsize=(15, 5))

        for i, (dataset_name, dataset) in enumerate(self.augmented_datasets.items()):
            plt.subplot(1, len(self.augmented_datasets), i+1)
            class_counts = dataset['label'].value_counts()
            plt.bar(class_counts.index, class_counts.values)
            plt.xlabel('Klasa')
            plt.ylabel('Liczba próbek')
            plt.title(f'Rozkład klas - {dataset_name}')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Porównanie jakości transformacji
        if 'transformation_quality' in self.analysis_results:
            quality_data = self.analysis_results['transformation_quality']

            methods = []
            avg_qualities = []
            std_qualities = []

            for method, data in quality_data.items():
                methods.append(method)
                avg_qualities.append(data['avg_quality'])
                std_qualities.append(data['std_quality'])

            plt.figure(figsize=(10, 6))
            plt.bar(methods, avg_qualities, yerr=std_qualities, capsize=5)
            plt.xlabel('Augmentation methods')
            plt.ylabel('Average transformation quality')
            plt.title('Comparison of transformation quality')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/transformation_quality_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        self.generate_radar_plot(output_dir)

        print(f"Wizualizacje zapisane w katalogu: {output_dir}")

    def generate_radar_plot(self, output_dir='analysis_results'):
        import matplotlib.pyplot as plt
        from math import pi

        df = self.analysis_results['comparison']['comparison_table'].copy()
        df.set_index('method', inplace=True)

        metrics = ['avg_quality', 'style_consistency_low', 'style_consistency_high',
                   'bleu_score', 'chrf_score', 'novelty_score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for method in df.index:
            values = df.loc[method, metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, label=method)
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_title('Porównanie metod augmentacji (radar plot)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig(f'{output_dir}/radar_plot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


    def save_analysis_report(self, output_file='analysis_report.txt'):
        """Zapisanie raportu analizy"""
        print(f"\n=== Zapisywanie raportu analizy ===")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RAPORT ANALIZY JAKOŚCI AUGMENTACJI DANYCH\n")
            f.write("=" * 50 + "\n\n")

            # Podstawowe statystyki
            f.write("1. PODSTAWOWE STATYSTYKI\n")
            f.write("-" * 25 + "\n")

            if 'basic_stats' in self.analysis_results:
                for dataset_name, stats in self.analysis_results['basic_stats'].items():
                    f.write(f"\n{dataset_name}:\n")
                    f.write(f"  Liczba próbek: {stats['total_samples']}\n")
                    f.write(f"  Rozkład klas: {stats['class_distribution']}\n")
                    f.write(f"  Średnia długość tekstu: {stats['avg_text_length']:.1f} ± {stats['std_text_length']:.1f} słów\n")
                    f.write(f"  Unikalność tekstów: {stats['uniqueness_ratio']:.3f}\n")
                    f.write(f"  Balans klas: {stats['balance_ratio']:.3f}\n")

            # Jakość transformacji
            f.write("\n\n2. JAKOŚĆ TRANSFORMACJI\n")
            f.write("-" * 25 + "\n")

            if 'transformation_quality' in self.analysis_results:
                for dataset_name, quality in self.analysis_results['transformation_quality'].items():
                    f.write(f"\n{dataset_name}:\n")
                    f.write(f"  Średnia jakość: {quality['avg_quality']:.3f} ± {quality['std_quality']:.3f}\n")
                    f.write(f"  Liczba analizowanych próbek: {quality['num_analyzed']}\n")

            # Konsystencja stylu
            f.write("\n\n3. KONSYSTENCJA STYLU\n")
            f.write("-" * 25 + "\n")

            if 'style_consistency' in self.analysis_results:
                for dataset_name, style in self.analysis_results['style_consistency'].items():
                    f.write(f"\n{dataset_name}:\n")
                    avg_scores = style['avg_style_scores']
                    f.write(f"  Konsystencja stylu 'low': {avg_scores['low']:.3f}\n")
                    f.write(f"  Konsystencja stylu 'high': {avg_scores['high']:.3f}\n")

            f.write("\n\n5. METRYKI NLP\n")
            f.write("-" * 25 + "\n")
            if 'nlp_metrics' in self.analysis_results:
                for dataset_name, metrics in self.analysis_results['nlp_metrics'].items():
                    f.write(f"\n{dataset_name}:\n")
                    f.write(f"  BLEU Score: {metrics.get('bleu_score', 0):.4f}\n")
                    f.write(f"  CHRF Score: {metrics.get('chrf_score', 0):.4f}\n")
                    f.write(f"  Novelty Score: {metrics.get('novelty_score', 0):.4f}\n")
                    f.write(f"  Perplexity: {metrics.get('perplexity', 0):.4f}\n")

            f.write("\n\n6. ZGODNOŚĆ INTENSYWNOŚCI EMOCJI\n")
            f.write("-" * 25 + "\n")

            if 'emotion_intensity_consistency' in self.analysis_results:
                for dataset_name, metrics in self.analysis_results['emotion_intensity_consistency'].items():
                    f.write(f"\n{dataset_name}:\n")
                    for key, val in metrics.items():
                        f.write(f"  {key}: {val:.4f}\n")

            # Ranking
            f.write("\n\n4. RANKING METOD AUGMENTACJI\n")
            f.write("-" * 25 + "\n")

            if 'comparison' in self.analysis_results:
                ranking = self.analysis_results['comparison']['ranking']
                for i, (method, score) in enumerate(ranking):
                    f.write(f"{i+1}. {method}: {score:.3f}\n")

        print(f"Raport zapisany w pliku: {output_file}")

class AdvancedEmotionIntensityAnalyzer:
    """
    Advanced emotion intensity analyzer using multiple established methods
    """

    def __init__(self):
        self.vader_analyzer = None
        self.emotion_models = {}
        self.lexicons = {}
        self.initialize_analyzers()

    def initialize_analyzers(self):
        """Initialize all available analyzers and lexicons"""

        # 1. VADER Sentiment Analyzer
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            print("✓ VADER Sentiment Analyzer initialized")

        # 2. Transformer-based emotion models
        try:
            self.emotion_models['roberta'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            print("✓ RoBERTa emotion model initialized")
        except Exception as e:
            print(f"× Failed to load RoBERTa emotion model: {e}")

        try:
            self.emotion_models['bert'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            print("✓ BERT sentiment model initialized")
        except Exception as e:
            print(f"× Failed to load BERT sentiment model: {e}")

        # 3. Load NRC Affect Intensity Lexicon
        self.load_nrc_affect_intensity_lexicon()

        # 4. Load other intensity lexicons
        self.load_additional_lexicons()

    def load_nrc_affect_intensity_lexicon(self):
        """Load NRC Affect Intensity Lexicon - the gold standard for emotion intensity"""
        try:
            # This would normally be downloaded from official NRC site
            # For demo purposes, we create a sample subset
            nrc_intensity_sample = {
                # High intensity words
                'ecstatic': {'joy': 0.95, 'valence': 0.9},
                'furious': {'anger': 0.98, 'valence': -0.9},
                'devastated': {'sadness': 0.92, 'valence': -0.85},
                'terrified': {'fear': 0.94, 'valence': -0.8},
                'thrilled': {'joy': 0.88, 'valence': 0.85},
                'outraged': {'anger': 0.91, 'valence': -0.87},
                'delighted': {'joy': 0.82, 'valence': 0.8},
                'horrified': {'fear': 0.89, 'valence': -0.82},

                # Medium intensity words
                'happy': {'joy': 0.65, 'valence': 0.6},
                'angry': {'anger': 0.7, 'valence': -0.65},
                'sad': {'sadness': 0.68, 'valence': -0.6},
                'worried': {'fear': 0.62, 'valence': -0.5},
                'pleased': {'joy': 0.58, 'valence': 0.55},
                'annoyed': {'anger': 0.55, 'valence': -0.5},
                'disappointed': {'sadness': 0.6, 'valence': -0.55},
                'concerned': {'fear': 0.52, 'valence': -0.4},

                # Low intensity words
                'content': {'joy': 0.35, 'valence': 0.3},
                'irritated': {'anger': 0.4, 'valence': -0.35},
                'melancholy': {'sadness': 0.38, 'valence': -0.3},
                'uneasy': {'fear': 0.35, 'valence': -0.25},
                'satisfied': {'joy': 0.42, 'valence': 0.4},
                'bothered': {'anger': 0.38, 'valence': -0.3},
                'wistful': {'sadness': 0.32, 'valence': -0.2},
                'cautious': {'fear': 0.3, 'valence': -0.15}
            }

            self.lexicons['nrc_intensity'] = nrc_intensity_sample
            print("✓ NRC Affect Intensity Lexicon loaded (sample)")

        except Exception as e:
            print(f"× Failed to load NRC Affect Intensity Lexicon: {e}")

    def load_additional_lexicons(self):
        """Load additional intensity-aware lexicons"""

        # Intensity modifiers and amplifiers
        self.lexicons['intensity_modifiers'] = {
            # Amplifiers with intensity scores
            'extremely': 2.0, 'incredibly': 1.9, 'absolutely': 1.8,
            'completely': 1.7, 'totally': 1.6, 'really': 1.4,
            'very': 1.3, 'quite': 1.2, 'rather': 1.1,

            # Diminishers with intensity scores
            'slightly': 0.7, 'somewhat': 0.8, 'a bit': 0.75,
            'kind of': 0.8, 'sort of': 0.8, 'fairly': 0.85,
            'barely': 0.3, 'hardly': 0.4, 'scarcely': 0.35,

            # Negations
            'not': -1.0, 'never': -1.0, 'no': -0.8,
            'without': -0.7, 'lack': -0.6
        }

        # Contextual intensity patterns
        self.lexicons['contextual_patterns'] = {
            'exclamation_marks': r'!+',
            'all_caps': r'\b[A-Z]{3,}\b',
            'repeated_letters': r'(\w)\1{2,}',
            'multiple_punctuation': r'[.!?]{2,}',
            'emphasis_words': ['really', 'truly', 'definitely', 'absolutely']
        }

        print("✓ Additional intensity lexicons loaded")

    def calculate_vader_intensity(self, text: str) -> Dict[str, float]:
        """
        Calculate emotion intensity using VADER
        VADER provides compound score that reflects intensity
        """
        if not self.vader_analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0, 'intensity': 0.5}

        scores = self.vader_analyzer.polarity_scores(text)

        # VADER compound score is already intensity-aware (-1 to 1)
        # Convert to 0-1 scale for consistency
        intensity = (abs(scores['compound']) + 1) / 2

        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg'],
            'intensity': intensity
        }

    def calculate_nrclex_intensity(self, text: str) -> Dict[str, float]:
        """
        Calculate emotion intensity using NRCLex
        """
        if not NRCLEX_AVAILABLE:
            return {'intensity': 0.5}

        try:
            emotion = NRCLex(text)

            # Get raw scores
            raw_scores = emotion.raw_emotion_scores
            affect_frequencies = emotion.affect_frequencies

            # Calculate intensity based on emotion presence and frequency
            total_words = len(text.split())
            if total_words == 0:
                return {'intensity': 0.5}

            # Sum of all emotional words divided by total words
            emotional_density = sum(affect_frequencies.values()) / total_words

            # Get strongest emotions
            if raw_scores:
                max_emotion_score = max(raw_scores.values())
                intensity = min(emotional_density * 2 + max_emotion_score * 0.1, 1.0)
            else:
                intensity = emotional_density * 2

            return {
                'intensity': intensity,
                'emotional_density': emotional_density,
                'raw_scores': raw_scores,
                'affect_frequencies': affect_frequencies
            }

        except Exception as e:
            print(f"Error in NRCLex analysis: {e}")
            return {'intensity': 0.5}

    def calculate_lexical_intensity_advanced(self, text: str) -> Dict[str, float]:
        """
        Advanced lexical intensity calculation using multiple lexicons
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        if not words:
            return {'intensity': 0.5, 'components': {}}

        components = {}

        # 1. NRC Affect Intensity scores
        nrc_scores = []
        if 'nrc_intensity' in self.lexicons:
            for word in words:
                if word in self.lexicons['nrc_intensity']:
                    # Get maximum intensity across all emotions for this word
                    word_intensities = [v for k, v in self.lexicons['nrc_intensity'][word].items()
                                        if k != 'valence']
                    if word_intensities:
                        nrc_scores.append(max(word_intensities))

        components['nrc_intensity'] = np.mean(nrc_scores) if nrc_scores else 0.5

        # 2. Intensity modifiers
        modifier_score = 1.0
        if 'intensity_modifiers' in self.lexicons:
            for word in words:
                if word in self.lexicons['intensity_modifiers']:
                    modifier_score *= self.lexicons['intensity_modifiers'][word]

        components['modifier_effect'] = min(max(modifier_score, 0.1), 3.0)  # Cap the effect

        # 3. Contextual patterns
        contextual_intensity = 1.0
        if 'contextual_patterns' in self.lexicons:
            patterns = self.lexicons['contextual_patterns']

            # Exclamation marks
            exclamations = len(re.findall(patterns['exclamation_marks'], text))
            contextual_intensity += exclamations * 0.2

            # All caps words
            caps_words = len(re.findall(patterns['all_caps'], text))
            contextual_intensity += caps_words * 0.3

            # Repeated letters (e.g., "soooo happy")
            repeated = len(re.findall(patterns['repeated_letters'], text))
            contextual_intensity += repeated * 0.25

            # Multiple punctuation
            multi_punct = len(re.findall(patterns['multiple_punctuation'], text))
            contextual_intensity += multi_punct * 0.15

        components['contextual_intensity'] = min(contextual_intensity, 2.0)

        # 4. Combine all components
        base_intensity = components['nrc_intensity']
        modified_intensity = base_intensity * components['modifier_effect']
        final_intensity = modified_intensity * components['contextual_intensity']

        # Normalize to 0-1 range
        final_intensity = min(max(final_intensity, 0.0), 1.0)

        return {
            'intensity': final_intensity,
            'components': components
        }

    def calculate_transformer_intensity(self, text: str) -> Dict[str, float]:
        """
        Calculate intensity using transformer-based models
        """
        results = {}

        # RoBERTa emotion model
        if 'roberta' in self.emotion_models:
            try:
                emotion_scores = self.emotion_models['roberta'](text)

                # Calculate intensity as max emotion confidence
                max_confidence = max([score['score'] for score in emotion_scores])

                # Also calculate emotional diversity (lower = more focused emotion)
                emotion_entropy = -sum([score['score'] * np.log(score['score'] + 1e-10)
                                        for score in emotion_scores])

                # Normalize entropy (higher entropy = lower intensity focus)
                max_entropy = -np.log(1/len(emotion_scores))
                normalized_entropy = emotion_entropy / max_entropy

                # Intensity combines max confidence and focus (inverse entropy)
                intensity = (max_confidence + (1 - normalized_entropy)) / 2

                results['roberta_intensity'] = intensity
                results['roberta_max_confidence'] = max_confidence
                results['roberta_emotion_focus'] = 1 - normalized_entropy

            except Exception as e:
                print(f"Error in RoBERTa analysis: {e}")
                results['roberta_intensity'] = 0.5

        # BERT sentiment model
        if 'bert' in self.emotion_models:
            try:
                sentiment_scores = self.emotion_models['bert'](text)

                # Calculate intensity as deviation from neutral
                max_score = max([score['score'] for score in sentiment_scores])

                # Find neutral score (if available)
                neutral_score = 0.33  # Default
                for score in sentiment_scores:
                    if 'neutral' in score['label'].lower() or 'label_1' in score['label']:
                        neutral_score = score['score']
                        break

                # Intensity is deviation from neutrality
                intensity = 1 - neutral_score

                results['bert_intensity'] = intensity
                results['bert_max_confidence'] = max_score

            except Exception as e:
                print(f"Error in BERT analysis: {e}")
                results['bert_intensity'] = 0.5

        return results

    def calculate_comprehensive_intensity(self, text: str) -> Dict[str, float]:
        """
        Comprehensive intensity calculation combining all methods
        """
        results = {}

        # 1. VADER intensity
        vader_result = self.calculate_vader_intensity(text)
        results.update({f'vader_{k}': v for k, v in vader_result.items()})

        # 2. NRCLex intensity
        nrclex_result = self.calculate_nrclex_intensity(text)
        results.update({f'nrclex_{k}': v for k, v in nrclex_result.items()
                        if k not in ['raw_scores', 'affect_frequencies']})

        # 3. Advanced lexical intensity
        lexical_result = self.calculate_lexical_intensity_advanced(text)
        results.update({f'lexical_{k}': v for k, v in lexical_result.items()
                        if k != 'components'})

        # 4. Transformer-based intensity
        transformer_result = self.calculate_transformer_intensity(text)
        results.update(transformer_result)

        # 5. Calculate ensemble intensity
        intensity_scores = []

        if 'vader_intensity' in results:
            intensity_scores.append(results['vader_intensity'])
        if 'nrclex_intensity' in results:
            intensity_scores.append(results['nrclex_intensity'])
        if 'lexical_intensity' in results:
            intensity_scores.append(results['lexical_intensity'])
        if 'roberta_intensity' in results:
            intensity_scores.append(results['roberta_intensity'])
        if 'bert_intensity' in results:
            intensity_scores.append(results['bert_intensity'])

        if intensity_scores:
            # Ensemble using weighted average (you can adjust weights)
            weights = [0.25, 0.2, 0.2, 0.2, 0.15][:len(intensity_scores)]
            weights = np.array(weights) / np.sum(weights)  # Normalize

            results['ensemble_intensity'] = np.average(intensity_scores, weights=weights)
            results['intensity_std'] = np.std(intensity_scores)
            results['intensity_agreement'] = 1 - (np.std(intensity_scores) / np.mean(intensity_scores) if np.mean(intensity_scores) > 0 else 0)
        else:
            results['ensemble_intensity'] = 0.5
            results['intensity_std'] = 0.0
            results['intensity_agreement'] = 0.0

        return results

    def batch_intensity_analysis(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze intensity for a batch of texts
        """
        results = []

        for i, text in enumerate(texts):
            print(f"Analyzing text {i+1}/{len(texts)}", end='\r')

            text_results = self.calculate_comprehensive_intensity(text)
            text_results['text'] = text[:100] + '...' if len(text) > 100 else text
            text_results['text_id'] = i

            results.append(text_results)

        print()  # New line after progress
        return pd.DataFrame(results)

    def compare_intensity_methods(self, text: str) -> None:
        """
        Compare different intensity calculation methods for a single text
        """
        print(f"Text: '{text[:100]}...' " if len(text) > 100 else f"Text: '{text}'")
        print("-" * 60)

        results = self.calculate_comprehensive_intensity(text)

        print("Intensity Scores:")
        methods = ['vader', 'nrclex', 'lexical', 'roberta', 'bert', 'ensemble']

        for method in methods:
            key = f'{method}_intensity'
            if key in results:
                score = results[key]
                print(f"  {method.upper():>10}: {score:.3f}")

        if 'intensity_std' in results:
            print(f"  {'STD':>10}: {results['intensity_std']:.3f}")
            print(f"  {'AGREEMENT':>10}: {results['intensity_agreement']:.3f}")

class EnhancedEmotionIntensityEvaluator:
    """
    Comprehensive evaluator for emotion intensity consistency in text augmentation
    """

    def __init__(self):
        """Initialize with advanced analyzers"""
        self.advanced_analyzer = AdvancedEmotionIntensityAnalyzer()
        self.sentiment_analyzer = None
        self.emotion_analyzer = None

        # Initialize transformer models (keeping your existing approach)
        self._initialize_sentiment_analyzer()
        self._initialize_emotion_analyzer()

    def _initialize_sentiment_analyzer(self):
        """Your existing sentiment analyzer initialization"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            print("Sentiment analyzer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize sentiment analyzer: {e}")
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                print("Fallback sentiment analyzer initialized")
            except Exception as e2:
                print(f"Could not initialize any sentiment analyzer: {e2}")

    def _initialize_emotion_analyzer(self):
        """Your existing emotion analyzer initialization"""
        try:
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            print("Emotion analyzer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize emotion analyzer: {e}")

    def calculate_lexical_intensity(self, text: str) -> float:
        """
        Enhanced lexical intensity using advanced methods instead of word lists
        """
        # Use the advanced analyzer instead of primitive word lists
        result = self.advanced_analyzer.calculate_comprehensive_intensity(text)
        return result.get('ensemble_intensity', 0.5)

    def calculate_advanced_sentdiff(self, original_texts: List[str],
                                    augmented_texts: List[str]) -> Dict[str, float]:
        """
        Enhanced SENTDIFF using multiple intensity measures
        """
        if len(original_texts) != len(augmented_texts):
            raise ValueError("Original and augmented texts must have the same length")

        results = {}

        # Calculate comprehensive intensity for all texts
        print("Calculating original text intensities...")
        orig_intensities = []
        for text in original_texts:
            intensity_data = self.advanced_analyzer.calculate_comprehensive_intensity(text)
            orig_intensities.append(intensity_data)

        print("Calculating augmented text intensities...")
        aug_intensities = []
        for text in augmented_texts:
            intensity_data = self.advanced_analyzer.calculate_comprehensive_intensity(text)
            aug_intensities.append(intensity_data)

        # Calculate differences for each method
        methods = ['vader_intensity', 'nrclex_intensity', 'lexical_intensity',
                   'roberta_intensity', 'bert_intensity', 'ensemble_intensity']

        for method in methods:
            diffs = []
            for orig, aug in zip(orig_intensities, aug_intensities):
                orig_score = orig.get(method, 0.5)
                aug_score = aug.get(method, 0.5)
                diffs.append(abs(orig_score - aug_score))

            if diffs:
                results[f'sentdiff_{method}'] = np.mean(diffs)
                results[f'sentdiff_{method}_std'] = np.std(diffs)

        # Overall SENTDIFF (ensemble)
        if 'sentdiff_ensemble_intensity' in results:
            results['sentdiff_overall'] = results['sentdiff_ensemble_intensity']

        return results

    def calculate_advanced_sentstd(self, augmented_texts: List[str],
                                   by_group: Optional[List] = None) -> Dict[str, float]:
        """
        Enhanced SENTSTD using multiple intensity measures
        """
        print("Calculating intensity scores for SENTSTD...")

        # Get intensity scores for all texts
        intensity_data = []
        for text in augmented_texts:
            scores = self.advanced_analyzer.calculate_comprehensive_intensity(text)
            intensity_data.append(scores)

        results = {}
        methods = ['vader_intensity', 'nrclex_intensity', 'lexical_intensity',
                   'roberta_intensity', 'bert_intensity', 'ensemble_intensity']

        if by_group is None:
            # Calculate overall standard deviation
            for method in methods:
                scores = [data.get(method, 0.5) for data in intensity_data]
                if scores:
                    results[f'sentstd_{method}'] = np.std(scores)

            # Overall SENTSTD (ensemble)
            if f'sentstd_ensemble_intensity' in results:
                results['sentstd_overall'] = results['sentstd_ensemble_intensity']

        else:
            # Calculate standard deviation by group
            unique_groups = list(set(by_group))

            for method in methods:
                group_stds = []
                for group in unique_groups:
                    group_indices = [i for i, g in enumerate(by_group) if g == group]
                    group_scores = [intensity_data[i].get(method, 0.5) for i in group_indices]

                    if len(group_scores) > 1:
                        group_stds.append(np.std(group_scores))

                if group_stds:
                    results[f'sentstd_{method}'] = np.mean(group_stds)

            # Overall SENTSTD (ensemble)
            if f'sentstd_ensemble_intensity' in results:
                results['sentstd_overall'] = results['sentstd_ensemble_intensity']

        return results

    def calculate_emotion_intensity_consistency_advanced(self,
                                                         original_texts: List[str],
                                                         augmented_texts: List[str],
                                                         original_labels: List[str]) -> Dict[str, float]:
        """
        Enhanced emotion intensity consistency using advanced methods
        """
        results = {}

        # 1. Advanced lexical intensity consistency
        print("Calculating advanced intensity consistency...")
        orig_intensities = []
        aug_intensities = []

        for orig_text, aug_text in zip(original_texts, augmented_texts):
            orig_data = self.advanced_analyzer.calculate_comprehensive_intensity(orig_text)
            aug_data = self.advanced_analyzer.calculate_comprehensive_intensity(aug_text)

            orig_intensities.append(orig_data)
            aug_intensities.append(aug_data)

        # Calculate consistency for each method
        methods = ['vader_intensity', 'nrclex_intensity', 'lexical_intensity',
                   'roberta_intensity', 'bert_intensity', 'ensemble_intensity']

        for method in methods:
            orig_scores = [data.get(method, 0.5) for data in orig_intensities]
            aug_scores = [data.get(method, 0.5) for data in aug_intensities]

            # Intensity differences
            diffs = [abs(o - a) for o, a in zip(orig_scores, aug_scores)]
            results[f'{method}_diff_mean'] = np.mean(diffs)
            results[f'{method}_diff_std'] = np.std(diffs)

            # Correlations
            if len(orig_scores) > 1:
                from scipy.stats import pearsonr, spearmanr
                try:
                    pearson_corr, _ = pearsonr(orig_scores, aug_scores)
                    spearman_corr, _ = spearmanr(orig_scores, aug_scores)
                    results[f'{method}_pearson_corr'] = pearson_corr
                    results[f'{method}_spearman_corr'] = spearman_corr
                except:
                    results[f'{method}_pearson_corr'] = 0.0
                    results[f'{method}_spearman_corr'] = 0.0

        # 2. Label-specific intensity analysis
        low_indices = [i for i, label in enumerate(original_labels) if label == 'low']
        high_indices = [i for i, label in enumerate(original_labels) if label == 'high']

        for method in ['ensemble_intensity']:  # Focus on best method for label analysis
            if low_indices:
                low_orig = [orig_intensities[i].get(method, 0.5) for i in low_indices]
                low_aug = [aug_intensities[i].get(method, 0.5) for i in low_indices]
                low_consistency = 1 - np.mean([abs(o - a) for o, a in zip(low_orig, low_aug)])
                results[f'low_{method}_consistency'] = max(0, low_consistency)

            if high_indices:
                high_orig = [orig_intensities[i].get(method, 0.5) for i in high_indices]
                high_aug = [aug_intensities[i].get(method, 0.5) for i in high_indices]
                high_consistency = 1 - np.mean([abs(o - a) for o, a in zip(high_orig, high_aug)])
                results[f'high_{method}_consistency'] = max(0, high_consistency)
    def batch_intensity_analysis(self, texts: List[str], save_path: Optional[str] = None) -> pd.DataFrame:
        results = []
        for i, text in enumerate(texts):
            print(f"Analyzing text {i+1}/{len(texts)}", end='\r')
            text_results = self.calculate_comprehensive_intensity(text)
            text_results['text'] = text
            text_results['text_id'] = i
            results.append(text_results)
        print()

        df = pd.DataFrame(results)
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Wyniki zapisane do: {save_path}")
        return df

def main():
    """Główna funkcja analizy"""

    # Ścieżki do plików
    ESCONV_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/esconv_both_parts.csv'
    MEISD_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/filtered_negative_MEISD_intensity_max_first_25_conv.csv'

    # Ścieżki do zaugmentowanych datassetów
    AUGMENTED_FILES = {
        'enhanced_classical_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/esconv_enhanced_classical_augmentation_70percent_balanced.xlsx',
        'enhanced_mixed_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/esconv_enhanced_mixed_augmentation_70percent_balanced.xlsx',
        'enhanced_llm_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/esconv_enhanced_llm_augmentation_70percent_balanced.xlsx',
        'enhanced_nlp_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/esconv_enhanced_nlp_augmentation_70percent_balanced.xlsx',
        'enhanced_nlp_llm_balanced': 'C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/esconv_enhanced_llm_nlp_augmentation_70percent_balanced.xlsx'
    }

    # Inicjalizacja analizatora
    analyzer = AugmentationQualityAnalyzer(ESCONV_PATH, MEISD_PATH)

    # Setup
    original_data = analyzer.setup()

    # Wczytanie zaugmentowanych datassetów
    analyzer.load_augmented_datasets(AUGMENTED_FILES)

    # Przeprowadzenie analiz
    print("\n" + "="*60)
    print("ROZPOCZĘCIE KOMPLEKSOWEJ ANALIZY JAKOŚCI AUGMENTACJI")
    print("="*60)

    # Analiza podstawowych statystyk
    analyzer.analyze_basic_statistics()

    # Analiza jakości transformacji
    analyzer.analyze_transformation_quality()

    # Analiza konsystencji stylu
    analyzer.analyze_style_consistency()

    # Analiza zgodna z artykulem
    analyzer.analyze_nlp_metrics()

    # NOWOŚĆ: analiza zgodności intensywności emocji
    analyzer.analyze_emotion_intensity_consistency()

    # Porównanie datassetów
    analyzer.compare_datasets()

    # Generowanie wizualizacji
    analyzer.generate_visualizations()

    # Zapisanie raportu
    analyzer.save_analysis_report()

    print("\n" + "="*60)
    print("ANALIZA ZAKOŃCZONA POMYŚLNIE")
    print("="*60)

    print("\nPliki wyjściowe:")
    print("- analysis_report.txt - szczegółowy raport tekstowy")
    print("- analysis_results/ - katalog z wizualizacjami")
    print("  - text_length_distribution.png")
    print("  - class_distribution.png")
    print("  - transformation_quality_comparison.png")


if __name__ == "__main__":
    main()


'''
=== Inicjalizacja analizatora jakości augmentacji ===
Loading ESConv data...
ESConv data loaded: 2450 rows
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
Setting up enhanced MEISD data augmenter...
MEISD data loaded: 1085 rows
Using existing intensity column: label
Using 2450 samples with valid intensity labels
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
ESConv style patterns analyzed:
  low: 10 sentence starters, avg length: 108.8 words
  high: 10 sentence starters, avg length: 123.2 words

=== Wczytywanie zaugmentowanych datassetów ===
Wczytano enhanced_classical_balanced: 4738 próbek
Wczytano enhanced_mixed_balanced: 4738 próbek
Wczytano enhanced_llm_balanced: 4738 próbek
Wczytano enhanced_nlp_balanced: 4738 próbek
Wczytano enhanced_nlp_llm_balanced: 4738 próbek

============================================================
ROZPOCZĘCIE KOMPLEKSOWEJ ANALIZY JAKOŚCI AUGMENTACJI
============================================================

=== Analiza podstawowych statystyk ===

--- Analiza enhanced_classical_balanced ---
  Liczba próbek: 4738
  Rozkład klas: {0: 2369, 1: 2369}
  Średnia długość tekstu: 68.5 ± 67.5 słów
  Unikalność tekstów: 0.896
  Balans klas: 1.000

--- Analiza enhanced_mixed_balanced ---
  Liczba próbek: 4738
  Rozkład klas: {0: 2369, 1: 2369}
  Średnia długość tekstu: 82.0 ± 60.7 słów
  Unikalność tekstów: 0.988
  Balans klas: 1.000

--- Analiza enhanced_llm_balanced ---
  Liczba próbek: 4738
  Rozkład klas: {0: 2369, 1: 2369}
  Średnia długość tekstu: 87.9 ± 56.6 słów
  Unikalność tekstów: 0.999
  Balans klas: 1.000

--- Analiza enhanced_nlp_balanced ---
  Liczba próbek: 4738
  Rozkład klas: {1: 2369, 0: 2369}
  Średnia długość tekstu: 67.4 ± 68.2 słów
  Unikalność tekstów: 0.975
  Balans klas: 1.000

--- Analiza enhanced_nlp_llm_balanced ---
  Liczba próbek: 4738
  Rozkład klas: {1: 2369, 0: 2369}
  Średnia długość tekstu: 87.7 ± 57.9 słów
  Unikalność tekstów: 0.991
  Balans klas: 1.000

=== Analiza jakości transformacji ===

--- Analiza jakości transformacji enhanced_classical_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
  Znaleziono 2288 zaugmentowanych próbek
Analizowanie jakości enhanced_classical_balanced: 100%|██████████| 2288/2288 [00:03<00:00, 616.35it/s]
  Średnia jakość transformacji: 0.398 ± 0.157
  Zakres jakości: 0.103 - 0.801

--- Analiza jakości transformacji enhanced_mixed_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
  Znaleziono 2287 zaugmentowanych próbek
Analizowanie jakości enhanced_mixed_balanced: 100%|██████████| 2287/2287 [00:03<00:00, 681.89it/s]
  Średnia jakość transformacji: 0.623 ± 0.215
  Zakres jakości: 0.104 - 0.999

--- Analiza jakości transformacji enhanced_llm_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
Analizowanie jakości enhanced_llm_balanced:   0%|          | 0/2286 [00:00<?, ?it/s]  Znaleziono 2286 zaugmentowanych próbek
Analizowanie jakości enhanced_llm_balanced: 100%|██████████| 2286/2286 [00:03<00:00, 689.57it/s]
  Średnia jakość transformacji: 0.719 ± 0.161
  Zakres jakości: 0.107 - 0.999

--- Analiza jakości transformacji enhanced_nlp_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
Analizowanie jakości enhanced_nlp_balanced:   0%|          | 0/2288 [00:00<?, ?it/s]  Znaleziono 2288 zaugmentowanych próbek
Analizowanie jakości enhanced_nlp_balanced: 100%|██████████| 2288/2288 [00:03<00:00, 720.43it/s]
  Średnia jakość transformacji: 0.328 ± 0.086
  Zakres jakości: 0.108 - 0.600

--- Analiza jakości transformacji enhanced_nlp_llm_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
  Znaleziono 2288 zaugmentowanych próbek
Analizowanie jakości enhanced_nlp_llm_balanced: 100%|██████████| 2288/2288 [00:03<00:00, 698.84it/s]
  Średnia jakość transformacji: 0.446 ± 0.124
  Zakres jakości: 0.180 - 0.600

=== Analiza konsystencji stylu z ESConv ===

--- Analiza stylu enhanced_classical_balanced ---
  Konsystencja stylu 'low': 0.542
  Konsystencja stylu 'high': 0.561

--- Analiza stylu enhanced_mixed_balanced ---
  Konsystencja stylu 'low': 0.597
  Konsystencja stylu 'high': 0.598

--- Analiza stylu enhanced_llm_balanced ---
  Konsystencja stylu 'low': 0.620
  Konsystencja stylu 'high': 0.613

--- Analiza stylu enhanced_nlp_balanced ---
  Konsystencja stylu 'low': 0.669
  Konsystencja stylu 'high': 0.721

--- Analiza stylu enhanced_nlp_llm_balanced ---
  Konsystencja stylu 'low': 0.669
  Konsystencja stylu 'high': 0.721

=== Analiza metryk NLP (BLEU, CHRF, Novelty, Perplexity, itd.) ===
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

--- NLP ocena: enhanced_classical_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
============================================================
TEXT AUGMENTATION EVALUATION REPORT
============================================================

1. TRANSLATION QUALITY
   BLEU Score: 0.9798

2. TEXT GENERATION QUALITY
   Novelty Score: 0.0000
   Diversity Metrics:
     Self-BLEU: 0.8671
     UTR (Unique Trigrams Ratio): 0.3619
     TTR (Type-Token Ratio): 0.0591
     RWORDS (Rare Words): 0.9938
   Perplexity: 0.0000
   Spelling Metrics:
     Misspelled Words Ratio: 0.0026
     Misspelled Chars Ratio: 0.0016

3. CHARACTER N-GRAM MATCHES
   CHRF Score: 1.0000
============================================================

--- NLP ocena: enhanced_mixed_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
============================================================
TEXT AUGMENTATION EVALUATION REPORT
============================================================

1. TRANSLATION QUALITY
   BLEU Score: 0.9936

2. TEXT GENERATION QUALITY
   Novelty Score: 0.0000
   Diversity Metrics:
     Self-BLEU: 0.7484
     UTR (Unique Trigrams Ratio): 0.3686
     TTR (Type-Token Ratio): 0.0322
     RWORDS (Rare Words): 0.9948
   Perplexity: 0.0000
   Spelling Metrics:
     Misspelled Words Ratio: 0.0030
     Misspelled Chars Ratio: 0.0022

3. CHARACTER N-GRAM MATCHES
   CHRF Score: 1.0000
============================================================

--- NLP ocena: enhanced_llm_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
============================================================
TEXT AUGMENTATION EVALUATION REPORT
============================================================

1. TRANSLATION QUALITY
   BLEU Score: 0.9989

2. TEXT GENERATION QUALITY
   Novelty Score: 0.0000
   Diversity Metrics:
     Self-BLEU: 0.7989
     UTR (Unique Trigrams Ratio): 0.3206
     TTR (Type-Token Ratio): 0.0251
     RWORDS (Rare Words): 0.9950
   Perplexity: 0.0000
   Spelling Metrics:
     Misspelled Words Ratio: 0.0029
     Misspelled Chars Ratio: 0.0021

3. CHARACTER N-GRAM MATCHES
   CHRF Score: 1.0000
============================================================

--- NLP ocena: enhanced_nlp_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
============================================================
TEXT AUGMENTATION EVALUATION REPORT
============================================================

1. TRANSLATION QUALITY
   BLEU Score: 0.9687

2. TEXT GENERATION QUALITY
   Novelty Score: 0.0000
   Diversity Metrics:
     Self-BLEU: 0.6381
     UTR (Unique Trigrams Ratio): 0.5667
     TTR (Type-Token Ratio): 0.0748
     RWORDS (Rare Words): 0.9950
   Perplexity: 0.0000
   Spelling Metrics:
     Misspelled Words Ratio: 0.0028
     Misspelled Chars Ratio: 0.0017

3. CHARACTER N-GRAM MATCHES
   CHRF Score: 1.0000
============================================================

--- NLP ocena: enhanced_nlp_llm_balanced ---
Classification dataset prepared: 2450 samples
Original intensity distribution: {1: 331, 2: 506, 3: 557, 4: 629, 5: 427}
Binary intensity distribution: {'low': 1394, 'high': 1056}
Binary label distribution: {0: 1394, 1: 1056}
============================================================
TEXT AUGMENTATION EVALUATION REPORT
============================================================

1. TRANSLATION QUALITY
   BLEU Score: 0.9994

2. TEXT GENERATION QUALITY
   Novelty Score: 0.0000
   Diversity Metrics:
     Self-BLEU: 0.8314
     UTR (Unique Trigrams Ratio): 0.2642
     TTR (Type-Token Ratio): 0.0231
     RWORDS (Rare Words): 0.9933
   Perplexity: 0.0000
   Spelling Metrics:
     Misspelled Words Ratio: 0.0017
     Misspelled Chars Ratio: 0.0006

3. CHARACTER N-GRAM MATCHES
   CHRF Score: 1.0000
============================================================

=== Porównanie różnych metod augmentacji ===

--- Tabela porównawcza ---
                     method  total_samples  balance_ratio  uniqueness_ratio  avg_text_length  avg_quality  style_consistency_low  style_consistency_high  bleu_score  chrf_score  novelty_score  perplexity
enhanced_classical_balanced           4738            1.0          0.895526        68.528282     0.398434               0.541600                0.561483    0.979761         1.0            0.0         0.0
    enhanced_mixed_balanced           4738            1.0          0.987547        81.992824     0.623186               0.597172                0.597619    0.993621         1.0            0.0         0.0
      enhanced_llm_balanced           4738            1.0          0.998523        87.936471     0.719234               0.620334                0.613080    0.998851         1.0            0.0         0.0
      enhanced_nlp_balanced           4738            1.0          0.974673        67.439637     0.328056               0.669305                0.721442    0.968678         1.0            0.0         0.0
  enhanced_nlp_llm_balanced           4738            1.0          0.990713        87.677712     0.445937               0.669305                0.721442    0.999402         1.0            0.0         0.0

--- Ranking metod augmentacji ---
1. enhanced_llm_balanced: 50000.717
2. enhanced_mixed_balanced: 50000.692
3. enhanced_nlp_llm_balanced: 50000.677
4. enhanced_nlp_balanced: 50000.647
5. enhanced_classical_balanced: 50000.627

=== Generowanie wizualizacji ===
Wizualizacje zapisane w katalogu: analysis_results

=== Zapisywanie raportu analizy ===
Raport zapisany w pliku: analysis_report.txt

============================================================
ANALIZA ZAKOŃCZONA POMYŚLNIE
============================================================

Pliki wyjściowe:
- analysis_report.txt - szczegółowy raport tekstowy
- analysis_results/ - katalog z wizualizacjami
  - text_length_distribution.png
  - class_distribution.png
  - transformation_quality_comparison.png
'''