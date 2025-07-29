'''Kluczowe ulepszenia:

Używa istniejących labelek intensity z ESConv - nie tworzy sztucznych na podstawie zawartości tekstu
Lepsze mapowanie intensity:

ESConv (1-5) → Binary: 1-2 = 'low', 3-5 = 'high'
MEISD (1-3) → Binary: ≤1.5 = 'low', >1.5 = 'high'


Analiza wzorców stylistycznych ESConv:

Ekstraktuje rzeczywiste wzorce językowe dla każdego poziomu intensywności
Analizuje długość zdań, słowa kluczowe, zaimki osobowe
Używa TF-IDF do identyfikacji charakterystycznych słów


Ulepszona transformacja LLM:

Używa prawdziwych przykładów z ESConv jako wzorców
Lepsze promptowanie z kontekstem stylistycznym
Instrukcje specyficzne dla poziomu intensywności


Ulepszona transformacja klasyczna:

Adaptuje tekst do wzorców ESConv (długość, styl)
Inteligentne wzmacnianie/łagodzenie języka
Dodawanie elementów personalnych charakterystycznych dla ESConv


System oceny jakości transformacji:

Sprawdza czy transformacja jest sensowna
Ocenia zgodność z wzorcami ESConv
Pomaga w monitorowaniu jakości augmentacji



Jak używać:

Ustaw ścieżki do swoich plików (ESConv z istniejącymi labelkami intensity, MEISD, opcjonalnie LLaMA)
Uruchom kod - automatycznie:

Wczyta ESConv z istniejącymi labelkami
Przeanalizuje wzorce stylistyczne dla każdego poziomu intensywności
Przetransformuje MEISD do stylu ESConv z zachowaniem odpowiedniego poziomu intensywności


Otrzymasz 3 wersje augmentowanych danych: enhanced_mixed (zalecana), enhanced_llm, enhanced_classical, enhanced_nlp, enhanced_llm_nlp'''

import os
import time
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from llama_cpp import Llama
from deep_translator import GoogleTranslator
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

class EnhancedESConvProcessor:
    """
    Enhanced ESConv processor with better intensity mapping and style analysis
    """

    def __init__(self, esconv_path):
        self.esconv_path = esconv_path
        self.esconv_data = None
        self.style_patterns = None
        self.intensity_keywords = None

    def load_data(self):
        """Load ESConv dataset"""
        print("Loading ESConv data...")
        self.esconv_data = pd.read_csv(self.esconv_path)
        print(f"ESConv data loaded: {len(self.esconv_data)} rows")
        return self.esconv_data

    def analyze_esconv_style_patterns(self):
        """Analyze ESConv linguistic patterns for better transformation"""
        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        # Find text column
        text_cols = [col for col in self.esconv_data.columns
                     if any(keyword in col.lower() for keyword in ['content', 'utterance', 'text', 'message'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        # Find existing intensity column
        intensity_col = [col for col in self.esconv_data.columns if 'label' in col.lower()]
        if not intensity_col:
            raise ValueError("No intensity column found in ESConv data. Please ensure the dataset has intensity labels.")

        intensity_col = intensity_col[0]
        print(f"Using existing intensity column: {intensity_col}")

        # Analyze patterns by intensity level
        self.style_patterns = {}
        self.intensity_keywords = {}

        # Create intensity mapping from original ESConv labels (1-5) to binary
        def map_to_binary_intensity(intensity):
            if pd.isna(intensity):
                return None  # Skip NaN values
            if intensity <= 3:
                return 'low'  # 1-2 -> low intensity
            else:
                return 'high'  # 3-5 -> high intensity

        # Apply intensity mapping using existing labels
        self.esconv_data['binary_intensity'] = self.esconv_data[intensity_col].apply(map_to_binary_intensity)

        # Remove rows with NaN intensity
        valid_data = self.esconv_data.dropna(subset=['binary_intensity'])
        print(f"Using {len(valid_data)} samples with valid intensity labels")
        print(f"Original intensity distribution: {self.esconv_data[intensity_col].value_counts().sort_index().to_dict()}")
        print(f"Binary intensity distribution: {valid_data['binary_intensity'].value_counts().to_dict()}")

        # Analyze patterns for each intensity level
        for intensity in ['low', 'high']:
            subset = valid_data[valid_data['binary_intensity'] == intensity]
            texts = subset[text_col].dropna().astype(str)

            if len(texts) == 0:
                print(f"Warning: No texts found for {intensity} intensity")
                continue

            # Extract common patterns
            self.style_patterns[intensity] = self._extract_style_patterns(texts)
            self.intensity_keywords[intensity] = self._extract_intensity_keywords(texts)

        print("ESConv style patterns analyzed:")
        for intensity, patterns in self.style_patterns.items():
            print(f"  {intensity}: {len(patterns.get('sentence_starters', []))} sentence starters, avg length: {patterns.get('avg_length', 0):.1f} words")

    def _extract_style_patterns(self, texts):
        """Extract common linguistic patterns"""
        patterns = {
            'sentence_starters': [],
            'common_phrases': [],
            'avg_length': 0,
            'question_ratio': 0,
            'exclamation_ratio': 0,
            'personal_pronouns': []
        }

        all_sentences = []
        for text in texts:
            sentences = re.split(r'[.!?]+', str(text))
            all_sentences.extend([s.strip() for s in sentences if s.strip()])

        # Sentence starters
        starters = [s.split()[0].lower() for s in all_sentences if len(s.split()) > 0]
        starter_counts = pd.Series(starters).value_counts()
        patterns['sentence_starters'] = starter_counts.head(10).index.tolist()

        # Average length
        patterns['avg_length'] = np.mean([len(text.split()) for text in texts])

        # Question and exclamation ratios
        patterns['question_ratio'] = sum(1 for text in texts if '?' in text) / len(texts)
        patterns['exclamation_ratio'] = sum(1 for text in texts if '!' in text) / len(texts)

        # Personal pronouns
        pronouns = ['i', 'me', 'my', 'myself', 'we', 'us', 'our']
        pronoun_counts = {}
        for text in texts:
            words = text.lower().split()
            for pronoun in pronouns:
                pronoun_counts[pronoun] = pronoun_counts.get(pronoun, 0) + words.count(pronoun)
        patterns['personal_pronouns'] = sorted(pronoun_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return patterns

    def _extract_intensity_keywords(self, texts):
        """Extract keywords specific to intensity level"""
        # Simple TF-IDF approach for keyword extraction
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Get top keywords
            top_indices = np.argsort(mean_scores)[-20:]
            keywords = [feature_names[i] for i in top_indices]
            return keywords
        except:
            return []
    def prepare_for_classification(self):
        """Prepare ESConv data for binary classification using existing intensity labels"""
        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        text_cols = [col for col in self.esconv_data.columns
                     if any(keyword in col.lower() for keyword in ['content', 'utterance', 'text', 'message'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        # Find existing intensity column
        intensity_col = [col for col in self.esconv_data.columns if 'label' in col.lower()]
        if not intensity_col:
            raise ValueError("No intensity column found in ESConv data.")

        intensity_col = intensity_col[0]

        # Create classification dataset using existing labels
        classification_data = self.esconv_data[[text_col, intensity_col]].copy()
        classification_data.columns = ['Utterances', 'original_intensity']

        # Map to binary intensity (keeping original for reference)
        classification_data['intensity'] = classification_data['original_intensity'].apply(
            lambda x: 'low' if pd.notna(x) and x <= 3 else ('high' if pd.notna(x) else None)
        )

        # Create binary labels
        classification_data['label'] = classification_data['intensity'].map({'low': 0, 'high': 1})

        # Remove any NaN values
        classification_data = classification_data.dropna(subset=['label'])

        print(f"Classification dataset prepared: {len(classification_data)} samples")
        print("Original intensity distribution:", classification_data['original_intensity'].value_counts().sort_index().to_dict())
        print("Binary intensity distribution:", classification_data['intensity'].value_counts().to_dict())
        print("Binary label distribution:", classification_data['label'].value_counts().to_dict())

        return classification_data[['Utterances', 'intensity', 'label']]


class EnhancedMEISDDataAugmenter:
    """
    Enhanced MEISD data augmenter with better style transformation
    """

    def __init__(self, meisd_path, esconv_processor, llama_path=None):
        self.meisd_path = meisd_path
        self.esconv_processor = esconv_processor
        self.llama_path = llama_path
        self.meisd_data = None
        self.llm = None
        self.style_transformer = None

    def setup(self):
        """Setup enhanced augmenter"""
        print("Setting up enhanced MEISD data augmenter...")

        # Load MEISD data
        self.meisd_data = pd.read_csv(self.meisd_path)
        print(f"MEISD data loaded: {len(self.meisd_data)} rows")

        # Map MEISD intensity (1-3) to binary
        self.meisd_data['binary_intensity'] = self.meisd_data['max_intensity'].apply(
            lambda x: 'low' if x <= 1.5 else 'high'
        )
        self.meisd_data['label'] = self.meisd_data['binary_intensity'].map({'low': 0, 'high': 1})

        # Analyze ESConv style patterns
        self.esconv_processor.analyze_esconv_style_patterns()

        # Initialize LLaMA if available
        if self.llama_path:
            try:
                print("Loading LLaMA model...")
                self.llm = Llama(model_path=self.llama_path, n_ctx=2048, n_threads=6, verbose=False)
                print("LLaMA model loaded successfully")
            except Exception as e:
                print(f"Failed to load LLaMA model: {e}")
                self.llm = None

    def _synonym_replacement(self, text, target_intensity=None):
        """Replace words with synonyms"""
        words = text.split()
        new_words = words[:]
        num_replacements = max(1, len(words) // 5)

        try:
            random_words = random.sample(words, min(num_replacements, len(words)))
            for word in random_words:
                # Clean word from punctuation
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word:
                    synonyms = wordnet.synsets(clean_word)
                    if synonyms:
                        synonym = random.choice(synonyms).lemmas()[0].name()
                        synonym = synonym.replace('_', ' ')
                        if synonym != clean_word:
                            # Replace maintaining original case
                            new_words = [synonym if w.lower() == word.lower() else w for w in new_words]
        except Exception as e:
            print(f"Synonym replacement error: {e}")

        return ' '.join(new_words)

    def _random_insertion(self, text, target_intensity=None, n=1):
        """Insert random words from the text"""
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            new_word = random.choice(words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, new_word)

        return ' '.join(words)

    def _random_deletion(self, text, target_intensity=None, p=0.3):
        """Randomly delete words from text"""
        words = text.split()
        if len(words) == 1:
            return text

        new_words = [word for word in words if random.uniform(0, 1) > p]
        return ' '.join(new_words) if new_words else random.choice(words)

    def _back_translation(self, text, target_intensity=None, src_lang='en', mid_lang='fr', max_retries=3):
        """Perform back translation for text augmentation"""
        attempt = 0
        while attempt < max_retries:
            try:
                # Translate to intermediate language
                translated = GoogleTranslator(source=src_lang, target=mid_lang).translate(text)
                if translated:
                    # Translate back to original language
                    back_translated = GoogleTranslator(source=mid_lang, target=src_lang).translate(translated)
                    if back_translated:
                        return back_translated
            except Exception as e:
                print(f"Back translation error on attempt {attempt + 1}: {e}")
                attempt += 1
                time.sleep(1)

        return text

    def _enhanced_nlp_transformation(self, text, target_intensity):
        """Apply random NLP augmentation techniques"""
        techniques = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_deletion,
            self._back_translation
        ]

        # Choose random technique
        technique = random.choice(techniques)

        try:
            transformed = technique(text, target_intensity)
            # Apply ESConv style adaptation
            transformed = self._adjust_to_esconv_style(transformed,
                                                       self.esconv_processor.style_patterns.get(target_intensity, {}))
            return transformed
        except Exception as e:
            print(f"NLP transformation error: {e}")
            return text

    def _enhanced_llm_nlp_transformation(self, text, target_intensity):
        """Combine LLM and NLP augmentation techniques"""
        if not self.llm:
            return self._enhanced_nlp_transformation(text, target_intensity)

        # 50% chance to start with LLM, 50% to start with NLP
        if random.random() < 0.5:
            # LLM first, then NLP
            llm_result = self._enhanced_llm_transformation(text, target_intensity)
            if llm_result != text:  # If LLM succeeded
                # Apply lighter NLP transformation
                nlp_techniques = [self._synonym_replacement, self._random_insertion]
                technique = random.choice(nlp_techniques)
                final_result = technique(llm_result, target_intensity)
            else:
                final_result = self._enhanced_nlp_transformation(text, target_intensity)
        else:
            # NLP first, then LLM
            nlp_result = self._enhanced_nlp_transformation(text, target_intensity)
            if nlp_result != text:  # If NLP succeeded
                # Apply LLM transformation with lighter approach
                final_result = self._enhanced_llm_transformation(nlp_result, target_intensity)
            else:
                final_result = self._enhanced_llm_transformation(text, target_intensity)

        return final_result



    def _enhanced_llm_transformation(self, text, target_intensity):
        """Enhanced LLM transformation with better prompting"""
        if not self.llm:
            return self._enhanced_classical_transformation(text, target_intensity)

        # Get ESConv style patterns for target intensity
        patterns = self.esconv_processor.style_patterns.get(target_intensity, {})
        keywords = self.esconv_processor.intensity_keywords.get(target_intensity, [])

        # Get actual ESConv examples
        esconv_examples = self._get_esconv_examples(target_intensity, max_examples=2)

        # Create intensity-specific instructions
        if target_intensity == 'low':
            intensity_instruction = """Transform this into a calm, reflective message that someone might share when feeling mildly concerned or seeking gentle support. 
                                        The tone should be measured and thoughtful, focusing on seeking understanding rather than expressing crisis."""
        else:
            intensity_instruction = """Transform this into a message expressing significant emotional distress that someone might share when feeling overwhelmed and needing urgent support.
                                        The tone should convey genuine emotional pain and urgency for help."""

        prompt = f"""You are helping transform messages to match the style of real emotional support conversations.
        Here are examples of {target_intensity} intensity messages from actual support conversations:
        {esconv_examples}

        Key characteristics for {target_intensity} intensity:
        - Common words: {', '.join(keywords[:8])}
        - Average length: {patterns.get('avg_length', 50):.0f} words
        - Personal tone: {patterns.get('personal_pronouns', [])}

        {intensity_instruction}

        Original message: "{text}"

        Rewrite it naturally in the style shown above:"""

        try:
            output = self.llm(prompt, max_tokens=200, temperature=0.7,
                              stop=["Original:", "Rewrite:", "Here are", "\n\nOriginal"])
            result = output["choices"][0]["text"].strip()

            # Clean up the result
            result = re.sub(r'^["\-\*\s]+', '', result)
            result = re.sub(r'["\-\*\s]+$', '', result)

            return result if result and len(result) > 10 else text
        except Exception as e:
            print(f"LLM transformation error: {e}")
            return self._enhanced_classical_transformation(text, target_intensity)

    def _get_esconv_examples(self, intensity, max_examples=3):
        """Get actual ESConv examples for the specified intensity"""
        if self.esconv_processor.esconv_data is None:
            return "No examples available."

        # Get text column
        text_cols = [col for col in self.esconv_processor.esconv_data.columns
                     if any(keyword in col.lower() for keyword in ['content', 'utterance', 'text', 'message'])]
        text_col = text_cols[0] if text_cols else self.esconv_processor.esconv_data.columns[0]

        # Filter by intensity
        filtered = self.esconv_processor.esconv_data[
            self.esconv_processor.esconv_data['binary_intensity'] == intensity
            ]

        # Get good examples (appropriate length)
        examples = []
        for _, row in filtered.iterrows():
            content = str(row[text_col]).strip()
            if 20 < len(content) < 150:
                examples.append(content)

        if not examples:
            return "No examples available."

        random.shuffle(examples)
        selected = examples[:max_examples]

        return "\n".join([f"- {ex}" for ex in selected])

    def _enhanced_classical_transformation(self, text, target_intensity):
        """Enhanced classical transformation using ESConv patterns"""
        patterns = self.esconv_processor.style_patterns.get(target_intensity, {})
        keywords = self.esconv_processor.intensity_keywords.get(target_intensity, [])

        transformed = text

        # Apply intensity-specific transformations
        if target_intensity == 'low':
            # Make language more measured and reflective
            transformed = self._soften_and_reflect(transformed, patterns)
        else:
            # Make language more emotionally intense
            transformed = self._intensify_emotion(transformed, patterns, keywords)

        # Apply style adjustments based on ESConv patterns
        transformed = self._adjust_to_esconv_style(transformed, patterns)

        return transformed

    def _soften_and_reflect(self, text, patterns):
        """Soften language for low intensity"""
        # Replace intense words with milder alternatives
        replacements = {
            'terrible': 'difficult',
            'awful': 'challenging',
            'horrible': 'tough',
            'hate': 'dislike',
            'furious': 'frustrated',
            'devastated': 'upset',
            'destroyed': 'affected'
        }

        for old, new in replacements.items():
            text = re.sub(r'\b' + old + r'\b', new, text, flags=re.IGNORECASE)

        # Add reflective elements if they appear in patterns
        starters = patterns.get('sentence_starters', [])
        if 'i' in starters and not text.lower().startswith('i'):
            if random.random() < 0.3:
                text = "I " + text.lower()

        return text

    def _calculate_transformation_quality(self, original, transformed, target_intensity):
        """Calculate quality score for transformation"""
        # Check if transformation is meaningful
        if original.lower() == transformed.lower():
            return 0.5  # No change

        # Check length appropriateness
        target_patterns = self.esconv_processor.style_patterns.get(target_intensity, {})
        target_length = target_patterns.get('avg_length', 50)
        actual_length = len(transformed.split())

        # Length score (closer to target = better)
        length_score = 1.0 - min(abs(actual_length - target_length) / target_length, 1.0)

        # Check for intensity-appropriate keywords
        keywords = self.esconv_processor.intensity_keywords.get(target_intensity, [])
        keyword_matches = sum(1 for keyword in keywords[:10] if keyword.lower() in transformed.lower())
        keyword_score = min(keyword_matches / 3.0, 1.0)  # Max score if 3+ keywords match

        # Check for personal pronouns (ESConv style)
        personal_pronouns = ['i', 'me', 'my', 'myself']
        has_personal = any(pronoun in transformed.lower() for pronoun in personal_pronouns)
        personal_score = 1.0 if has_personal else 0.5

        # Overall quality score
        quality_score = (length_score * 0.4 + keyword_score * 0.4 + personal_score * 0.2)

        return quality_score

    def _intensify_emotion(self, text, patterns, keywords):
        """Intensify language for high intensity"""
        # Add emotional keywords if appropriate
        if keywords and random.random() < 0.4:
            keyword = random.choice(keywords[:5])
            if keyword not in text.lower():
                # Add keyword naturally
                if random.random() < 0.5:
                    text = f"I'm feeling {keyword} - {text}"
                else:
                    text = f"{text} I just feel so {keyword}."

        # Increase personal involvement
        if 'i' not in text.lower()[:10] and random.random() < 0.5:
            text = "I " + text.lower()

        return text

    def _adjust_to_esconv_style(self, text, patterns):
        """Adjust text to match ESConv style patterns"""
        words = text.split()
        target_length = patterns.get('avg_length', 50)

        # Adjust length if needed
        if len(words) > target_length * 1.5:
            # Shorten text
            words = words[:int(target_length * 1.2)]
            text = ' '.join(words)
        elif len(words) < target_length * 0.5:
            # Slightly expand with connecting phrases
            connectors = ["and", "but", "because", "so", "also"]
            if random.random() < 0.3:
                connector = random.choice(connectors)
                text = f"{text} {connector} I'm not sure what to do."

        return text

    def balance_augmented_datasets(self):
        print("=== Balancing Augmented Datasets with High Quality ===")

        augmented_files = {
            'mixed': 'esconv_enhanced_mixed_augmentation_70percent.xlsx',
            'llm': 'esconv_enhanced_llm_augmentation_70percent.xlsx',
            'classical': 'esconv_enhanced_classical_augmentation_70percent.xlsx'
        }

        for method_name, filename in augmented_files.items():
            try:
                print(f"\n--- Balancing {method_name} dataset ---")

                augmented_data = pd.read_excel(filename)
                print(f"Loaded {filename}: {len(augmented_data)} samples")
                print(f"Current distribution: {augmented_data['label'].value_counts().to_dict()}")

                balanced_data = self.augment_esconv_with_meisd_balanced(
                    augmented_data,
                    target_size_per_class=None,
                    mode=f'enhanced_{method_name}' if method_name != 'mixed' else 'enhanced_mixed'
                )

                augmented_data_updated = pd.concat([augmented_data, balanced_data.iloc[len(augmented_data):]], ignore_index=True)
                augmented_data_updated.to_excel(filename, index=False)
                print(f"Appended balanced data and saved to original file: {filename}")

            except FileNotFoundError:
                print(f"File {filename} not found, skipping...")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    def augment_esconv_with_meisd(self, esconv_data, augment_percent=70, mode='enhanced_mixed'):
        """
        Augment ESConv data using enhanced MEISD transformation

        Args:
            esconv_data (pd.DataFrame): ESConv classification dataset
            augment_percent (int): Percentage increase in data size
            mode (str): 'enhanced_llm', 'enhanced_classical', or 'enhanced_mixed'

        Returns:
            pd.DataFrame: Augmented dataset
        """
        if self.meisd_data is None:
            raise ValueError("MEISD data not loaded. Call setup() first.")

        print(f"Augmenting ESConv data with enhanced MEISD transformation using {mode} method ({augment_percent}% increase)")

        # Calculate how many samples to add per class
        class_counts = esconv_data['label'].value_counts()
        print(f"Original ESConv class distribution: {class_counts.to_dict()}")

        augmented_samples = {'Utterances': [], 'intensity': [], 'label': []}

        for label in class_counts.index:
            num_to_add = int(class_counts[label] * (augment_percent / 100))

            # Get MEISD samples for this label
            meisd_label_samples = self.meisd_data[self.meisd_data['label'] == label]

            if len(meisd_label_samples) == 0:
                print(f"Warning: No MEISD samples found for label {label}")
                continue

            print(f"Adding {num_to_add} samples for label {label}")

            # Transform MEISD samples to ESConv style
            quality_scores = []
            for i in tqdm(range(num_to_add), desc=f"Transforming MEISD for label {label}"):
                # Sample random MEISD entry
                sample = meisd_label_samples.sample(1).iloc[0]
                original_text = sample['conversation']
                target_intensity = sample['label']

                # Apply transformation based on mode
                if mode == 'enhanced_llm':
                    transformed_text = self._enhanced_llm_transformation(original_text, target_intensity)
                elif mode == 'enhanced_classical':
                    transformed_text = self._enhanced_classical_transformation(original_text, target_intensity)
                elif mode == 'enhanced_nlp':
                    transformed_text = self._enhanced_nlp_transformation(original_text, target_intensity)
                elif mode == 'enhanced_llm_nlp':
                    transformed_text = self._enhanced_llm_nlp_transformation(original_text, target_intensity)
                elif mode == 'enhanced_mixed':
                    # 70% enhanced LLM, 30% enhanced classical
                    if random.random() < 0.7:
                        transformed_text = self._enhanced_llm_transformation(original_text, target_intensity)
                    else:
                        transformed_text = self._enhanced_classical_transformation(original_text, target_intensity)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Calculate quality score
                quality = self._calculate_transformation_quality(original_text, transformed_text, target_intensity)
                quality_scores.append(quality)

                augmented_samples['Utterances'].append(transformed_text)
                augmented_samples['intensity'].append(target_intensity)
                augmented_samples['label'].append(label)

            print(f"Average transformation quality for label {label}: {np.mean(quality_scores):.3f}")

        # Create augmented dataframe
        augmented_df = pd.DataFrame(augmented_samples)

        # Combine with original ESConv data
        final_df = pd.concat([esconv_data, augmented_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

        final_counts = final_df['label'].value_counts()
        print(f"Final class distribution: {final_counts.to_dict()}")
        print(f"Total samples: {len(final_df)} (increase: {(len(final_df)/len(esconv_data)-1)*100:.1f}%)")

        return final_df

    def augment_esconv_with_meisd_balanced(self, esconv_data, target_size_per_class=None, mode='enhanced_mixed'):
        """
        Augment ESConv data with balanced classes

        Args:
            esconv_data (pd.DataFrame): ESConv classification dataset
            target_size_per_class (int): Target number of samples per class. If None, uses majority class size
            mode (str): 'enhanced_llm', 'enhanced_classical', or 'enhanced_mixed'

        Returns:
            pd.DataFrame: Balanced augmented dataset
        """
        if self.meisd_data is None:
            raise ValueError("MEISD data not loaded. Call setup() first.")

        print(f"Creating balanced augmented dataset using {mode} method")

        # Calculate target size
        class_counts = esconv_data['label'].value_counts()
        print(f"Original ESConv class distribution: {class_counts.to_dict()}")

        if target_size_per_class is None:
            # Use majority class size as target
            target_size_per_class = class_counts.max()

        print(f"Target size per class: {target_size_per_class}")

        augmented_samples = {'Utterances': [], 'intensity': [], 'label': []}

        for label in class_counts.index:
            current_count = class_counts[label]
            num_to_add = max(0, target_size_per_class - current_count)

            if num_to_add == 0:
                print(f"Class {label} already has target size ({current_count}), no augmentation needed")
                continue

            # Get MEISD samples for this label
            meisd_label_samples = self.meisd_data[self.meisd_data['label'] == label]

            if len(meisd_label_samples) == 0:
                print(f"Warning: No MEISD samples found for label {label}")
                continue

            print(f"Adding {num_to_add} samples for label {label} (current: {current_count}, target: {target_size_per_class})")

            # Transform MEISD samples to ESConv style
            quality_scores = []
            for i in tqdm(range(num_to_add), desc=f"Balancing class {label}"):
                # Sample random MEISD entry (with replacement if needed)
                sample = meisd_label_samples.sample(1).iloc[0]
                original_text = sample['conversation']
                target_intensity = sample['label']

                # Apply transformation based on mode
                if mode == 'enhanced_llm':
                    transformed_text = self._enhanced_llm_transformation(original_text, target_intensity)
                elif mode == 'enhanced_classical':
                    transformed_text = self._enhanced_classical_transformation(original_text, target_intensity)
                elif mode == 'enhanced_nlp':
                    transformed_text = self._enhanced_nlp_transformation(original_text, target_intensity)
                elif mode == 'enhanced_llm_nlp':
                    transformed_text = self._enhanced_llm_nlp_transformation(original_text, target_intensity)
                elif mode == 'enhanced_mixed':
                    # 70% enhanced LLM, 30% enhanced classical
                    if random.random() < 0.7:
                        transformed_text = self._enhanced_llm_transformation(original_text, target_intensity)
                    else:
                        transformed_text = self._enhanced_classical_transformation(original_text, target_intensity)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                # Calculate quality score
                quality = self._calculate_transformation_quality(original_text, transformed_text, target_intensity)
                quality_scores.append(quality)

                augmented_samples['Utterances'].append(transformed_text)
                augmented_samples['intensity'].append(target_intensity)
                augmented_samples['label'].append(label)

            if quality_scores:
                print(f"Average transformation quality for label {label}: {np.mean(quality_scores):.3f}")

        # Create augmented dataframe
        augmented_df = pd.DataFrame(augmented_samples)

        # Combine with original ESConv data
        final_df = pd.concat([esconv_data, augmented_df], ignore_index=True)


        final_counts = final_df['label'].value_counts()
        print(f"Final balanced class distribution: {final_counts.to_dict()}")
        print(f"Total samples: {len(final_df)}")
        print(f"Balance ratio: {final_counts.min()/final_counts.max():.3f}")

        return final_df

# Example usage
if __name__ == "__main__":
    ESCONV_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/esconv_both_parts.csv'
    MEISD_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/filtered_negative_MEISD_intensity_max_first_25_conv.csv'
    LLAMA_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/chatbot/llama-2-7b-chat.Q4_K_M.gguf'

    print("=== Enhanced ESConv-MEISD Data Augmentation ===")

    # Step 1: Setup Enhanced ESConv processor (uses existing intensity labels)
    print("\n=== Step 1: Processing ESConv with existing intensity labels ===")
    esconv_processor = EnhancedESConvProcessor(ESCONV_PATH)
    esconv_data = esconv_processor.load_data()

    # This will now use existing intensity labels from the dataset
    classification_data = esconv_processor.prepare_for_classification()

    # Step 2: Setup Enhanced MEISD augmenter
    print("\n=== Step 2: Setting up enhanced MEISD data augmenter ===")
    meisd_augmenter = EnhancedMEISDDataAugmenter(MEISD_PATH, esconv_processor, LLAMA_PATH)
    meisd_augmenter.setup()

    # Step 3: Run enhanced augmentation experiments
    print("\n=== Step 3: Running enhanced augmentation experiments ===")

    # Enhanced mixed augmentation (recommended)
    # print("\n--- Running enhanced mixed augmentation ---")
    # df_enhanced_mixed = meisd_augmenter.augment_esconv_with_meisd(
    #     classification_data, mode='enhanced_mixed', augment_percent=70
    # )
    #
    # df_enhanced_mixed_balanced = meisd_augmenter.augment_esconv_with_meisd_balanced(
    #     df_enhanced_mixed, mode='enhanced_mixed'
    # )
    # df_enhanced_mixed_balanced.to_excel("esconv_enhanced_mixed_augmentation_70percent_balanced.xlsx", index=False)

    # --- ENHANCED LLM ---
    # if meisd_augmenter.llm:
    #     print("\n--- Running enhanced LLM augmentation ---")
    #     df_enhanced_llm = meisd_augmenter.augment_esconv_with_meisd(
    #         classification_data, mode='enhanced_llm', augment_percent=70
    #     )
    #
    #     df_enhanced_llm_balanced = meisd_augmenter.augment_esconv_with_meisd_balanced(
    #         df_enhanced_llm, mode='enhanced_llm'
    #     )
    #     df_enhanced_llm_balanced.to_excel("esconv_enhanced_llm_augmentation_70percent_balanced.xlsx", index=False)
    #
    # # --- ENHANCED CLASSICAL ---
    # print("\n--- Running enhanced classical augmentation ---")
    # df_enhanced_classical = meisd_augmenter.augment_esconv_with_meisd(
    #     classification_data, mode='enhanced_classical', augment_percent=70
    # )
    #
    # df_enhanced_classical_balanced = meisd_augmenter.augment_esconv_with_meisd_balanced(
    #     df_enhanced_classical, mode='enhanced_classical'
    # )
    # df_enhanced_classical_balanced.to_excel("esconv_enhanced_classical_augmentation_70percent_balanced.xlsx", index=False)

    print("\n--- Running enhanced enhanced_nlp, enhanced_nlp augmentation ---")
    df_enhanced_nlp = meisd_augmenter.augment_esconv_with_meisd(
        classification_data, mode='enhanced_nlp', augment_percent=70
    )

    df_enhanced_nlp_balanced = meisd_augmenter.augment_esconv_with_meisd_balanced(
        df_enhanced_nlp, mode='enhanced_nlp'
    )
    df_enhanced_nlp_balanced.to_excel("esconv_enhanced_nlp_augmentation_70percent_balanced.xlsx", index=False)

    print("\n--- Running enhanced enhanced_nlp, enhanced_llm_nlp augmentation ---")
    df_enhanced_llm_nlp = meisd_augmenter.augment_esconv_with_meisd(
        classification_data, mode='enhanced_llm_nlp', augment_percent=70
    )

    df_enhanced_llm_nlp_balanced = meisd_augmenter.augment_esconv_with_meisd_balanced(
        df_enhanced_llm_nlp, mode='enhanced_llm_nlp'
    )
    df_enhanced_llm_nlp_balanced.to_excel("esconv_enhanced_llm_nlp_augmentation_70percent_balanced.xlsx", index=False)


    # Step 4: Save results
    print("\n=== Step 4: Saving results ===")
    classification_data.to_excel("esconv_original_with_existing_labels.xlsx", index=False)
    df_enhanced_llm_nlp.to_excel("esconv_enhanced_llm_nlp_augmentation_70percent.xlsx", index=False)

    # if meisd_augmenter.llm:
    #     df_enhanced_llm.to_excel("esconv_enhanced_llm_augmentation_70percent.xlsx", index=False)

    df_enhanced_nlp.to_excel("esconv_enhanced_nlp_augmentation_70percent.xlsx", index=False)

    print("Enhanced ESConv emotion intensity classification datasets ready!")
    print("\nKey improvements:")
    print("- Uses existing ESConv intensity labels (no artificial creation)")
    print("- Enhanced style transformation based on actual ESConv patterns")
    print("- Better LLM prompting with real ESConv examples")
    print("- Quality scoring for transformations")
    print("- Improved classical augmentation with ESConv style adaptation")

    print("\nZamykam komputer za 1 minutę...")
    time.sleep(60)
    os.system("shutdown /s /t 1")


