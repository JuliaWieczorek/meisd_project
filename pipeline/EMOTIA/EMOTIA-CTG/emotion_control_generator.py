"""
Enhanced Emotion-Controlled Text Generation (CTG)
Improvements:
- Fine-grained intensity control (1-3 scale)
- Multi-emotion blending with weights
- Sentiment-aware generation
- Style pattern extraction from ESConv
- Quality scoring and validation
- Batch generation support
"""

import numpy as np
import pandas as pd
import re
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer


class EnhancedEmotionCTG:
    """
    Controlled Text Generator with emotion, intensity, and sentiment control
    """

    def __init__(self, llama_path: str, esconv_path: str, verbose: bool = True):
        self.llama_path = llama_path
        self.esconv_path = esconv_path
        self.verbose = verbose
        self.llm = None
        self.esconv_data = None
        self.style_patterns = {}  # (emotion, intensity, sentiment) -> patterns

        self._load_llama()
        self._load_and_analyze_esconv()

    def _load_llama(self):
        """Load LLaMA model"""
        if self.verbose:
            print("Loading LLaMA model...")
        try:
            self.llm = Llama(
                model_path=self.llama_path,
                n_ctx=2048,
                n_threads=6,
                verbose=False
            )
            if self.verbose:
                print("LLaMA loaded successfully")
        except Exception as e:
            print(f"Failed to load LLaMA: {e}")
            self.llm = None

    def _load_and_analyze_esconv(self):
        """Load ESConv and extract style patterns"""
        if self.verbose:
            print("Loading ESConv data...")

        self.esconv_data = pd.read_csv(self.esconv_path, encoding='utf-8')

        # Find columns
        text_col = next((c for c in self.esconv_data.columns
                         if any(k in c.lower() for k in ['utterance', 'text', 'message'])),
                        self.esconv_data.columns[0])

        emotion_col = next((c for c in self.esconv_data.columns if 'emotion1' in c.lower()), None)
        intensity_col = next((c for c in self.esconv_data.columns if 'intensity1' in c.lower()), None)
        sentiment_col = next((c for c in self.esconv_data.columns if 'sentiment' in c.lower()), None)

        if not all([emotion_col, intensity_col, sentiment_col]):
            raise ValueError(f"Missing required columns in ESConv")

        if self.verbose:
            print(f"Analyzing style patterns from ESConv...")

        # Prepare data
        df = self.esconv_data.copy()
        df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
        df['intensity'] = df[intensity_col].astype(float).fillna(2.0)
        df['sentiment'] = df[sentiment_col].astype(str).str.lower().str.strip()
        df['text'] = df[text_col].astype(str)

        # Group by (emotion, intensity, sentiment)
        grouped = df.groupby(['emotion', 'intensity', 'sentiment'])

        for (emotion, intensity, sentiment), group in grouped:
            if len(group) < 3:  # Skip small groups
                continue

            texts = group['text'].dropna().tolist()
            if not texts:
                continue

            patterns = self._extract_detailed_patterns(texts)
            key = (emotion, int(intensity), sentiment)
            self.style_patterns[key] = patterns

        if self.verbose:
            print(f"Extracted {len(self.style_patterns)} style patterns")

    def _extract_detailed_patterns(self, texts: List[str]) -> Dict:
        """Extract linguistic patterns from texts"""
        patterns = {}

        # 1. Length statistics
        lengths = [len(t.split()) for t in texts]
        patterns['avg_length'] = np.mean(lengths)
        patterns['std_length'] = np.std(lengths)
        patterns['min_length'] = np.min(lengths)
        patterns['max_length'] = np.max(lengths)

        # 2. Sentence starters
        sentences = []
        for text in texts:
            sents = re.split(r'[.!?]+', text)
            sentences.extend([s.strip() for s in sents if s.strip()])

        starters = [s.split()[0].lower() for s in sentences if len(s.split()) > 0]
        starter_counts = Counter(starters)
        patterns['top_starters'] = [s for s, _ in starter_counts.most_common(10)]

        # 3. Personal pronouns
        personal_pronouns = ['i', 'me', 'my', 'myself', 'we', 'us', 'our', 'you', 'your']
        pronoun_freq = {}
        for pronoun in personal_pronouns:
            count = sum(text.lower().count(f' {pronoun} ') + text.lower().startswith(f'{pronoun} ')
                        for text in texts)
            pronoun_freq[pronoun] = count / len(texts)
        patterns['pronoun_freq'] = pronoun_freq

        # 4. Punctuation patterns
        patterns['question_ratio'] = sum(1 for t in texts if '?' in t) / len(texts)
        patterns['exclamation_ratio'] = sum(1 for t in texts if '!' in t) / len(texts)
        patterns['ellipsis_ratio'] = sum(1 for t in texts if '...' in t) / len(texts)

        # 5. Common phrases (TF-IDF)
        try:
            vectorizer = TfidfVectorizer(
                max_features=30,
                stop_words='english',
                ngram_range=(2, 3),  # Bigrams and trigrams
                min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-15:]
            patterns['key_phrases'] = [feature_names[i] for i in top_indices]
        except:
            patterns['key_phrases'] = []

        # 6. Examples
        patterns['examples'] = texts[:3]  # Keep 3 examples

        return patterns

    def generate_single(
            self,
            emotions: List[str],
            intensities: List[float],
            sentiment: str,
            base_text: str = "",
            temperature: float = 0.8,
            max_tokens: int = 200,
            use_style_guidance: bool = True
    ) -> Dict:
        """
        Generate text with emotion/intensity/sentiment control

        Args:
            emotions: List of emotion names (e.g., ['joy', 'hope'])
            intensities: List of intensity values 1-3 (e.g., [2, 3])
            sentiment: 'positive', 'negative', or 'neutral'
            base_text: Optional base text to rewrite
            temperature: Generation temperature (0.1-1.0)
            max_tokens: Maximum tokens to generate
            use_style_guidance: Whether to use ESConv style patterns

        Returns:
            Dict with 'generated_text', 'quality_score', and metadata
        """

        if not self.llm:
            return {
                'generated_text': base_text,
                'quality_score': 0.0,
                'error': 'LLaMA model not loaded'
            }

        # Filter valid emotions
        valid_pairs = [(e, i) for e, i in zip(emotions, intensities)
                       if pd.notna(e) and pd.notna(i)]

        if not valid_pairs:
            return {
                'generated_text': base_text,
                'quality_score': 0.0,
                'error': 'No valid emotion/intensity pairs'
            }

        # Build emotion context
        emotion_context = ", ".join([f"{e} (intensity {i})" for e, i in valid_pairs])

        # Get primary emotion for style matching
        primary_emotion = valid_pairs[0][0].lower()
        primary_intensity = int(valid_pairs[0][1])

        # Build prompt
        prompt = self._build_prompt(
            emotion_context=emotion_context,
            sentiment=sentiment,
            base_text=base_text,
            primary_emotion=primary_emotion,
            primary_intensity=primary_intensity,
            use_style_guidance=use_style_guidance
        )

        # Generate
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\nOriginal:", "Examples:", "\n\n---"]
            )
            result = output["choices"][0]["text"].strip()

            # Clean up
            result = re.sub(r'^["*\-\s]+', '', result)
            result = re.sub(r'["*\-\s]+$', '', result)

            # Fallback if too short
            if len(result) < 10:
                result = base_text if base_text else "I'm not sure how to express this."

            # Calculate quality
            quality = self._calculate_quality(
                result,
                primary_emotion,
                primary_intensity,
                sentiment
            )

            return {
                'generated_text': result,
                'quality_score': quality,
                'emotion_context': emotion_context,
                'sentiment': sentiment,
                'base_text': base_text,
                'temperature': temperature
            }

        except Exception as e:
            return {
                'generated_text': base_text,
                'quality_score': 0.0,
                'error': str(e)
            }

    def _build_prompt(
            self,
            emotion_context: str,
            sentiment: str,
            base_text: str,
            primary_emotion: str,
            primary_intensity: int,
            use_style_guidance: bool
    ) -> str:
        """Build generation prompt with style guidance"""

        # Get style patterns
        key = (primary_emotion, primary_intensity, sentiment)
        patterns = self.style_patterns.get(key, {})

        # Build prompt sections
        sections = [
            "You are an empathetic assistant generating emotionally expressive text.",
            "",
            f"Target emotional state: {emotion_context}",
            f"Target sentiment: {sentiment}",
            ""
        ]

        if use_style_guidance and patterns:
            # Add style guidance
            sections.append("Style characteristics to match:")

            # Length
            avg_len = patterns.get('avg_length', 50)
            sections.append(f"- Length: around {int(avg_len)} words")

            # Sentence starters
            starters = patterns.get('top_starters', [])
            if starters:
                sections.append(f"- Start sentences with words like: {', '.join(starters[:5])}")

            # Key phrases
            phrases = patterns.get('key_phrases', [])
            if phrases:
                sections.append(f"- Use phrases like: {', '.join(phrases[:5])}")

            # Pronouns
            pronoun_freq = patterns.get('pronoun_freq', {})
            top_pronouns = sorted(pronoun_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_pronouns:
                pronoun_list = [p[0] for p in top_pronouns]
                sections.append(f"- Use pronouns: {', '.join(pronoun_list)}")

            # Examples
            examples = patterns.get('examples', [])
            if examples:
                sections.append("")
                sections.append("Examples of similar emotional expressions:")
                for ex in examples[:2]:
                    sections.append(f'  "{ex}"')

            sections.append("")

        # Task instruction
        if base_text:
            sections.append(f'Original message: "{base_text}"')
            sections.append("")
            sections.append("Rewrite this message to:")
        else:
            sections.append("Generate a message that:")

        sections.extend([
            f"- Expresses {emotion_context}",
            f"- Maintains {sentiment} sentiment",
            "- Sounds natural and authentic",
            "- Is emotionally supportive and empathetic",
            "",
            "Generated message:"
        ])

        return "\n".join(sections)

    def _calculate_quality(
            self,
            text: str,
            emotion: str,
            intensity: int,
            sentiment: str
    ) -> float:
        """Calculate quality score for generated text"""

        if not text or len(text) < 10:
            return 0.0

        score = 0.0

        # Get patterns
        key = (emotion, intensity, sentiment)
        patterns = self.style_patterns.get(key, {})

        if not patterns:
            return 0.5  # Neutral score if no patterns available

        # 1. Length score (25%)
        target_len = patterns.get('avg_length', 50)
        actual_len = len(text.split())
        length_diff = abs(actual_len - target_len) / max(target_len, 1)
        length_score = max(0, 1.0 - length_diff)
        score += length_score * 0.25

        # 2. Key phrases score (30%)
        phrases = patterns.get('key_phrases', [])
        if phrases:
            phrase_matches = sum(1 for phrase in phrases if phrase.lower() in text.lower())
            phrase_score = min(phrase_matches / 3.0, 1.0)
            score += phrase_score * 0.30
        else:
            score += 0.15

        # 3. Pronoun usage score (20%)
        pronoun_freq = patterns.get('pronoun_freq', {})
        if pronoun_freq:
            text_lower = text.lower()
            pronoun_matches = sum(1 for p in pronoun_freq.keys() if f' {p} ' in text_lower or text_lower.startswith(f'{p} '))
            pronoun_score = min(pronoun_matches / 3.0, 1.0)
            score += pronoun_score * 0.20
        else:
            score += 0.10

        # 4. Sentence starter score (15%)
        starters = patterns.get('top_starters', [])
        if starters:
            first_word = text.split()[0].lower() if text.split() else ''
            starter_match = first_word in starters
            score += (1.0 if starter_match else 0.5) * 0.15
        else:
            score += 0.075

        # 5. Punctuation score (10%)
        punct_score = 0.0
        if patterns.get('question_ratio', 0) > 0.3 and '?' in text:
            punct_score += 0.5
        if patterns.get('exclamation_ratio', 0) > 0.2 and '!' in text:
            punct_score += 0.5
        score += punct_score * 0.10

        return round(score, 3)

    def generate_batch(
            self,
            batch_configs: List[Dict],
            output_csv: Optional[str] = None,
            show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Generate multiple texts in batch

        Args:
            batch_configs: List of dicts with keys: emotions, intensities, sentiment, base_text
            output_csv: Optional path to save results
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with generated texts and metadata
        """

        results = []

        iterator = tqdm(batch_configs, desc="Generating") if show_progress else batch_configs

        for config in iterator:
            result = self.generate_single(
                emotions=config.get('emotions', []),
                intensities=config.get('intensities', []),
                sentiment=config.get('sentiment', 'neutral'),
                base_text=config.get('base_text', ''),
                temperature=config.get('temperature', 0.8),
                use_style_guidance=config.get('use_style_guidance', True)
            )

            # Merge config and result
            full_result = {**config, **result}
            results.append(full_result)

        df = pd.DataFrame(results)

        if output_csv:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            if self.verbose:
                print(f"\nSaved {len(df)} generated samples to: {output_csv}")

        return df

    def evaluate_quality(self, df: pd.DataFrame) -> Dict:
        """Evaluate overall quality of generated texts"""

        if 'quality_score' not in df.columns:
            return {'error': 'No quality_score column found'}

        summary = {
            'total_samples': len(df),
            'avg_quality': round(df['quality_score'].mean(), 3),
            'std_quality': round(df['quality_score'].std(), 3),
            'min_quality': round(df['quality_score'].min(), 3),
            'max_quality': round(df['quality_score'].max(), 3),
            'high_quality_ratio': round((df['quality_score'] >= 0.7).sum() / len(df), 3),
            'low_quality_ratio': round((df['quality_score'] < 0.4).sum() / len(df), 3)
        }

        return summary


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Paths
    LLAMA_PATH = "C:/Users/juwieczo/DataspellProjects/meisd_project/models/llama-2-7b-chat.Q5_K_M.gguf"
    ESCONV_PATH = "C:/Users/juwieczo/DataspellProjects/meisd_project/data/ESConv_DA_ready.csv"
    OUTPUT_DIR = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/EMOTIA/EMOTIA-DA/outputs"

    # Initialize generator
    print("\n" + "="*60)
    print("=== ENHANCED EMOTION-CONTROLLED TEXT GENERATION ===")
    print("="*60)

    generator = EnhancedEmotionCTG(
        llama_path=LLAMA_PATH,
        esconv_path=ESCONV_PATH,
        verbose=True
    )

    # Example 1: Single generation
    print("\n" + "-"*60)
    print("Example 1: Single generation")
    print("-"*60)

    result = generator.generate_single(
        emotions=['joy', 'hope'],
        intensities=[2, 3],
        sentiment='positive',
        base_text="I feel like things might finally get better.",
        temperature=0.8
    )

    print(f"\nOriginal: {result['base_text']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Quality: {result['quality_score']:.3f}")

    # Example 2: Batch generation
    print("\n" + "-"*60)
    print("Example 2: Batch generation")
    print("-"*60)

    batch_configs = [
        {
            'emotions': ['sadness', 'fear'],
            'intensities': [3, 2],
            'sentiment': 'negative',
            'base_text': "I don't know if I can handle this anymore.",
        },
        {
            'emotions': ['anger'],
            'intensities': [3],
            'sentiment': 'negative',
            'base_text': "This situation is completely unfair.",
        },
        {
            'emotions': ['joy', 'gratitude'],
            'intensities': [3, 2],
            'sentiment': 'positive',
            'base_text': "Thank you for being there when I needed support.",
        },
        {
            'emotions': ['neutral'],
            'intensities': [2],
            'sentiment': 'neutral',
            'base_text': "I'm thinking about what to do next.",
        }
    ]

    df_generated = generator.generate_batch(
        batch_configs=batch_configs,
        output_csv=f"{OUTPUT_DIR}/ctg_examples.csv",
        show_progress=True
    )

    # Evaluate quality
    quality_summary = generator.evaluate_quality(df_generated)

    print("\n" + "="*60)
    print("QUALITY SUMMARY")
    print("="*60)
    for key, value in quality_summary.items():
        print(f"  {key:20s}: {value}")

    print("\nCTG demonstration complete!")