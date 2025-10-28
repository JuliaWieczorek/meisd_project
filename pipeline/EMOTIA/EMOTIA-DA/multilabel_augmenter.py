# ============================================================
# === ENHANCED MULTILABEL AUGMENTATION (Emotion + Sentiment + Intensity) ===
# ============================================================
import pandas as pd
import numpy as np
import random
import re
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# === ENHANCED ESConv PROCESSOR (Multilabel-aware) ===
# ------------------------------------------------------------
class MultilabelESConvProcessor:
    """
    ESConv processor that extracts style patterns for:
    - Emotion (emotion1, emotion2, emotion3)
    - Intensity (1-3 numeric scale)
    - Sentiment (positive, negative, neutral)
    """

    def __init__(self, esconv_path):
        self.esconv_path = esconv_path
        self.esconv_data = None
        self.style_patterns = {}  # (emotion, intensity, sentiment) -> patterns

    def load_data(self):
        """Load ESConv data"""
        print("Loading ESConv data...")
        self.esconv_data = pd.read_csv(self.esconv_path)
        print(f"ESConv data loaded: {len(self.esconv_data)} rows")
        return self.esconv_data

    def analyze_multilabel_patterns(self, save_analysis=True, output_dir=None):
        """
        Analyze ESConv patterns grouped by (emotion, intensity, sentiment)

        Args:
            save_analysis: Whether to save detailed analysis reports
            output_dir: Directory to save analysis files
        """
        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        # Find columns
        text_cols = [c for c in self.esconv_data.columns
                     if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        # Find emotion, intensity, sentiment columns
        emotion_col = next((c for c in self.esconv_data.columns if 'emotion1' in c.lower()), None)
        intensity_col = next((c for c in self.esconv_data.columns if 'intensity1' in c.lower()), None)
        sentiment_col = next((c for c in self.esconv_data.columns if 'sentiment' in c.lower()), None)

        if not all([emotion_col, intensity_col, sentiment_col]):
            raise ValueError(f"Missing required columns. Found: emotion={emotion_col}, intensity={intensity_col}, sentiment={sentiment_col}")

        print(f"Using columns: text={text_col}, emotion={emotion_col}, intensity={intensity_col}, sentiment={sentiment_col}")

        # Prepare data
        df = self.esconv_data.copy()
        df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
        df['intensity'] = df[intensity_col].astype(float).fillna(2.0)
        df['sentiment'] = df[sentiment_col].astype(str).str.lower().str.strip()
        df['text'] = df[text_col].astype(str)

        # Storage for analysis
        analysis_data = []

        # Group by (emotion, intensity, sentiment)
        print("\nAnalyzing style patterns by (emotion, intensity, sentiment)...")
        grouped = df.groupby(['emotion', 'intensity', 'sentiment'])

        for (emotion, intensity, sentiment), group in tqdm(grouped, desc="Analyzing groups"):
            if len(group) < 5:  # Skip small groups
                continue

            texts = group['text'].dropna().tolist()
            if not texts:
                continue

            # Extract patterns
            patterns = self._extract_patterns(texts)
            patterns['sample_count'] = len(texts)

            # Extract keywords using TF-IDF
            keywords = self._extract_keywords_tfidf(texts, top_n=20)
            patterns['keywords'] = keywords

            # Store
            key = (emotion, int(intensity), sentiment)
            self.style_patterns[key] = patterns

            # Store for analysis report
            analysis_data.append({
                'emotion': emotion,
                'intensity': int(intensity),
                'sentiment': sentiment,
                'sample_count': patterns['sample_count'],
                'avg_length': round(patterns['avg_length'], 2),
                'num_keywords': len(keywords),
                'top_5_keywords': ', '.join(keywords[:5]),
                'top_3_starters': ', '.join(patterns['sentence_starters'][:3]),
                'question_ratio': round(patterns['question_ratio'], 3),
                'exclamation_ratio': round(patterns['exclamation_ratio'], 3)
            })

        print(f"Extracted patterns for {len(self.style_patterns)} unique (emotion, intensity, sentiment) combinations")

        # Print sample
        sample_keys = list(self.style_patterns.keys())[:3]
        for key in sample_keys:
            emotion, intensity, sentiment = key
            patterns = self.style_patterns[key]
            print(f"  {emotion} (intensity={intensity}, sentiment={sentiment}): "
                  f"{patterns['sample_count']} samples, "
                  f"avg_length={patterns['avg_length']:.1f}, "
                  f"{len(patterns['keywords'])} keywords")

        # Save detailed analysis if requested
        if save_analysis and output_dir:
            self._save_analysis_reports(analysis_data, output_dir)

    def extract_emotion_intensifiers(self, top_n=10):
        """
        Extract emotion-specific intensifier words using TF-IDF from ESConv data.
        Intensifiers are typically adverbs that modify emotional expressions.

        Returns:
            dict: {emotion: [list of intensifier words]}
        """
        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        print("\nExtracting emotion-specific intensifiers from ESConv...")

        # Find text column
        text_cols = [c for c in self.esconv_data.columns
                     if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        # Find emotion column
        emotion_col = next((c for c in self.esconv_data.columns if 'emotion1' in c.lower()), None)
        if not emotion_col:
            print("No emotion column found")
            return {}

        df = self.esconv_data.copy()
        df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
        df['text'] = df[text_col].astype(str)

        # Common intensifier patterns (adverbs, degree words)
        intensifier_keywords = [
            'very', 'really', 'so', 'extremely', 'absolutely', 'completely',
            'totally', 'deeply', 'highly', 'incredibly', 'particularly',
            'especially', 'quite', 'rather', 'fairly', 'pretty', 'somewhat',
            'constantly', 'always', 'never', 'forever', 'endlessly',
            'utterly', 'purely', 'entirely', 'wholly', 'fully'
        ]

        emotion_intensifiers = {}

        # Group by emotion
        grouped = df.groupby('emotion')

        for emotion, group in grouped:
            if len(group) < 10:  # Skip small groups
                continue

            texts = group['text'].dropna().tolist()
            if not texts:
                continue

            try:
                # Extract using TF-IDF with focus on unigrams
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words=None,  # Don't remove stop words (intensifiers might be there)
                    ngram_range=(1, 1),  # Only single words
                    min_df=2,
                    token_pattern=r'\b[a-zA-Z]+\b'  # Only alphabetic tokens
                )
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()

                # Get mean TF-IDF scores
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

                # Create word-score dictionary
                word_scores = {word: score for word, score in zip(feature_names, mean_scores)}

                # Filter for intensifier keywords only
                emotion_specific_intensifiers = []
                for word in intensifier_keywords:
                    if word in word_scores:
                        emotion_specific_intensifiers.append((word, word_scores[word]))

                # Sort by TF-IDF score
                emotion_specific_intensifiers.sort(key=lambda x: x[1], reverse=True)

                # Take top N
                top_intensifiers = [word for word, score in emotion_specific_intensifiers[:top_n]]

                if top_intensifiers:
                    emotion_intensifiers[emotion] = top_intensifiers
                    print(f"  {emotion}: {', '.join(top_intensifiers[:5])} ...")
                else:
                    # Fallback: extract any high TF-IDF adverbs
                    all_words_sorted = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
                    # Look for words ending in 'ly' (adverbs) or common intensifiers
                    adverbs = [w for w, s in all_words_sorted
                               if w.endswith('ly') or w in intensifier_keywords][:top_n]
                    if adverbs:
                        emotion_intensifiers[emotion] = adverbs
                        print(f"  {emotion}: {', '.join(adverbs[:5])} ... (adverbs)")

            except Exception as e:
                print(f"Error extracting intensifiers for {emotion}: {e}")
                continue

        print(f"\nExtracted intensifiers for {len(emotion_intensifiers)} emotions")

        # Save to file for inspection
        if hasattr(self, 'output_dir') and self.output_dir:
            intensifiers_path = os.path.join(self.output_dir, 'emotion_intensifiers_tfidf.txt')
            with open(intensifiers_path, 'w', encoding='utf-8') as f:
                f.write("EMOTION-SPECIFIC INTENSIFIERS (TF-IDF)\n")
                f.write("=" * 70 + "\n\n")
                for emotion, intensifiers in sorted(emotion_intensifiers.items()):
                    f.write(f"{emotion.upper()}:\n")
                    f.write(f"  {', '.join(intensifiers)}\n\n")
            print(f"Intensifiers saved to: {intensifiers_path}")

        return emotion_intensifiers

    def analyze_multilabel_patterns(self, save_analysis=True, output_dir=None):
        """
        Analyze ESConv patterns grouped by (emotion, intensity, sentiment)

        Args:
            save_analysis: Whether to save detailed analysis reports
            output_dir: Directory to save analysis files
        """
        # Store output_dir for later use
        self.output_dir = output_dir

        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        # ========================================
        # DODAJ TO ZABEZPIECZENIE (z wcześniejszej poprawki)
        # ========================================
        if len(self.esconv_data) == 0:
            print("\nWARNING: ESConv data is empty after filtering!")
            print("   Skipping pattern analysis.")
            return
        # ========================================

        # Find columns
        text_cols = [c for c in self.esconv_data.columns
                     if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        # Find emotion, intensity, sentiment columns
        emotion_col = next((c for c in self.esconv_data.columns if 'emotion1' in c.lower()), None)
        intensity_col = next((c for c in self.esconv_data.columns if 'intensity1' in c.lower()), None)
        sentiment_col = next((c for c in self.esconv_data.columns if 'sentiment' in c.lower()), None)

        if not all([emotion_col, intensity_col, sentiment_col]):
            raise ValueError(
                f"Missing required columns. Found: emotion={emotion_col}, intensity={intensity_col}, sentiment={sentiment_col}")

        print(
            f"Using columns: text={text_col}, emotion={emotion_col}, intensity={intensity_col}, sentiment={sentiment_col}")

        # Prepare data
        df = self.esconv_data.copy()
        df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
        df['intensity'] = df[intensity_col].astype(float).fillna(2.0)
        df['sentiment'] = df[sentiment_col].astype(str).str.lower().str.strip()
        df['text'] = df[text_col].astype(str)

        # Storage for analysis
        analysis_data = []  # ← TO JEST INICJALIZACJA, której brakowało w Twoim kodzie

        # Group by (emotion, intensity, sentiment)
        print("\nAnalyzing style patterns by (emotion, intensity, sentiment)...")
        grouped = df.groupby(['emotion', 'intensity', 'sentiment'])

        for (emotion, intensity, sentiment), group in tqdm(grouped, desc="Analyzing groups"):
            if len(group) < 5:  # Skip small groups
                continue

            texts = group['text'].dropna().tolist()
            if not texts:
                continue

            # Extract patterns
            patterns = self._extract_patterns(texts)
            patterns['sample_count'] = len(texts)

            # Extract keywords using TF-IDF
            keywords = self._extract_keywords_tfidf(texts, top_n=20)
            patterns['keywords'] = keywords

            # Store
            key = (emotion, int(intensity), sentiment)
            self.style_patterns[key] = patterns

            # Store for analysis report
            analysis_data.append({
                'emotion': emotion,
                'intensity': int(intensity),
                'sentiment': sentiment,
                'sample_count': patterns['sample_count'],
                'avg_length': round(patterns['avg_length'], 2),
                'num_keywords': len(keywords),
                'top_5_keywords': ', '.join(keywords[:5]),
                'top_3_starters': ', '.join(patterns['sentence_starters'][:3]),
                'question_ratio': round(patterns['question_ratio'], 3),
                'exclamation_ratio': round(patterns['exclamation_ratio'], 3)
            })

        print(f"Extracted patterns for {len(self.style_patterns)} unique (emotion, intensity, sentiment) combinations")

        # Print sample
        sample_keys = list(self.style_patterns.keys())[:3]
        for key in sample_keys:
            emotion, intensity, sentiment = key
            patterns = self.style_patterns[key]
            print(f"  {emotion} (intensity={intensity}, sentiment={sentiment}): "
                  f"{patterns['sample_count']} samples, "
                  f"avg_length={patterns['avg_length']:.1f}, "
                  f"{len(patterns['keywords'])} keywords")

        # Save detailed analysis if requested
        if save_analysis and output_dir:
            self._save_analysis_reports(analysis_data, output_dir)

            # ========================================
            # ✨ Extract intensifiers (both types)
            # ========================================
            print("\n" + "=" * 70)
            print("EXTRACTING EMOTION-SPECIFIC FEATURES")
            print("=" * 70)

            # 1. Emotion-only intensifiers (fallback)
            self.emotion_intensifiers = self.extract_emotion_intensifiers(top_n=10)

            # 2. Intensity-aware intensifiers (preferred)
            self.intensity_aware_intensifiers = self.extract_intensity_aware_intensifiers(top_n=10)

            print("\nAll analysis and feature extraction complete!")
            # ========================================
    def extract_intensity_aware_intensifiers(self, top_n=10):
        """
        Extract intensifiers grouped by (emotion, intensity) for more precise augmentation

        Returns:
            dict: {(emotion, intensity): [list of intensifier words]}
        """
        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        print("\nExtracting intensity-aware intensifiers from ESConv...")

        text_col = next((c for c in self.esconv_data.columns
                         if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])),
                        self.esconv_data.columns[0])
        emotion_col = next((c for c in self.esconv_data.columns if 'emotion1' in c.lower()), None)
        intensity_col = next((c for c in self.esconv_data.columns if 'intensity1' in c.lower()), None)

        if not all([emotion_col, intensity_col]):
            return {}

        df = self.esconv_data.copy()
        df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
        df['intensity'] = df[intensity_col].astype(float)
        df['text'] = df[text_col].astype(str)

        intensifier_keywords = [
            'very', 'really', 'so', 'extremely', 'absolutely', 'completely',
            'totally', 'deeply', 'highly', 'incredibly', 'particularly',
            'especially', 'quite', 'rather', 'fairly', 'pretty', 'somewhat',
            'constantly', 'always', 'never', 'forever', 'endlessly',
            'utterly', 'purely', 'entirely', 'wholly', 'fully', 'barely',
            'slightly', 'somewhat', 'moderately', 'considerably'
        ]

        intensity_aware_intensifiers = {}

        # Group by (emotion, intensity)
        grouped = df.groupby(['emotion', 'intensity'])

        for (emotion, intensity), group in grouped:
            if len(group) < 10:
                continue

            texts = group['text'].dropna().tolist()
            if not texts:
                continue

            try:
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words=None,
                    ngram_range=(1, 1),
                    min_df=2,
                    token_pattern=r'\b[a-zA-Z]+\b'
                )
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                word_scores = {word: score for word, score in zip(feature_names, mean_scores)}

                emotion_specific = []
                for word in intensifier_keywords:
                    if word in word_scores:
                        emotion_specific.append((word, word_scores[word]))

                emotion_specific.sort(key=lambda x: x[1], reverse=True)
                top_intensifiers = [word for word, score in emotion_specific[:top_n]]

                if top_intensifiers:
                    key = (emotion, int(intensity))
                    intensity_aware_intensifiers[key] = top_intensifiers
                    print(f"  {emotion} (intensity={int(intensity)}): {', '.join(top_intensifiers[:3])}")

            except Exception as e:
                continue

        return intensity_aware_intensifiers

    def _extract_patterns(self, texts):
        """Extract linguistic patterns from texts"""
        patterns = {}

        # Average length
        patterns['avg_length'] = np.mean([len(t.split()) for t in texts])

        # Sentence starters
        sentences = []
        for text in texts:
            sents = re.split(r'[.!?]+', text)
            sentences.extend([s.strip() for s in sents if s.strip()])

        starters = [s.split()[0].lower() for s in sentences if len(s.split()) > 0]
        starter_counts = Counter(starters)
        patterns['sentence_starters'] = [s for s, _ in starter_counts.most_common(15)]

        # Personal pronouns frequency
        personal_pronouns = ['i', 'me', 'my', 'myself', 'we', 'us', 'our']
        pronoun_freq = {}
        for pronoun in personal_pronouns:
            count = sum(text.lower().count(f' {pronoun} ') + text.lower().count(f'{pronoun} ')
                        for text in texts)
            pronoun_freq[pronoun] = count / len(texts)
        patterns['pronoun_freq'] = pronoun_freq

        # Question/exclamation ratio
        patterns['question_ratio'] = sum(1 for t in texts if '?' in t) / len(texts)
        patterns['exclamation_ratio'] = sum(1 for t in texts if '!' in t) / len(texts)

        return patterns

    def _extract_keywords_tfidf(self, texts, top_n=20):
        """Extract keywords using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-top_n:]
            keywords = [feature_names[i] for i in top_indices]

            return keywords
        except Exception as e:
            print(f"TF-IDF extraction error: {e}")
            return []

    def get_examples(self, emotion, intensity, sentiment, max_examples=3):
        """Get example texts for given emotion, intensity, sentiment"""
        if self.esconv_data is None:
            return []

        # Find matching samples
        df = self.esconv_data.copy()
        text_col = next((c for c in df.columns
                         if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])),
                        df.columns[0])

        emotion_col = next((c for c in df.columns if 'emotion1' in c.lower()), None)
        intensity_col = next((c for c in df.columns if 'intensity1' in c.lower()), None)
        sentiment_col = next((c for c in df.columns if 'sentiment' in c.lower()), None)

        if not all([emotion_col, intensity_col, sentiment_col]):
            return []

        # Filter
        mask = (
                (df[emotion_col].astype(str).str.lower() == str(emotion).lower()) &
                (df[intensity_col].astype(float) == float(intensity)) &
                (df[sentiment_col].astype(str).str.lower() == str(sentiment).lower())
        )

        matches = df[mask]
        if len(matches) == 0:
            return []

        # Sample
        sample_size = min(max_examples, len(matches))
        samples = matches.sample(sample_size)[text_col].tolist()

        return samples

    def _save_analysis_reports(self, analysis_data, output_dir):
        """Save detailed analysis reports for paper/article"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. Main analysis table
        df_analysis = pd.DataFrame(analysis_data)
        df_analysis = df_analysis.sort_values(['emotion', 'intensity', 'sentiment'])

        analysis_path = os.path.join(output_dir, 'esconv_pattern_analysis.csv')
        df_analysis.to_csv(analysis_path, index=False, encoding='utf-8')
        print(f"\nSaved pattern analysis to: {analysis_path}")

        # 2. Detailed keyword report
        keyword_data = []
        for key, patterns in self.style_patterns.items():
            emotion, intensity, sentiment = key
            keywords = patterns.get('keywords', [])

            for rank, keyword in enumerate(keywords[:20], 1):
                keyword_data.append({
                    'emotion': emotion,
                    'intensity': intensity,
                    'sentiment': sentiment,
                    'keyword_rank': rank,
                    'keyword': keyword
                })

        df_keywords = pd.DataFrame(keyword_data)
        keywords_path = os.path.join(output_dir, 'esconv_tfidf_keywords.csv')
        df_keywords.to_csv(keywords_path, index=False, encoding='utf-8')
        print(f"Saved TF-IDF keywords to: {keywords_path}")

        # 3. Summary statistics by emotion
        summary_by_emotion = df_analysis.groupby('emotion').agg({
            'sample_count': 'sum',
            'avg_length': 'mean',
            'num_keywords': 'mean',
            'question_ratio': 'mean',
            'exclamation_ratio': 'mean'
        }).round(2)

        emotion_summary_path = os.path.join(output_dir, 'esconv_emotion_summary.csv')
        summary_by_emotion.to_csv(emotion_summary_path, encoding='utf-8')
        print(f"📊 Saved emotion summary to: {emotion_summary_path}")

        # 4. Summary statistics by intensity
        summary_by_intensity = df_analysis.groupby('intensity').agg({
            'sample_count': 'sum',
            'avg_length': 'mean',
            'num_keywords': 'mean',
            'question_ratio': 'mean',
            'exclamation_ratio': 'mean'
        }).round(2)

        intensity_summary_path = os.path.join(output_dir, 'esconv_intensity_summary.csv')
        summary_by_intensity.to_csv(intensity_summary_path, encoding='utf-8')
        print(f"Saved intensity summary to: {intensity_summary_path}")

        # 5. Summary statistics by sentiment
        summary_by_sentiment = df_analysis.groupby('sentiment').agg({
            'sample_count': 'sum',
            'avg_length': 'mean',
            'num_keywords': 'mean',
            'question_ratio': 'mean',
            'exclamation_ratio': 'mean'
        }).round(2)

        sentiment_summary_path = os.path.join(output_dir, 'esconv_sentiment_summary.csv')
        summary_by_sentiment.to_csv(sentiment_summary_path, encoding='utf-8')
        print(f"📊 Saved sentiment summary to: {sentiment_summary_path}")

        # 6. Text-based report for paper
        report_lines = [
            "="*80,
            "ESConv MULTILABEL PATTERN ANALYSIS REPORT",
            "="*80,
            "",
            f"Total unique (emotion, intensity, sentiment) combinations: {len(self.style_patterns)}",
            f"Total samples analyzed: {df_analysis['sample_count'].sum()}",
            "",
            "="*80,
            "SUMMARY BY EMOTION",
            "="*80,
            summary_by_emotion.to_string(),
            "",
            "="*80,
            "SUMMARY BY INTENSITY",
            "="*80,
            summary_by_intensity.to_string(),
            "",
            "="*80,
            "SUMMARY BY SENTIMENT",
            "="*80,
            summary_by_sentiment.to_string(),
            "",
            "="*80,
            "TOP 5 MOST COMMON COMBINATIONS",
            "="*80,
            ]

        top_combinations = df_analysis.nlargest(5, 'sample_count')[
            ['emotion', 'intensity', 'sentiment', 'sample_count', 'avg_length', 'top_5_keywords']
        ]
        report_lines.append(top_combinations.to_string(index=False))

        report_lines.extend([
            "",
            "="*80,
            "EXAMPLE KEYWORDS BY EMOTION (Top 3 combinations)",
            "="*80,
            ])

        for emotion in df_analysis['emotion'].unique()[:5]:
            emotion_data = df_analysis[df_analysis['emotion'] == emotion].head(3)
            if len(emotion_data) > 0:
                report_lines.append(f"\n{emotion.upper()}:")
                for _, row in emotion_data.iterrows():
                    report_lines.append(
                        f"  Intensity={row['intensity']}, Sentiment={row['sentiment']}: "
                        f"{row['top_5_keywords']}"
                    )

        report_path = os.path.join(output_dir, 'esconv_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f" Saved text report to: {report_path}")

        print("\n All analysis reports saved!")
        print(f"   Use these files for your paper/article:")
        print(f"   - Pattern analysis: esconv_pattern_analysis.csv")
        print(f"   - TF-IDF keywords: esconv_tfidf_keywords.csv")
        print(f"   - Summaries by emotion/intensity/sentiment")
        print(f"   - Text report: esconv_analysis_report.txt")


# ------------------------------------------------------------
# === ENHANCED MULTILABEL AUGMENTER ===
# ------------------------------------------------------------
class MultilabelMEISDAugmenter:
    """
    Enhanced augmenter that considers:
    - Multiple emotions (emotion1, emotion2, emotion3)
    - Multiple intensities (intensity1, intensity2, intensity3)
    - Sentiment (positive, negative, neutral)
    """

    def __init__(self, meisd_path, esconv_processor, llama_path=None):
        self.meisd_path = meisd_path
        self.esconv_processor = esconv_processor
        self.llama_path = llama_path
        self.meisd_data = None
        self.llm = None

    def setup(self):
        """Setup augmenter"""
        print("Setting up multilabel MEISD augmenter...")

        # Load MEISD
        self.meisd_data = pd.read_csv(self.meisd_path)
        print(f"MEISD data loaded: {len(self.meisd_data)} rows")

        # Analyze ESConv patterns if not done
        if not self.esconv_processor.style_patterns:
            print("Analyzing ESConv multilabel patterns...")
            self.esconv_processor.analyze_multilabel_patterns()

        # Load LLaMA
        if self.llama_path:
            try:
                print("Loading LLaMA model...")
                # ========================================
                # POPRAWKA: Konwertuj Path na string
                # ========================================
                llama_path_str = str(self.llama_path)  # ← DODAJ TO!
                self.llm = Llama(
                    model_path=llama_path_str,
                    n_ctx=2048,
                    n_threads=6,
                    chat_format="llama-2",
                    verbose=False
                )
                # ========================================
                print("LLaMA loaded successfully")
            except Exception as e:
                print(f"Failed to load LLaMA: {e}")
                self.llm = None

    def augment_multilabel(self, num_samples=10, mode='llm', save_details=True,
                           target_emotion=None, target_intensity=None, target_sentiment=None):
        """
        Main augmentation method

        Args:
            num_samples: Number of samples to augment
            mode: 'llm', 'eda', or 'mixed'
            save_details: Whether to save detailed transformation info
            target_emotion: Target emotion for augmentation (overrides MEISD labels)
            target_intensity: Target intensity for augmentation (overrides MEISD labels)
            target_sentiment: Target sentiment for augmentation (overrides MEISD labels)

        Returns:
            DataFrame with augmented samples
        """
        samples = self.meisd_data.sample(min(num_samples, len(self.meisd_data)))
        augmented = []
        transformation_details = []

        for idx, row in tqdm(samples.iterrows(), total=len(samples), desc=f"Augmenting ({mode})"):
            # Extract original data
            text = str(row.get('Utterances') or row.get('conversation') or '')

            # ========================================
            # ZMIANA: Użyj TARGET labels jeśli podane
            # ========================================
            if target_emotion is not None and target_intensity is not None:
                # Używamy targetowych etykiet zamiast oryginalnych z MEISD
                primary_emotion = target_emotion
                primary_intensity = target_intensity
                sentiment = target_sentiment if target_sentiment is not None else 'negative'

                # Dla compatibility z resztą kodu
                emotions = [primary_emotion, None, None]
                intensities = [primary_intensity, None, None]
            else:
                # Oryginalny kod (gdy nie ma target labels)
                emotions = [
                    row.get('emotion1'),
                    row.get('emotion2'),
                    row.get('emotion3')
                ]
                intensities = [
                    row.get('intensity1'),
                    row.get('intensity2'),
                    row.get('intensity3')
                ]
                sentiment = row.get('sentiment', 'neutral')

                primary_emotion = emotions[0] if pd.notna(emotions[0]) else 'neutral'
                primary_intensity = intensities[0] if pd.notna(intensities[0]) else 2.0
            # ========================================

            # Get ESConv style patterns FOR TARGET emotion/intensity/sentiment
            key = (str(primary_emotion).lower(), int(primary_intensity), str(sentiment).lower())
            patterns = self.esconv_processor.style_patterns.get(key, {})

            # Transform based on mode
            if mode == 'llm' and self.llm:
                transformed = self._llm_transform(
                    text, emotions, intensities, sentiment, patterns
                )
            elif mode == 'eda':
                transformed = self._eda_transform(
                    text, emotions, intensities, sentiment, patterns
                )
            elif mode == 'mixed':
                if random.random() < 0.7 and self.llm:
                    transformed = self._llm_transform(
                        text, emotions, intensities, sentiment, patterns
                    )
                else:
                    transformed = self._eda_transform(
                        text, emotions, intensities, sentiment, patterns
                    )
            else:
                transformed = text

            # Calculate quality
            quality = self._calculate_quality(text, transformed, patterns)

            # Store result
            augmented.append({
                'sentiment': sentiment,
                'emotion1': emotions[0],
                'intensity1': intensities[0],
                'emotion2': emotions[1],
                'intensity2': intensities[1],
                'emotion3': emotions[2],
                'intensity3': intensities[2],
                'original': text,
                'augmented': transformed,
                'quality': quality,
                'mode': mode
            })

            # Store transformation details for analysis
            if save_details:
                transformation_details.append({
                    'sample_id': idx,
                    'original_length': len(text.split()),
                    'transformed_length': len(transformed.split()),
                    'target_length': patterns.get('avg_length', 'N/A'),
                    'keywords_available': len(patterns.get('keywords', [])),
                    'pattern_key': f"{primary_emotion}_{int(primary_intensity)}_{sentiment}",
                    'quality': quality,
                    'mode': mode
                })

        df_result = pd.DataFrame(augmented)

        # Store transformation details
        if save_details:
            self.transformation_details = pd.DataFrame(transformation_details)

        return df_result

    def balance_and_expand_dataset(self, esconv_df, mode='mixed', balance_by='emotion1', expand_percent=70):
        """
        1. Balance dataset by chosen label (emotion1 / intensity1 / sentiment)
        2. Expand balanced dataset by given percentage
        """
        print(f"\n=== BALANCING by {balance_by.upper()} ===")
        class_counts = esconv_df[balance_by].value_counts()
        print("Current distribution:", class_counts.to_dict())

        majority = class_counts.max()
        balanced_samples = []

        for label, count in class_counts.items():
            needed = majority - count
            subset = esconv_df[esconv_df[balance_by] == label]

            if needed > 0:
                print(f"→ Augmenting {label}: need {needed} more samples")
                # używamy self (nie trzeba podawać augmenter)
                aug_subset = self.augment_multilabel(num_samples=min(needed, len(subset)), mode=mode)
                aug_subset[balance_by] = label
                balanced_samples.append(aug_subset)

        df_balanced = pd.concat([esconv_df] + balanced_samples, ignore_index=True)
        print("New balanced distribution:", df_balanced[balance_by].value_counts().to_dict())

        # --- Expansion step ---
        print(f"\n=== EXPANDING balanced dataset by {expand_percent}% ===")
        expand_n = int(len(df_balanced) * (expand_percent / 100))
        df_expand = self.augment_multilabel(num_samples=expand_n, mode=mode)
        df_expanded = pd.concat([df_balanced, df_expand], ignore_index=True)

        print(f"Final dataset size: {len(df_expanded)} (was {len(esconv_df)})")
        return df_expanded

    def balance_and_expand_2d(self, esconv_df, mode='mixed', expand_percent=40):
        """
        1. Balance dataset by (emotion1, intensity1)
        2. Expand balanced dataset by given percentage
        """
        print("\n=== BALANCING by (emotion1, intensity1) ===")
        df = esconv_df.copy()
        df['emotion1'] = df['emotion1'].astype(str).str.lower().str.strip()
        df['intensity1'] = df['intensity1'].astype(float)

        group_counts = df.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
        print("Current combination distribution:")
        print(group_counts.sort_values('count', ascending=False).head(10))

        max_count = group_counts['count'].max()
        balanced_groups = []

        for _, row in group_counts.iterrows():
            emotion, intensity, count = row['emotion1'], row['intensity1'], row['count']
            needed = int(max_count - count)
            subset = df[(df['emotion1'] == emotion) & (df['intensity1'] == intensity)]

            if needed > 0 and not subset.empty:
                print(f"→ Augmenting ({emotion}, {intensity}) by {needed} samples")

                # ========================================
                # ZMIANA: Przekaż TARGET emotion/intensity
                # ========================================
                aug_subset = self.augment_multilabel(
                    num_samples=min(needed, len(subset)),
                    mode=mode,
                    target_emotion=emotion,  # ← NOWE!
                    target_intensity=intensity,  # ← NOWE!
                    target_sentiment='negative'  # ← NOWE! (domyślnie negative dla ESConv)
                )
                # ========================================

                aug_subset['emotion1'] = emotion
                aug_subset['intensity1'] = intensity
                balanced_groups.append(aug_subset)

        df_balanced = pd.concat([df] + balanced_groups, ignore_index=True)

        new_counts = df_balanced.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
        print("\nNew balanced distribution (top 10):")
        print(new_counts.sort_values('count', ascending=False).head(10))

        # Expansion
        print(f"\n=== EXPANDING balanced dataset by {expand_percent}% ===")
        expand_n = int(len(df_balanced) * (expand_percent / 100))

        # Expansion: losowo wybieraj target emotion/intensity z rozkładu
        df_expand_samples = []
        for _ in range(expand_n):
            # Losuj kombinację proporcjonalnie do występowania
            random_combo = df_balanced.sample(1)
            target_emotion = random_combo['emotion1'].values[0]
            target_intensity = random_combo['intensity1'].values[0]

            aug_sample = self.augment_multilabel(
                num_samples=1,
                mode=mode,
                target_emotion=target_emotion,
                target_intensity=target_intensity,
                target_sentiment='negative'
            )
            df_expand_samples.append(aug_sample)

        df_expand = pd.concat(df_expand_samples, ignore_index=True) if df_expand_samples else pd.DataFrame()

        df_final = pd.concat([df_balanced, df_expand], ignore_index=True)
        print(f"Final dataset size: {len(df_final)} (was {len(esconv_df)})")

        return df_final

    def _llm_transform(self, text, emotions, intensities, sentiment, patterns):
        """Transform using LLM with ESConv style guidance (improved version)"""
        if not self.llm:
            return text

        # Build emotion context
        emotion_list = [f"{e} (intensity {i})" for e, i in zip(emotions, intensities) if pd.notna(e) and pd.notna(i)]
        emotion_context = ", ".join(emotion_list) if emotion_list else "neutral"

        # Get ESConv examples
        primary_emotion = emotions[0] if pd.notna(emotions[0]) else 'neutral'
        primary_intensity = intensities[0] if pd.notna(intensities[0]) else 2.0

        examples = self.esconv_processor.get_examples(primary_emotion, int(primary_intensity), sentiment,
                                                      max_examples=2)
        examples_str = "\n".join([f"- {ex}" for ex in examples]) if examples else "No examples available."

        # Get keywords and target length
        keywords = patterns.get('keywords', [])
        keywords_str = ", ".join(keywords[:10]) if keywords else "supportive, understanding"
        target_length = patterns.get('avg_length', 50)

        # ---- CHAT-STYLE PROMPT ----
        prompt = f"""[INST]
    You are an empathetic rewriting assistant.
    Rewrite the following message so that it matches the tone and style of emotional support conversations.

    Emotional context: {emotion_context}
    Sentiment: {sentiment}
    Target length: about {int(target_length)} words.

    Examples from real support conversations:
    {examples_str}

    Use similar language and keep the response natural and conversational.
    Focus on expressing emotions and empathy.

    Original: "{text}"
    Rewritten: [/INST]
    """

        try:
            output = self.llm(prompt, max_tokens=200, temperature=0.8)
            result = output["choices"][0].get("text", "").strip()

            # Fallback if empty
            if not result:
                print("⚠️ Empty LLaMA output — retrying with simpler prompt")
                simple_prompt = f"Rewrite to express {primary_emotion} (intensity {primary_intensity}/3): {text}"
                output = self.llm(simple_prompt, max_tokens=150, temperature=0.8)
                result = output["choices"][0].get("text", "").strip()

            # Clean up
            result = re.sub(r'^["\-\*\s]+', '', result)
            result = re.sub(r'["\-\*\s]+$', '', result)

            return result if len(result.split()) > 3 else text

        except Exception as e:
            print(f"LLM error: {e}")
            return text

    def _eda_transform(self, text, emotions, intensities, sentiment, patterns):
        """
        Enhanced EDA transformation using ESConv patterns
        """
        transformed = text
        words = transformed.split()

        # Get target characteristics
        keywords = patterns.get('keywords', [])
        target_length = patterns.get('avg_length', 50)
        starters = patterns.get('sentence_starters', [])

        # Get emotion info
        primary_emotion = emotions[0] if pd.notna(emotions[0]) else 'neutral'
        primary_intensity = intensities[0] if pd.notna(intensities[0]) else 2.0

        # ========================================
        # ZAKTUALIZOWANE: Użyj intensity-aware TF-IDF intensifiers
        # ========================================
        # Try to get intensity-aware intensifiers first (emotion, intensity)
        emotion_intensifiers = getattr(self.esconv_processor, 'intensity_aware_intensifiers', {})

        # Get intensifiers based on (emotion, intensity) tuple
        key = (str(primary_emotion).lower(), int(primary_intensity))
        intensifiers = emotion_intensifiers.get(key, [])

        # Fallback 1: Try emotion-only (without intensity)
        if not intensifiers:
            emotion_only_intensifiers = getattr(self.esconv_processor, 'emotion_intensifiers', {})
            intensifiers = emotion_only_intensifiers.get(str(primary_emotion).lower(), [])

        # Fallback 2: Use general intensifiers from all emotions
        if not intensifiers:
            all_intensifiers = []
            # Try intensity-aware first
            if emotion_intensifiers:
                for int_list in emotion_intensifiers.values():
                    all_intensifiers.extend(int_list)
            # Then try emotion-only
            if not all_intensifiers:
                emotion_only_intensifiers = getattr(self.esconv_processor, 'emotion_intensifiers', {})
                for int_list in emotion_only_intensifiers.values():
                    all_intensifiers.extend(int_list)

            intensifiers = list(set(all_intensifiers))[:10] if all_intensifiers else ['really', 'very', 'so']

        # Add intensifier based on intensity level
        # Higher intensity = more likely to add intensifier
        intensity_probability = {
            1.0: 0.2,  # Low intensity: 20% chance
            2.0: 0.4,  # Medium intensity: 40% chance
            3.0: 0.6  # High intensity: 60% chance
        }

        prob = intensity_probability.get(float(primary_intensity), 0.3)

        if intensifiers and random.random() < prob:
            intensifier = random.choice(intensifiers)

            # Smart insertion: before adjectives/adverbs or after "feel/felt"
            intensifier_inserted = False

            # Try to insert after "feel" or "felt"
            for i, word in enumerate(words):
                if word.lower() in ['feel', 'felt', 'feeling', 'feels'] and i < len(words) - 1:
                    words.insert(i + 1, intensifier)
                    intensifier_inserted = True
                    break

            # If no "feel" found, insert at random position (not first or last)
            if not intensifier_inserted and len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                words.insert(insert_pos, intensifier)

        transformed = ' '.join(words)
        # ========================================

        # 1. Insert relevant keywords (20% chance per keyword)
        words = transformed.split()  # Re-split after intensifier insertion
        if keywords and len(words) < target_length * 1.2:
            num_inserts = max(1, int(len(words) * 0.2))
            selected_keywords = random.sample(keywords, min(num_inserts, len(keywords)))

            for kw in selected_keywords:
                if kw.lower() not in transformed.lower():
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, kw)

        transformed = ' '.join(words)

        # 2. Adjust sentence starter
        if starters and random.random() < 0.3:
            starter = random.choice(starters)
            words = transformed.split()
            current_starter = words[0].lower() if words else ''
            if current_starter not in starters:
                words[0] = starter.capitalize()
                transformed = ' '.join(words)

        # 3. Add personal pronouns if missing
        personal_pronouns = ['I', 'me', 'my', 'myself']
        has_personal = any(p.lower() in transformed.lower() for p in personal_pronouns)

        if not has_personal and random.random() < 0.4:
            pronoun = random.choice(personal_pronouns)
            transformed = f"{pronoun} feel like {transformed.lower()}"

        # 4. Adjust length
        words = transformed.split()
        if len(words) > target_length * 1.5:
            transformed = ' '.join(words[:int(target_length * 1.2)])
        elif len(words) < target_length * 0.5:
            connectors = ["and", "because", "but"]
            transformed += f" {random.choice(connectors)} I'm not sure what to do."

        # 5. Synonym replacement
        words = transformed.split()
        num_replace = max(1, len(words) // 10)

        try:
            from nltk.corpus import wordnet

            for _ in range(num_replace):
                if len(words) < 2:
                    break

                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                word_clean = re.sub(r'[^\w]', '', word.lower())

                if len(word_clean) < 3 or word_clean in ['the', 'and', 'but', 'for', 'not']:
                    continue

                synsets = wordnet.synsets(word_clean)
                if synsets:
                    lemmas = synsets[0].lemmas()
                    synonyms = [l.name().replace('_', ' ') for l in lemmas if l.name() != word_clean]

                    if synonyms:
                        synonym = random.choice(synonyms)
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        words[idx] = synonym
        except:
            pass

        transformed = ' '.join(words)

        return transformed
    def _calculate_quality(self, original, transformed, patterns):
        """Calculate transformation quality score (0-1)"""
        if original.strip().lower() == transformed.strip().lower():
            return 0.5  # No change

        score = 0.0

        # Length score (40%)
        target_length = patterns.get('avg_length', 50)
        actual_length = len(transformed.split())
        length_diff = abs(actual_length - target_length) / target_length
        length_score = max(0, 1.0 - length_diff)
        score += length_score * 0.4

        # Keyword score (30%)
        keywords = patterns.get('keywords', [])
        if keywords:
            keyword_matches = sum(1 for kw in keywords[:10] if kw.lower() in transformed.lower())
            keyword_score = min(keyword_matches / 3.0, 1.0)
            score += keyword_score * 0.3
        else:
            score += 0.15  # Partial credit if no keywords available

        # Personal pronoun score (20%)
        personal_pronouns = ['i', 'me', 'my', 'myself', 'we', 'us', 'our']
        has_personal = any(p in transformed.lower() for p in personal_pronouns)
        score += (1.0 if has_personal else 0.5) * 0.2

        # Sentence starter score (10%)
        starters = patterns.get('sentence_starters', [])
        if starters:
            first_word = transformed.split()[0].lower() if transformed.split() else ''
            starter_match = first_word in starters
            score += (1.0 if starter_match else 0.5) * 0.1
        else:
            score += 0.05

        return round(score, 3)


# ------------------------------------------------------------
# === HELPER FUNCTIONS ===
# ------------------------------------------------------------
def summarize_augmentation_quality(df_aug, mode, save_path=None):
    """Summarize augmentation quality"""
    summary = {
        "mode": mode,
        "samples": len(df_aug),
        "avg_quality": round(df_aug["quality"].mean(), 3) if "quality" in df_aug.columns else "N/A",
        "min_quality": round(df_aug["quality"].min(), 3) if "quality" in df_aug.columns else "N/A",
        "max_quality": round(df_aug["quality"].max(), 3) if "quality" in df_aug.columns else "N/A"
    }

    if save_path:
        pd.DataFrame([summary]).to_csv(save_path, index=False)
        print(f"Summary saved to: {save_path}")

    return summary


def filter_meisd_for_esconv_compatibility(meisd_df, esconv_df, min_samples=20, allowed_intensities=[1.0, 2.0, 3.0],  # ← ZMIANA: dodane 3.0
        remove_incompatible_emotions=True):
    """
    Filter MEISD dataset for ESConv compatibility:
    1. Keep only emotions present in ESConv (optional)
    2. Keep specified intensities (default: 1.0, 2.0, 3.0)
    3. Remove rare combinations (< min_samples)

    Args:
        meisd_df: MEISD DataFrame
        esconv_df: ESConv DataFrame (for extracting allowed emotions)
        min_samples: Minimum samples per (emotion, intensity) combination
        allowed_intensities: List of allowed intensity values
        remove_incompatible_emotions: If True, remove emotions not in ESConv

    Returns:
        Filtered MEISD DataFrame, detailed report
    """
    print("\n" + "=" * 70)
    print("FILTERING MEISD FOR ESCONV COMPATIBILITY")
    print("=" * 70)

    # Extract allowed emotions from ESConv
    esconv_emotions = set(esconv_df['emotion1'].astype(str).str.lower().str.strip().unique())
    print(f"\nESConv emotions ({len(esconv_emotions)}): {sorted(esconv_emotions)}")

    # MEISD emotions
    meisd_emotions = set(meisd_df['emotion1'].astype(str).str.lower().str.strip().unique())
    print(f"MEISD emotions ({len(meisd_emotions)}): {sorted(meisd_emotions)}")

    # Find compatible and incompatible emotions
    compatible_emotions = esconv_emotions & meisd_emotions
    incompatible_emotions = meisd_emotions - esconv_emotions

    print(f"\nCompatible emotions ({len(compatible_emotions)}): {sorted(compatible_emotions)}")
    print(f"Incompatible emotions ({len(incompatible_emotions)}): {sorted(incompatible_emotions)}")

    # Statistics before filtering
    original_size = len(meisd_df)

    meisd_filtered = meisd_df.copy()
    meisd_filtered['emotion1'] = meisd_filtered['emotion1'].astype(str).str.lower().str.strip()
    meisd_filtered['intensity1'] = meisd_filtered['intensity1'].astype(float)

    # STEP 1: Filter by emotion compatibility (OPTIONAL)
    if remove_incompatible_emotions:
        print(f"\n{'=' * 70}")
        print("STEP 1: Filtering by emotion compatibility")
        print(f"{'=' * 70}")

        # Count samples to be removed by emotion
        removed_by_emotion = {}
        for emotion in incompatible_emotions:
            count = len(meisd_filtered[meisd_filtered['emotion1'] == emotion])
            removed_by_emotion[emotion] = count
            print(f"Removing '{emotion}': {count} samples")

        meisd_filtered = meisd_filtered[meisd_filtered['emotion1'].isin(compatible_emotions)]
        print(
            f"\n  After emotion filter: {len(meisd_filtered)} samples (removed {original_size - len(meisd_filtered)})")
    else:
        print(f"\nSKIPPING emotion filtering (keeping all emotions)")
        removed_by_emotion = {}

    # STEP 2: Filter by intensity (UPDATED - keeps 3.0)
    print(f"\n{'=' * 70}")
    print("STEP 2: Filtering by intensity")
    print(f"{'=' * 70}")
    print(f"  Allowed intensities: {allowed_intensities}")

    # Count samples by intensity
    intensity_counts = meisd_filtered['intensity1'].value_counts().sort_index()
    print(f"\n  Current MEISD intensity distribution:")
    removed_by_intensity_count = 0
    for intensity, count in intensity_counts.items():
        if intensity in allowed_intensities:
            print(f"    ✓ KEEP intensity {intensity}: {count} samples")
        else:
            print(f"    ✗ REMOVE intensity {intensity}: {count} samples")
            removed_by_intensity_count += count

    before_intensity_filter = len(meisd_filtered)
    meisd_filtered = meisd_filtered[meisd_filtered['intensity1'].isin(allowed_intensities)]
    removed_by_intensity = before_intensity_filter - len(meisd_filtered)

    if removed_by_intensity > 0:
        print(f"\n  After intensity filter: {len(meisd_filtered)} samples (removed {removed_by_intensity})")
    else:
        print(f"\n  ✓ All intensities kept: {len(meisd_filtered)} samples")

    # STEP 3: Filter rare combinations
    print(f"\n{'=' * 70}")
    print(f"STEP 3: Filtering rare combinations (min_samples={min_samples})")
    print(f"{'=' * 70}")

    group_counts = meisd_filtered.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
    rare_combinations = group_counts[group_counts['count'] < min_samples]

    if len(rare_combinations) > 0:
        print(f"\n  🗑️  Removing {len(rare_combinations)} rare combinations:")
        for _, row in rare_combinations.iterrows():
            print(f"    - {row['emotion1']} (intensity={row['intensity1']}): {row['count']} samples")

        # Filter out rare combinations
        keep_combinations = group_counts[group_counts['count'] >= min_samples]
        mask = meisd_filtered.set_index(['emotion1', 'intensity1']).index.isin(
            keep_combinations.set_index(['emotion1', 'intensity1']).index
        )
        before_rarity_filter = len(meisd_filtered)
        meisd_filtered = meisd_filtered[mask].reset_index(drop=True)
        removed_by_rarity = before_rarity_filter - len(meisd_filtered)

        print(f"\n  After rarity filter: {len(meisd_filtered)} samples (removed {removed_by_rarity})")
    else:
        print(f"\n  ✓ No rare combinations found (all have ≥{min_samples} samples)")
        removed_by_rarity = 0

    # FINAL SUMMARY
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Original MEISD size: {original_size}")
    print(f"  Filtered MEISD size: {len(meisd_filtered)}")
    print(
        f"  Total removed: {original_size - len(meisd_filtered)} ({(original_size - len(meisd_filtered)) / original_size * 100:.1f}%)")
    print(f"\n  Breakdown:")
    print(f"    - Removed by emotion: {sum(removed_by_emotion.values())} samples")
    print(f"    - Removed by intensity: {removed_by_intensity} samples")
    print(f"    - Removed by rarity: {removed_by_rarity} samples")

    # Final distribution
    print(f"\n{'=' * 70}")
    print("FINAL DISTRIBUTION (sorted by count)")
    print(f"{'=' * 70}")
    final_counts = meisd_filtered.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
    final_counts = final_counts.sort_values('count', ascending=False)
    print(final_counts.to_string(index=False))

    # Generate detailed report
    report = {
        'original_size': original_size,
        'filtered_size': len(meisd_filtered),
        'removed_total': original_size - len(meisd_filtered),
        'removed_by_emotion': removed_by_emotion,
        'removed_by_intensity': removed_by_intensity,
        'removed_by_rarity': removed_by_rarity,
        'compatible_emotions': list(compatible_emotions) if remove_incompatible_emotions else list(meisd_emotions),
        'incompatible_emotions': list(incompatible_emotions) if remove_incompatible_emotions else [],
        'final_combinations': final_counts.to_dict('records')
    }

    return meisd_filtered, report


# ------------------------------------------------------------
# === FILTERING FUNCTIONS ===
# ------------------------------------------------------------

def filter_rare_combinations(df, min_samples=20, group_cols=['emotion1', 'intensity1']):
    """
    Remove rare (emotion, intensity) combinations that have too few samples

    Args:
        df: DataFrame to filter
        min_samples: Minimum samples required per combination
        group_cols: Columns to group by (default: emotion1, intensity1)

    Returns:
        Filtered DataFrame, removed combinations info
    """
    print(f"\n{'=' * 70}")
    print(f"FILTERING RARE COMBINATIONS (min_samples={min_samples})")
    print(f"{'=' * 70}")

    # WAŻNE: Pracuj na KOPII, aby nie modyfikować oryginalnych danych
    df_work = df.copy()

    # Normalizuj tylko do celów liczenia (bez modyfikacji oryginalnych kolumn)
    df_work['_emotion_normalized'] = df_work[group_cols[0]].astype(str).str.lower().str.strip()
    df_work['_intensity_normalized'] = df_work[group_cols[1]].astype(float)

    # Count samples per combination
    group_counts = df_work.groupby(['_emotion_normalized', '_intensity_normalized']).size().reset_index(name='count')
    group_counts.columns = [group_cols[0], group_cols[1], 'count']  # Przywróć oryginalne nazwy

    # Identify rare and keep combinations
    rare_combinations = group_counts[group_counts['count'] < min_samples]
    keep_combinations = group_counts[group_counts['count'] >= min_samples]

    print(f"\nOriginal combinations: {len(group_counts)}")
    print(f"   Combinations to REMOVE: {len(rare_combinations)}")
    print(f"   Combinations to KEEP: {len(keep_combinations)}")

    if len(rare_combinations) > 0:
        print(f"\n🗑️  REMOVING these rare combinations:")
        for _, row in rare_combinations.iterrows():
            emotion = row[group_cols[0]]
            intensity = row[group_cols[1]]
            count = row['count']
            print(f"    - {emotion} (intensity={intensity}): {count} samples")

    # Filter: keep only combinations with enough samples
    if len(keep_combinations) > 0:
        # Utwórz maskę na znormalizowanych kolumnach
        keep_set = set(zip(keep_combinations[group_cols[0]], keep_combinations[group_cols[1]]))
        mask = df_work.apply(
            lambda row: (row['_emotion_normalized'], row['_intensity_normalized']) in keep_set,
            axis=1
        )

        df_filtered = df[mask].reset_index(drop=True)  # Użyj ORYGINALNEGO df, nie df_work
    else:
        print("\nWARNING: No combinations meet the minimum threshold!")
        df_filtered = pd.DataFrame(columns=df.columns)  # Pusty DataFrame z tymi samymi kolumnami

    print(f"\nDataset filtered: {len(df)} → {len(df_filtered)} samples")
    print(f"   Removed: {len(df) - len(df_filtered)} samples ({(len(df) - len(df_filtered)) / len(df) * 100:.1f}%)")

    # Summary statistics
    removed_info = {
        'original_size': len(df),
        'filtered_size': len(df_filtered),
        'removed_samples': len(df) - len(df_filtered),
        'removed_combinations': rare_combinations.to_dict('records')
    }

    return df_filtered, removed_info

def filter_meisd_for_esconv_compatibility(meisd_df, esconv_df, min_samples=15, allowed_intensities=[1.0, 2.0, 3.0],
        remove_incompatible_emotions=True):
    """
    Filter MEISD dataset for ESConv compatibility:
    1. Keep only emotions present in ESConv (optional)
    2. Keep specified intensities (default: 1.0, 2.0, 3.0)
    3. Remove rare combinations (< min_samples)

    Args:
        meisd_df: MEISD DataFrame
        esconv_df: ESConv DataFrame (for extracting allowed emotions)
        min_samples: Minimum samples per (emotion, intensity) combination
        allowed_intensities: List of allowed intensity values
        remove_incompatible_emotions: If True, remove emotions not in ESConv

    Returns:
        Filtered MEISD DataFrame, detailed report
    """
    print("\n" + "=" * 70)
    print("FILTERING MEISD FOR ESCONV COMPATIBILITY")
    print("=" * 70)

    # Extract allowed emotions from ESConv
    esconv_emotions = set(esconv_df['emotion1'].astype(str).str.lower().str.strip().unique())
    print(f"\nESConv emotions ({len(esconv_emotions)}): {sorted(esconv_emotions)}")

    # MEISD emotions
    meisd_emotions = set(meisd_df['emotion1'].astype(str).str.lower().str.strip().unique())
    print(f"MEISD emotions ({len(meisd_emotions)}): {sorted(meisd_emotions)}")

    # Find compatible and incompatible emotions
    compatible_emotions = esconv_emotions & meisd_emotions
    incompatible_emotions = meisd_emotions - esconv_emotions

    print(f"\n✓ Compatible emotions ({len(compatible_emotions)}): {sorted(compatible_emotions)}")
    print(f"✗ Incompatible emotions ({len(incompatible_emotions)}): {sorted(incompatible_emotions)}")

    # Statistics before filtering
    original_size = len(meisd_df)

    meisd_filtered = meisd_df.copy()
    meisd_filtered['emotion1'] = meisd_filtered['emotion1'].astype(str).str.lower().str.strip()
    meisd_filtered['intensity1'] = meisd_filtered['intensity1'].astype(float)

    # STEP 1: Filter by emotion compatibility (OPTIONAL)
    if remove_incompatible_emotions:
        print(f"\n{'=' * 70}")
        print("STEP 1: Filtering by emotion compatibility")
        print(f"{'=' * 70}")

        # Count samples to be removed by emotion
        removed_by_emotion = {}
        for emotion in incompatible_emotions:
            count = len(meisd_filtered[meisd_filtered['emotion1'] == emotion])
            removed_by_emotion[emotion] = count
            print(f" Removing '{emotion}': {count} samples")

        meisd_filtered = meisd_filtered[meisd_filtered['emotion1'].isin(compatible_emotions)]
        print(
            f"\n  After emotion filter: {len(meisd_filtered)} samples (removed {original_size - len(meisd_filtered)})")
    else:
        print(f"\nSKIPPING emotion filtering (keeping all emotions)")
        removed_by_emotion = {}

    # STEP 2: Filter by intensity
    print(f"\n{'=' * 70}")
    print("STEP 2: Filtering by intensity")
    print(f"{'=' * 70}")
    print(f"  Allowed intensities: {allowed_intensities}")

    # Count samples by intensity
    intensity_counts = meisd_filtered['intensity1'].value_counts().sort_index()
    print(f"\n  Current MEISD intensity distribution:")
    removed_by_intensity_count = 0
    for intensity, count in intensity_counts.items():
        if intensity in allowed_intensities:
            print(f"    ✓ KEEP intensity {intensity}: {count} samples")
        else:
            print(f"    ✗ REMOVE intensity {intensity}: {count} samples")
            removed_by_intensity_count += count

    before_intensity_filter = len(meisd_filtered)
    meisd_filtered = meisd_filtered[meisd_filtered['intensity1'].isin(allowed_intensities)]
    removed_by_intensity = before_intensity_filter - len(meisd_filtered)

    if removed_by_intensity > 0:
        print(f"\n  After intensity filter: {len(meisd_filtered)} samples (removed {removed_by_intensity})")
    else:
        print(f"\n  All intensities kept: {len(meisd_filtered)} samples")

    # STEP 3: Filter rare combinations
    print(f"\n{'=' * 70}")
    print(f"STEP 3: Filtering rare combinations (min_samples={min_samples})")
    print(f"{'=' * 70}")

    group_counts = meisd_filtered.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
    rare_combinations = group_counts[group_counts['count'] < min_samples]

    if len(rare_combinations) > 0:
        print(f"\n  Removing {len(rare_combinations)} rare combinations:")
        for _, row in rare_combinations.iterrows():
            print(f"    - {row['emotion1']} (intensity={row['intensity1']}): {row['count']} samples")

        # Filter out rare combinations
        keep_combinations = group_counts[group_counts['count'] >= min_samples]
        mask = meisd_filtered.set_index(['emotion1', 'intensity1']).index.isin(
            keep_combinations.set_index(['emotion1', 'intensity1']).index
        )
        before_rarity_filter = len(meisd_filtered)
        meisd_filtered = meisd_filtered[mask].reset_index(drop=True)
        removed_by_rarity = before_rarity_filter - len(meisd_filtered)

        print(f"\n  After rarity filter: {len(meisd_filtered)} samples (removed {removed_by_rarity})")
    else:
        print(f"\n  ✓ No rare combinations found (all have ≥{min_samples} samples)")
        removed_by_rarity = 0

    # FINAL SUMMARY
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Original MEISD size: {original_size}")
    print(f"  Filtered MEISD size: {len(meisd_filtered)}")
    print(
        f"  Total removed: {original_size - len(meisd_filtered)} ({(original_size - len(meisd_filtered)) / original_size * 100:.1f}%)")
    print(f"\n  Breakdown:")
    print(f"    - Removed by emotion: {sum(removed_by_emotion.values())} samples")
    print(f"    - Removed by intensity: {removed_by_intensity} samples")
    print(f"    - Removed by rarity: {removed_by_rarity} samples")

    # Final distribution
    print(f"\n{'=' * 70}")
    print("FINAL DISTRIBUTION (sorted by count)")
    print(f"{'=' * 70}")
    final_counts = meisd_filtered.groupby(['emotion1', 'intensity1']).size().reset_index(name='count')
    final_counts = final_counts.sort_values('count', ascending=False)
    print(final_counts.to_string(index=False))

    # Generate detailed report
    report = {
        'original_size': original_size,
        'filtered_size': len(meisd_filtered),
        'removed_total': original_size - len(meisd_filtered),
        'removed_by_emotion': removed_by_emotion,
        'removed_by_intensity': removed_by_intensity,
        'removed_by_rarity': removed_by_rarity,
        'compatible_emotions': list(compatible_emotions) if remove_incompatible_emotions else list(meisd_emotions),
        'incompatible_emotions': list(incompatible_emotions) if remove_incompatible_emotions else [],
        'final_combinations': final_counts.to_dict('records')
    }

    return meisd_filtered, report

# ------------------------------------------------------------
# === MAIN EXECUTION ===
# ------------------------------------------------------------
if __name__ == "__main__":
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent.parent.parent
    ESCONV_PATH = PROJECT_DIR / "data" / "ESConv_DA_ready.csv"
    MEISD_PATH = PROJECT_DIR / "data" / "MEISD_DA_ready.csv"
    LLAMA_PATH = PROJECT_DIR / "model" / "llama-2-7b-chat.Q5_K_M.gguf"
    OUTPUT_DIR = BASE_DIR / "./outputs"

    LLAMA_PATH = str(LLAMA_PATH)

    # Validate paths
    for path, name in [(ESCONV_PATH, "ESConv"), (MEISD_PATH, "MEISD")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download NLTK data if needed
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        print("NLTK WordNet not available - synonym replacement will be skipped")

    print("\n" + "="*60)
    print("=== MULTILABEL EMOTION-AWARE AUGMENTATION ===")
    print("="*60)

    print("\n[1/4] Processing ESConv data...")
    esconv_processor = MultilabelESConvProcessor(ESCONV_PATH)
    esconv_data = esconv_processor.load_data()

    # Filter ESConv (remove rare combinations)
    esconv_data_filtered, esconv_removed = filter_rare_combinations(
        esconv_data,
        min_samples=20,
        group_cols=['emotion1', 'intensity1']
    )
    esconv_processor.esconv_data = esconv_data_filtered

    print("\n[2/4] Loading and filtering MEISD data...")
    meisd_raw = pd.read_csv(MEISD_PATH)

    meisd_filtered, meisd_report = filter_meisd_for_esconv_compatibility(
        meisd_raw,
        esconv_data_filtered,
        min_samples=5,  # ← próg rzadkich kombinacji
        allowed_intensities=[1.0, 2.0, 3.0] # ← tylko 1.0 i 2.0 (jak w ESConv)
    )

    # Save filtered MEISD
    meisd_filtered_path = OUTPUT_DIR / "MEISD_filtered_for_ESConv.csv"
    meisd_filtered.to_csv(meisd_filtered_path, index=False, encoding='utf-8')
    print(f"\nFiltered MEISD saved to: {meisd_filtered_path}")

    # Save detailed report
    report_path = OUTPUT_DIR / "meisd_filtering_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MEISD FILTERING REPORT FOR ESCONV COMPATIBILITY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Original size: {meisd_report['original_size']}\n")
        f.write(f"Filtered size: {meisd_report['filtered_size']}\n")
        f.write(
            f"Removed: {meisd_report['removed_total']} ({meisd_report['removed_total'] / meisd_report['original_size'] * 100:.1f}%)\n\n")
        f.write(f"Removed by emotion: {sum(meisd_report['removed_by_emotion'].values())}\n")
        for emotion, count in meisd_report['removed_by_emotion'].items():
            f.write(f"  - {emotion}: {count}\n")
        f.write(f"\nRemoved by intensity: {meisd_report['removed_by_intensity']}\n")
        f.write(f"Removed by rarity: {meisd_report['removed_by_rarity']}\n\n")
        f.write(f"Compatible emotions: {meisd_report['compatible_emotions']}\n")
        f.write(f"Incompatible emotions: {meisd_report['incompatible_emotions']}\n")

    print(f"Detailed report saved to: {report_path}")

    print("\n[3/4] Analyzing ESConv patterns...")
    esconv_processor.analyze_multilabel_patterns(
        save_analysis=True,
        output_dir=OUTPUT_DIR
    )

    if hasattr(esconv_processor, 'emotion_intensifiers'):
        print("\nEmotion-specific intensifiers extracted from ESConv:")
        for emotion, intensifiers in list(esconv_processor.emotion_intensifiers.items())[:3]:
            print(f"  {emotion}: {', '.join(intensifiers[:5])}")

    print("\n[4/4] Setting up augmenter with FILTERED MEISD...")
    # Use filtered MEISD instead of raw
    augmenter = MultilabelMEISDAugmenter(
        meisd_filtered_path,  # ← używamy przefiltrowanego MEISD
        esconv_processor,
        LLAMA_PATH
    )
    augmenter.setup()

    df_aug = augmenter.augment_multilabel(num_samples=3, mode='mixed', save_details=True)

    print("\n[3/3] Balancing + Expanding dataset...")
    balanced_expanded_df = augmenter.balance_and_expand_2d(
        esconv_processor.esconv_data,
        mode='mixed',
        expand_percent=40    )

    output_path = os.path.join(OUTPUT_DIR, "ESConv_balanced_expanded_2D.csv")
    balanced_expanded_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Balanced & expanded dataset saved to: {output_path}")
    summary = summarize_augmentation_quality(df_aug, "mixed",
                                             save_path=os.path.join(OUTPUT_DIR, "augmentation_summary.csv"))

    df_aug.to_csv(os.path.join(OUTPUT_DIR, "MEISD_augmented.csv"), index=False, encoding='utf-8')

    # Zapis podsumowania jakości
    summary = summarize_augmentation_quality(
        df_aug,
        "mixed",
        save_path=os.path.join(OUTPUT_DIR, "augmentation_summary.csv")
    )

    print("\n=== Augmentation Summary ===")
    print(summary)