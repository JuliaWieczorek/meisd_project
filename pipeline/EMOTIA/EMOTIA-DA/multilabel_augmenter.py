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
        print(f"\n📊 Saved pattern analysis to: {analysis_path}")

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
        print(f"📊 Saved TF-IDF keywords to: {keywords_path}")

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
        print(f"📊 Saved intensity summary to: {intensity_summary_path}")

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
        print(f"📄 Saved text report to: {report_path}")

        print("\n✅ All analysis reports saved!")
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
                self.llm = Llama(model_path=self.llama_path, n_ctx=2048, n_threads=6, verbose=False)
                print("LLaMA loaded successfully")
            except Exception as e:
                print(f"Failed to load LLaMA: {e}")
                self.llm = None

    def augment_multilabel(self, num_samples=10, mode='llm', save_details=True):
        """
        Main augmentation method

        Args:
            num_samples: Number of samples to augment
            mode: 'llm', 'eda', or 'mixed'
            save_details: Whether to save detailed transformation info

        Returns:
            DataFrame with augmented samples
        """
        samples = self.meisd_data.sample(min(num_samples, len(self.meisd_data)))
        augmented = []
        transformation_details = []

        for idx, row in tqdm(samples.iterrows(), total=len(samples), desc=f"Augmenting ({mode})"):
            # Extract original data
            text = str(row.get('Utterances') or row.get('conversation') or '')

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

            # Select primary emotion (emotion1) for style matching
            primary_emotion = emotions[0] if pd.notna(emotions[0]) else 'neutral'
            primary_intensity = intensities[0] if pd.notna(intensities[0]) else 2.0

            # Get ESConv style patterns
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

    def _llm_transform(self, text, emotions, intensities, sentiment, patterns):
        """Transform using LLM with ESConv style guidance"""
        if not self.llm:
            return text

        # Build emotion context
        emotion_list = []
        for e, i in zip(emotions, intensities):
            if pd.notna(e) and pd.notna(i):
                emotion_list.append(f"{e} (intensity {i})")
        emotion_context = ", ".join(emotion_list) if emotion_list else "neutral"

        # Get ESConv examples
        primary_emotion = emotions[0] if pd.notna(emotions[0]) else 'neutral'
        primary_intensity = intensities[0] if pd.notna(intensities[0]) else 2.0

        examples = self.esconv_processor.get_examples(
            primary_emotion, int(primary_intensity), sentiment, max_examples=2
        )
        examples_str = "\n".join([f"- {ex}" for ex in examples]) if examples else "No examples available"

        # Get keywords
        keywords = patterns.get('keywords', [])
        keywords_str = ", ".join(keywords[:10]) if keywords else "supportive, understanding"

        # Target length
        target_length = patterns.get('avg_length', 50)

        # Build prompt
        prompt = f"""You are rewriting a message to match the style of emotional support conversations.

Emotional context: {emotion_context}
Sentiment: {sentiment}

Examples of similar messages from real support conversations:
{examples_str}

Key characteristics to maintain:
- Use these types of words/phrases: {keywords_str}
- Keep length around {int(target_length)} words
- Maintain {sentiment} sentiment
- Express emotions: {emotion_context}

Original message:
"{text}"

Rewritten message (natural, conversational, supportive):"""

        try:
            output = self.llm(prompt, max_tokens=200, temperature=0.8, stop=["\n\nOriginal:", "Examples:"])
            result = output["choices"][0]["text"].strip()

            # Clean up
            result = re.sub(r'^["\-\*\s]+', '', result)
            result = re.sub(r'["\-\*\s]+$', '', result)

            return result if len(result) > 10 else text
        except Exception as e:
            print(f"LLM error: {e}")
            return text

    def _eda_transform(self, text, emotions, intensities, sentiment, patterns):
        """
        Enhanced EDA transformation using ESConv patterns:
        - Insert keywords from TF-IDF
        - Adjust length to match ESConv
        - Add personal pronouns
        - Synonym replacement (basic)
        """
        transformed = text
        words = transformed.split()

        # Get target characteristics
        keywords = patterns.get('keywords', [])
        target_length = patterns.get('avg_length', 50)
        starters = patterns.get('sentence_starters', [])

        # 1. Insert relevant keywords (20% chance per keyword)
        if keywords and len(words) < target_length * 1.2:
            num_inserts = max(1, int(len(words) * 0.2))
            selected_keywords = random.sample(keywords, min(num_inserts, len(keywords)))

            for kw in selected_keywords:
                if kw.lower() not in transformed.lower():
                    # Insert at random position
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, kw)

        transformed = ' '.join(words)

        # 2. Adjust sentence starter (30% chance)
        if starters and random.random() < 0.3:
            starter = random.choice(starters)
            # Replace first word if it's not already a good starter
            current_starter = words[0].lower() if words else ''
            if current_starter not in starters:
                words[0] = starter.capitalize()
                transformed = ' '.join(words)

        # 3. Add personal pronouns if missing (sentiment-specific)
        personal_pronouns = ['I', 'me', 'my', 'myself']
        has_personal = any(p.lower() in transformed.lower() for p in personal_pronouns)

        if not has_personal and random.random() < 0.4:
            pronoun = random.choice(personal_pronouns)
            transformed = f"{pronoun} feel like {transformed.lower()}"

        # 4. Adjust length
        words = transformed.split()
        if len(words) > target_length * 1.5:
            # Trim
            transformed = ' '.join(words[:int(target_length * 1.2)])
        elif len(words) < target_length * 0.5:
            # Expand slightly
            connectors = ["and", "because", "but"]
            transformed += f" {random.choice(connectors)} I'm not sure what to do."

        # 5. Context-aware synonym replacement using WordNet (avoid bias)
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

                # Skip if too short or is a pronoun/article
                if len(word_clean) < 3 or word_clean in ['the', 'and', 'but', 'for', 'not']:
                    continue

                # Get synonyms from WordNet
                synsets = wordnet.synsets(word_clean)
                if synsets:
                    # Get all lemmas from first synset (most common meaning)
                    lemmas = synsets[0].lemmas()
                    synonyms = [l.name().replace('_', ' ') for l in lemmas if l.name() != word_clean]

                    if synonyms:
                        synonym = random.choice(synonyms)
                        # Preserve original case
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        words[idx] = synonym
        except ImportError:
            # If WordNet not available, skip synonym replacement
            pass
        except Exception as e:
            # Silently fail on any error to avoid breaking augmentation
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


# ------------------------------------------------------------
# === MAIN EXECUTION ===
# ------------------------------------------------------------
if __name__ == "__main__":
    # Paths
    ESCONV_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\ESConv_DA_ready.csv"
    MEISD_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\MEISD_DA_ready.csv"
    LLAMA_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\models\llama-2-7b-chat.Q5_K_M.gguf"
    OUTPUT_DIR = r"C:\Users\juwieczo\DataspellProjects\meisd_project\pipeline\EMOTIA\EMOTIA-DA\outputs"

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

    # Step 1: Process ESConv
    print("\n[1/3] Processing ESConv data...")
    esconv_processor = MultilabelESConvProcessor(ESCONV_PATH)
    esconv_processor.load_data()

    esconv_processor.analyze_multilabel_patterns(save_analysis=True, output_dir=OUTPUT_DIR)