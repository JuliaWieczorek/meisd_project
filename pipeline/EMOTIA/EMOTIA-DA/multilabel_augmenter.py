# ============================================================
# === COMPLETE MULTILABEL AUGMENTATION + TRANSFER LEARNING ===
# ============================================================
import pandas as pd
import numpy as np
import random
import re
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# === FIXED ESCONV PROCESSOR (ALL 3 EMOTIONS) ===
# ============================================================
class MultilabelESConvProcessor:
    def __init__(self, esconv_path):
        self.esconv_path = esconv_path
        self.esconv_data = None
        self.style_patterns = {}
        self.all_emotions = set()
        self.emotion_intensifiers = {}
        self.intensity_aware_intensifiers = {}

    def load_data(self):
        print("Loading ESConv data...")
        self.esconv_data = pd.read_csv(self.esconv_path)
        print(f"ESConv data loaded: {len(self.esconv_data)} rows")
        return self.esconv_data

    def analyze_multilabel_patterns(self, save_analysis=True, output_dir=None):
        """✅ FIXED: Analyze ALL 3 emotions"""
        self.output_dir = output_dir

        if self.esconv_data is None:
            raise ValueError("ESConv data not loaded.")

        if len(self.esconv_data) == 0:
            print("\nWARNING: ESConv data is empty!")
            return

        text_cols = [c for c in self.esconv_data.columns
                     if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        sentiment_col = next((c for c in self.esconv_data.columns if 'sentiment' in c.lower()), None)
        if not sentiment_col:
            raise ValueError("No sentiment column found!")

        print(f"Using columns: text={text_col}, sentiment={sentiment_col}")

        df = self.esconv_data.copy()
        df['text'] = df[text_col].astype(str)
        df['sentiment'] = df[sentiment_col].astype(str).str.lower().str.strip()

        print("\n🔍 Extracting ALL emotions (emotion1, emotion2, emotion3)...")

        emotion_records = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing all emotions"):
            text = row['text']
            sentiment = row['sentiment']

            for i in [1, 2, 3]:
                emotion_col = f'emotion{i}'
                intensity_col = f'intensity{i}'

                if emotion_col in row.index and intensity_col in row.index:
                    emotion = row.get(emotion_col)
                    intensity = row.get(intensity_col)

                    if pd.notna(emotion) and str(emotion).strip() != '' and pd.notna(intensity):
                        try:
                            emotion = str(emotion).lower().strip()
                            intensity = float(intensity)

                            emotion_records.append({
                                'emotion': emotion,
                                'intensity': intensity,
                                'sentiment': sentiment,
                                'text': text
                            })

                            self.all_emotions.add(emotion)

                        except (ValueError, TypeError):
                            continue

        print(f"\n✅ Extracted {len(emotion_records)} emotion instances")
        print(f"✅ Found {len(self.all_emotions)} unique emotions: {sorted(self.all_emotions)}")

        if len(emotion_records) == 0:
            print("⚠️ WARNING: No valid emotions found!")
            return

        df_emotions = pd.DataFrame(emotion_records)

        print("\n🔍 Analyzing patterns by (emotion, intensity, sentiment)...")
        grouped = df_emotions.groupby(['emotion', 'intensity', 'sentiment'])

        analysis_data = []

        for (emotion, intensity, sentiment), group in tqdm(grouped, desc="Analyzing groups"):
            if len(group) < 5:
                continue

            texts = group['text'].dropna().tolist()
            if not texts:
                continue

            patterns = self._extract_patterns(texts)
            patterns['sample_count'] = len(texts)

            keywords = self._extract_keywords_tfidf(texts, top_n=20)
            patterns['keywords'] = keywords

            key = (emotion, int(intensity), sentiment)
            self.style_patterns[key] = patterns

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

        print(f"\n✅ Extracted patterns for {len(self.style_patterns)} combinations")

        if save_analysis and output_dir:
            self._save_analysis_reports(analysis_data, output_dir)

        # Extract intensifiers
        print("\n🔍 Extracting emotion-specific intensifiers...")
        self.emotion_intensifiers = self.extract_emotion_intensifiers(top_n=10)
        self.intensity_aware_intensifiers = self.extract_intensity_aware_intensifiers(top_n=10)

        print("\n✅ Pattern analysis complete!")

    def _extract_patterns(self, texts):
        patterns = {}
        patterns['avg_length'] = np.mean([len(t.split()) for t in texts])

        sentences = []
        for text in texts:
            sents = re.split(r'[.!?]+', text)
            sentences.extend([s.strip() for s in sents if s.strip()])

        starters = [s.split()[0].lower() for s in sentences if len(s.split()) > 0]
        starter_counts = Counter(starters)
        patterns['sentence_starters'] = [s for s, _ in starter_counts.most_common(15)]

        personal_pronouns = ['i', 'me', 'my', 'myself', 'we', 'us', 'our']
        pronoun_freq = {}
        for pronoun in personal_pronouns:
            count = sum(text.lower().count(f' {pronoun} ') + text.lower().count(f'{pronoun} ')
                        for text in texts)
            pronoun_freq[pronoun] = count / len(texts)
        patterns['pronoun_freq'] = pronoun_freq

        patterns['question_ratio'] = sum(1 for t in texts if '?' in t) / len(texts)
        patterns['exclamation_ratio'] = sum(1 for t in texts if '!' in t) / len(texts)

        return patterns

    def _extract_keywords_tfidf(self, texts, top_n=20):
        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-top_n:]
            keywords = [feature_names[i] for i in top_indices]

            return keywords
        except Exception as e:
            return []

    def extract_emotion_intensifiers(self, top_n=10):
        """Extract emotion-specific intensifiers using TF-IDF"""
        if self.esconv_data is None:
            return {}

        print("\nExtracting emotion-specific intensifiers...")

        text_cols = [c for c in self.esconv_data.columns
                     if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])]
        text_col = text_cols[0] if text_cols else self.esconv_data.columns[0]

        df = self.esconv_data.copy()
        df['text'] = df[text_col].astype(str)

        intensifier_keywords = [
            'very', 'really', 'so', 'extremely', 'absolutely', 'completely',
            'totally', 'deeply', 'highly', 'incredibly', 'particularly',
            'especially', 'quite', 'rather', 'fairly', 'pretty', 'somewhat',
            'constantly', 'always', 'never', 'forever', 'endlessly',
            'utterly', 'purely', 'entirely', 'wholly', 'fully'
        ]

        emotion_intensifiers = {}

        # Group by each emotion column
        for i in [1, 2, 3]:
            emotion_col = f'emotion{i}'
            if emotion_col not in df.columns:
                continue

            df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
            grouped = df.groupby('emotion')

            for emotion, group in grouped:
                if len(group) < 10 or emotion == '' or pd.isna(emotion):
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
                        if emotion not in emotion_intensifiers:
                            emotion_intensifiers[emotion] = top_intensifiers
                        else:
                            emotion_intensifiers[emotion].extend(top_intensifiers)
                            emotion_intensifiers[emotion] = list(set(emotion_intensifiers[emotion]))[:top_n]

                except Exception as e:
                    continue

        print(f"Extracted intensifiers for {len(emotion_intensifiers)} emotions")
        return emotion_intensifiers

    def extract_intensity_aware_intensifiers(self, top_n=10):
        """Extract intensifiers grouped by (emotion, intensity)"""
        if self.esconv_data is None:
            return {}

        print("\nExtracting intensity-aware intensifiers...")

        text_col = next((c for c in self.esconv_data.columns
                         if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])),
                        self.esconv_data.columns[0])

        df = self.esconv_data.copy()
        df['text'] = df[text_col].astype(str)

        intensifier_keywords = [
            'very', 'really', 'so', 'extremely', 'absolutely', 'completely',
            'totally', 'deeply', 'highly', 'incredibly', 'particularly',
            'especially', 'quite', 'rather', 'fairly', 'pretty', 'somewhat',
            'constantly', 'always', 'never', 'forever', 'endlessly',
            'utterly', 'purely', 'entirely', 'wholly', 'fully', 'barely',
            'slightly', 'somewhat', 'moderately', 'considerably'
        ]

        intensity_aware = {}

        for i in [1, 2, 3]:
            emotion_col = f'emotion{i}'
            intensity_col = f'intensity{i}'

            if emotion_col not in df.columns or intensity_col not in df.columns:
                continue

            df['emotion'] = df[emotion_col].astype(str).str.lower().str.strip()
            df['intensity'] = df[intensity_col].astype(float)

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
                        intensity_aware[key] = top_intensifiers

                except Exception as e:
                    continue

        return intensity_aware

    def get_examples(self, emotion, intensity, sentiment, max_examples=3):
        """Get example texts for given emotion, intensity, sentiment"""
        if self.esconv_data is None:
            return []

        df = self.esconv_data.copy()
        text_col = next((c for c in df.columns
                         if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])),
                        df.columns[0])

        matches = []
        for i in [1, 2, 3]:
            emotion_col = f'emotion{i}'
            intensity_col = f'intensity{i}'
            sentiment_col = next((c for c in df.columns if 'sentiment' in c.lower()), None)

            if not all([emotion_col in df.columns, intensity_col in df.columns, sentiment_col]):
                continue

            mask = (
                    (df[emotion_col].astype(str).str.lower() == str(emotion).lower()) &
                    (df[intensity_col].astype(float) == float(intensity)) &
                    (df[sentiment_col].astype(str).str.lower() == str(sentiment).lower())
            )

            matches.extend(df[mask][text_col].tolist())

        if len(matches) == 0:
            return []

        sample_size = min(max_examples, len(matches))
        return random.sample(matches, sample_size)

    def _save_analysis_reports(self, analysis_data, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        df_analysis = pd.DataFrame(analysis_data)
        df_analysis = df_analysis.sort_values(['emotion', 'intensity', 'sentiment'])

        analysis_path = os.path.join(output_dir, 'esconv_pattern_analysis_ALL_EMOTIONS.csv')
        df_analysis.to_csv(analysis_path, index=False, encoding='utf-8')
        print(f"\n✅ Analysis saved: {analysis_path}")


# ============================================================
# === MULTILABEL AUGMENTER (TRANSFER LEARNING) ===
# ============================================================
class MultilabelMEISDAugmenter:
    """
    ✅ Enhanced augmenter with transfer learning from ESConv
    """

    def __init__(self, meisd_path, esconv_processor, llama_obj=None):
        self.meisd_path = meisd_path
        self.esconv_processor = esconv_processor
        self.meisd_data = None
        self.llm = llama_obj
        self.transformation_details = None

    def setup(self):
        """Load MEISD data"""
        if self.meisd_data is None:
            self.meisd_data = pd.read_csv(self.meisd_path)
            print(f"MEISD data loaded: {len(self.meisd_data)} rows")

        if not getattr(self.esconv_processor, 'style_patterns', None):
            print("Analyzing ESConv patterns...")
            self.esconv_processor.analyze_multilabel_patterns()

    def augment_multilabel(self, num_samples=10, mode='mixed', save_details=True, default_sentiment='negative'):
        """
        ✅ Main augmentation function - multi-emotion aware

        Args:
            num_samples: Number of samples to augment
            mode: 'llm', 'eda', or 'mixed'
            save_details: Save transformation details
            default_sentiment: Default sentiment if missing
        """
        if self.meisd_data is None:
            self.setup()

        n = min(num_samples, len(self.meisd_data))
        samples = self.meisd_data.sample(n)
        augmented_rows = []
        details = []

        for idx, row in tqdm(samples.iterrows(), total=len(samples), desc="Augmenting (multi-emotion)"):
            text_col = next((c for c in row.index if any(k in c.lower() for k in ['utterance', 'text', 'message', 'content'])), None)
            text = str(row.get('Utterances') or row.get('utterance') or row.get('conversation') or row.get(text_col) or '')

            # Extract emotion bundle (1-3 emotions)
            emotion_bundle = extract_multilabel_emotions(row)
            sentiment = row.get('sentiment', default_sentiment) if pd.notna(row.get('sentiment')) else default_sentiment

            if not emotion_bundle:
                emotion_bundle = [('neutral', 2.0)]

            # Merge ESConv patterns for bundle
            patterns = merge_patterns_for_bundle(self.esconv_processor, emotion_bundle, sentiment)

            # Transform
            if mode == 'llm' and self.llm:
                transformed = self._llm_transform_multi(text, emotion_bundle, sentiment, patterns)
            elif mode == 'eda':
                transformed = self._eda_transform_multi(text, emotion_bundle, sentiment, patterns)
            elif mode == 'mixed':
                if self.llm and random.random() < 0.65:
                    transformed = self._llm_transform_multi(text, emotion_bundle, sentiment, patterns)
                else:
                    transformed = self._eda_transform_multi(text, emotion_bundle, sentiment, patterns)
            else:
                transformed = text

            quality = self._calculate_quality(text, transformed, patterns)

            out_row = {
                'sentiment': sentiment,
                'emotion1': row.get('emotion1'),
                'intensity1': row.get('intensity1'),
                'emotion2': row.get('emotion2'),
                'intensity2': row.get('intensity2'),
                'emotion3': row.get('emotion3'),
                'intensity3': row.get('intensity3'),
                'original': text,
                'augmented': transformed,
                'quality': quality,
                'mode': mode
            }
            augmented_rows.append(out_row)

            if save_details:
                details.append({
                    'sample_idx': int(idx) if isinstance(idx, (int, np.integer)) else idx,
                    'original_len': len(text.split()),
                    'transformed_len': len(transformed.split()),
                    'bundle': emotion_bundle,
                    'quality': quality,
                    'mode': mode
                })

        df_out = pd.DataFrame(augmented_rows)
        if save_details:
            self.transformation_details = pd.DataFrame(details)

        return df_out

    def _llm_transform_multi(self, text, emotion_bundle, sentiment, patterns):
        """LLM-based transformation with multi-emotion prompt"""
        if not self.llm:
            return text

        emotion_ctx = ", ".join([f"{e} (intensity {int(i)})" for e, i in emotion_bundle])
        keywords = patterns.get('keywords', [])[:12]
        target_len = int(patterns.get('avg_length', 50))

        primary = emotion_bundle[0]
        examples = self.esconv_processor.get_examples(primary[0], int(primary[1]), sentiment, max_examples=2)
        examples_str = "\n".join([f"- {e}" for e in examples]) if examples else "No examples."

        prompt = f"""[INST]
You are an empathetic rewriting assistant. Rewrite the following message to convey these emotions naturally:

Emotions: {emotion_ctx}
Sentiment: {sentiment}
Target length: ~{target_len} words

Examples:
{examples_str}

Original: "{text}"
Rewritten: [/INST]
"""

        try:
            output = self.llm(prompt, max_tokens=300, temperature=0.8)
            res = ""
            if isinstance(output, dict):
                choices = output.get("choices", [])
                if choices:
                    res = choices[0].get("text", "").strip()
            elif isinstance(output, str):
                res = output.strip()

            if not res or len(res.split()) < 3:
                return text

            res = re.sub(r'^[\-\*\s"]+','', res)
            res = re.sub(r'[\-\*\s"]+$','', res)
            return res
        except Exception as e:
            print(f"LLM error: {e}")
            return text

    def _eda_transform_multi(self, text, emotion_bundle, sentiment, patterns):
        """Rule-based augmentation with multi-emotion awareness"""
        transformed = text
        words = transformed.split()
        keywords = patterns.get('keywords', [])
        starters = patterns.get('sentence_starters', [])
        intensifiers = patterns.get('intensifiers', [])

        # Insert intensifier based on max intensity
        max_intensity = max([int(i) for _, i in emotion_bundle])
        intensity_prob = {1: 0.2, 2: 0.45, 3: 0.75}
        prob = intensity_prob.get(int(max_intensity), 0.4)

        if intensifiers and random.random() < prob:
            intens = random.choice(intensifiers)
            inserted = False
            for i, w in enumerate(words):
                if w.lower() in ['feel', 'felt', 'feeling', 'feels'] and i < len(words)-1:
                    words.insert(i+1, intens)
                    inserted = True
                    break
            if not inserted and len(words) > 2:
                pos = random.randint(1, len(words)-1)
                words.insert(pos, intens)

        transformed = " ".join(words)

        # Insert keywords
        words = transformed.split()
        if keywords:
            to_insert = min(max(1, len(words)//12), 3)
            available = [k for k in keywords if k.lower() not in transformed.lower()]
            for kw in random.sample(available, min(len(available), to_insert)):
                pos = random.randint(0, len(words))
                words.insert(pos, kw)

        transformed = " ".join(words)

        # Adjust starter
        words = transformed.split()
        if starters and random.random() < 0.3:
            st = random.choice(starters)
            if words:
                words[0] = st.capitalize()
            transformed = " ".join(words)

        # Length adjustment
        target_len = int(patterns.get('avg_length', 50))
        if len(words) < target_len * 0.6:
            transformed += " I'm not sure what to do."
        elif len(words) > target_len * 1.6:
            transformed = " ".join(words[:int(target_len * 1.2)])

        # Add pronoun if missing
        if not any(p in transformed.lower() for p in ['i ', ' me ', 'my ', 'myself']):
            if random.random() < 0.35:
                transformed = "I feel " + transformed[0].lower() + transformed[1:] if transformed else "I feel concerned."

        return transformed

    def _calculate_quality(self, original, transformed, patterns):
        """Calculate augmentation quality score"""
        if original.strip().lower() == transformed.strip().lower():
            return 0.5

        score = 0.5

        # Length similarity
        target = patterns.get('avg_length', 50)
        diff = abs(len(transformed.split()) - target) / max(1, target)
        score -= min(0.3, diff * 0.3)

        # Keyword presence
        kw = patterns.get('keywords', [])[:5]
        if kw:
            matches = sum(1 for k in kw if k.lower() in transformed.lower())
            score += min(0.4, matches * 0.12)

        return float(max(0.0, min(1.0, score)))

    def balance_and_expand_multilabel(self, target_multiplier=1.0, mode="mixed"):
        """
        ✅ Balance dataset by multi-emotion combinations
        """
        if self.meisd_data is None:
            self.setup()

        df = self.meisd_data.copy()

        def multilabel_key(row):
            bundle = extract_multilabel_emotions(row)
            if not bundle:
                return ("neutral:2",)
            key = tuple(sorted([f"{e}:{int(i)}" for e, i in bundle]))
            return key

        df["emotion_bundle_key"] = df.apply(multilabel_key, axis=1)

        group_counts = df["emotion_bundle_key"].value_counts()
        max_count = group_counts.max()
        target_count = int(max_count * float(target_multiplier))

        print(f"[Balance] Max={max_count}, target={target_count}, groups={len(group_counts)}")

        balanced_frames = []

        for group_key, count in group_counts.items():
            group_df = df[df["emotion_bundle_key"] == group_key]

            if count == target_count:
                balanced_frames.append(group_df)
                continue

            if count > target_count:
                keep = group_df.sample(target_count, replace=False)
                balanced_frames.append(keep)
                continue

            # Augment
            needed = target_count - count
            print(f"  Augmenting {needed} for {group_key}")

            old_data = self.meisd_data
            self.meisd_data = group_df

            augmented = self.augment_multilabel(
                num_samples=min(needed, len(group_df)),
                mode=mode,
                save_details=False
            )

            if len(augmented) < needed:
                still_needed = needed - len(augmented)
                extra = self.augment_multilabel(
                    num_samples=still_needed,
                    mode=mode,
                    save_details=False
                )
                augmented = pd.concat([augmented, extra], ignore_index=True)

            self.meisd_data = old_data

            balanced_frames.append(pd.concat([group_df, augmented], ignore_index=True))

        out = pd.concat(balanced_frames, ignore_index=True)
        out.drop(columns=["emotion_bundle_key"], inplace=True)

        print(f"[Balance] Final size: {len(out)}")
        return out


# ============================================================
# === HELPER FUNCTIONS ===
# ============================================================
def extract_multilabel_emotions(row):
    """Extract emotion bundle (1-3 emotions) from row"""
    bundle = []
    for e_col, i_col in [("emotion1", "intensity1"), ("emotion2", "intensity2"), ("emotion3", "intensity3")]:
        e = row.get(e_col)
        i = row.get(i_col)
        if pd.notna(e) and e != '' and pd.notna(i):
            try:
                bundle.append((str(e).lower().strip(), float(i)))
            except:
                continue
    return bundle


def merge_patterns_for_bundle(esconv_processor, emotion_bundle, sentiment):
    """Merge ESConv patterns for emotion bundle"""
    merged = {
        "keywords": set(),
        "sentence_starters": set(),
        "avg_length": 0.0,
        "intensifiers": set()
    }
    counts = 0

    for (emotion, intensity) in emotion_bundle:
        key = (emotion, int(intensity), str(sentiment).lower())
        p = esconv_processor.style_patterns.get(key)
        if p:
            merged["keywords"].update(p.get("keywords", []))
            merged["sentence_starters"].update(p.get("sentence_starters", []))
            merged["avg_length"] += p.get("avg_length", 0.0)
            counts += 1

        # Intensifiers
        ia_key = (emotion, int(intensity))
        ia = getattr(esconv_processor, 'intensity_aware_intensifiers', {}).get(ia_key, [])
        if ia:
            merged["intensifiers"].update(ia)
        else:
            emo_only = getattr(esconv_processor, 'emotion_intensifiers', {}).get(emotion, [])
            merged["intensifiers"].update(emo_only)

    if counts > 0:
        merged["avg_length"] = merged["avg_length"] / counts
    else:
        merged["avg_length"] = 50.0

    return {
        "keywords": list(merged["keywords"]),
        "sentence_starters": list(merged["sentence_starters"]),
        "avg_length": merged["avg_length"],
        "intensifiers": list(merged["intensifiers"])
    }


def filter_meisd_for_esconv_compatibility_FIXED(
        meisd_df,
        esconv_processor,
        min_samples=15,
        allowed_intensities=[1.0, 2.0, 3.0],
        remove_incompatible_emotions=True
):
    """✅ FIXED: Filter MEISD for ALL 3 emotions"""
    print("\n" + "=" * 70)
    print("FILTERING MEISD FOR ESCONV COMPATIBILITY (ALL 3 EMOTIONS)")
    print("=" * 70)

    esconv_emotions = esconv_processor.all_emotions

    if len(esconv_emotions) == 0:
        print("\n⚠️ ERROR: ESConv processor has NO emotions!")
        raise ValueError("ESConv processor has no emotions analyzed")

    print(f"\n✅ ESConv emotions ({len(esconv_emotions)}): {sorted(esconv_emotions)}")

    meisd_emotions = set()
    for i in [1, 2, 3]:
        col = f'emotion{i}'
        if col in meisd_df.columns:
            emotions = meisd_df[col].dropna().astype(str).str.lower().str.strip()
            meisd_emotions.update(emotions[emotions != ''])

    print(f"✅ MEISD emotions ({len(meisd_emotions)}): {sorted(meisd_emotions)}")

    compatible_emotions = esconv_emotions & meisd_emotions
    incompatible_emotions = meisd_emotions - esconv_emotions

    print(f"\n✅ Compatible emotions ({len(compatible_emotions)}): {sorted(compatible_emotions)}")
    print(f"❌ Incompatible emotions ({len(incompatible_emotions)}): {sorted(incompatible_emotions)}")

    original_size = len(meisd_df)
    meisd_filtered = meisd_df.copy()

    removed_by_emotion = 0

    if remove_incompatible_emotions:
        print(f"\n{'=' * 70}")
        print("STEP 1: Filtering by emotion compatibility")
        print(f"{'=' * 70}")

        def has_compatible_emotion(row):
            for i in [1, 2, 3]:
                col = f'emotion{i}'
                if col in row.index:
                    emotion = row[col]
                    if pd.notna(emotion) and emotion != '':
                        emotion = str(emotion).lower().strip()
                        if emotion in compatible_emotions:
                            return True
            return False

        mask = meisd_filtered.apply(has_compatible_emotion, axis=1)
        removed_by_emotion = len(meisd_filtered) - mask.sum()
        meisd_filtered = meisd_filtered[mask]

        print(f"  Removed {removed_by_emotion} rows")
        print(f"  Remaining: {len(meisd_filtered)} rows")
    else:
        print(f"\nSKIPPING emotion filtering")

    print(f"\n{'=' * 70}")
    print("STEP 2: Filtering by intensity")
    print(f"{'=' * 70}")

    def has_valid_intensity(row):
        for i in [1, 2, 3]:
            col = f'intensity{i}'
            if col in row.index:
                intensity = row[col]
                if pd.notna(intensity):
                    try:
                        if float(intensity) not in allowed_intensities:
                            return False
                    except (ValueError, TypeError):
                        return False
        return True

    before_intensity = len(meisd_filtered)
    mask = meisd_filtered.apply(has_valid_intensity, axis=1)
    meisd_filtered = meisd_filtered[mask]
    removed_by_intensity = before_intensity - len(meisd_filtered)

    print(f"  Removed {removed_by_intensity} rows")
    print(f"  Remaining: {len(meisd_filtered)} rows")

    print(f"\n{'=' * 70}")
    print(f"STEP 3: Filtering rare combinations (min={min_samples})")
    print(f"{'=' * 70}")

    combination_counts = {}

    for idx, row in meisd_filtered.iterrows():
        for i in [1, 2, 3]:
            e_col = f'emotion{i}'
            i_col = f'intensity{i}'

            if e_col in row.index and i_col in row.index:
                emotion = row[e_col]
                intensity = row[i_col]

                if pd.notna(emotion) and emotion != '' and pd.notna(intensity):
                    emotion = str(emotion).lower().strip()
                    try:
                        intensity = float(intensity)
                        key = (emotion, intensity)
                        combination_counts[key] = combination_counts.get(key, 0) + 1
                    except (ValueError, TypeError):
                        continue

    rare_combinations = {k: v for k, v in combination_counts.items() if v < min_samples}

    if rare_combinations:
        print(f"  Found {len(rare_combinations)} rare combinations")

        def has_common_combination(row):
            for i in [1, 2, 3]:
                e_col = f'emotion{i}'
                i_col = f'intensity{i}'

                if e_col in row.index and i_col in row.index:
                    emotion = row[e_col]
                    intensity = row[i_col]

                    if pd.notna(emotion) and emotion != '' and pd.notna(intensity):
                        try:
                            emotion = str(emotion).lower().strip()
                            intensity = float(intensity)
                            key = (emotion, intensity)
                            if key not in rare_combinations:
                                return True
                        except (ValueError, TypeError):
                            continue
            return False

        before_rarity = len(meisd_filtered)
        mask = meisd_filtered.apply(has_common_combination, axis=1)
        meisd_filtered = meisd_filtered[mask]
        removed_by_rarity = before_rarity - len(meisd_filtered)

        print(f"  Removed {removed_by_rarity} rows")
        print(f"  Remaining: {len(meisd_filtered)} rows")
    else:
        print(f"  ✅ No rare combinations found")
        removed_by_rarity = 0

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Original: {original_size}")
    print(f"  Filtered: {len(meisd_filtered)}")
    print(f"  Removed: {original_size - len(meisd_filtered)} "
          f"({(original_size - len(meisd_filtered)) / original_size * 100:.1f}%)")

    report = {
        'original_size': original_size,
        'filtered_size': len(meisd_filtered),
        'removed_total': original_size - len(meisd_filtered),
        'removed_by_emotion': removed_by_emotion,
        'removed_by_intensity': removed_by_intensity,
        'removed_by_rarity': removed_by_rarity,
        'compatible_emotions': list(compatible_emotions),
        'incompatible_emotions': list(incompatible_emotions),
    }

    return meisd_filtered, report


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


# ============================================================
# === MAIN EXECUTION ===
# ============================================================
if __name__ == "__main__":
    from pathlib import Path

    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = BASE_DIR.parent.parent.parent
    ESCONV_PATH = PROJECT_DIR / "data" / "ESConv_DA_ready.csv"
    MEISD_PATH = PROJECT_DIR / "data" / "MEISD_DA_ready.csv"
    LLAMA_PATH = PROJECT_DIR / "models" / "llama-2-7b-chat.Q5_K_M.gguf"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # Validate
    for path, name in [(ESCONV_PATH, "ESConv"), (MEISD_PATH, "MEISD")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # NLTK data
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        print("NLTK WordNet not available")

    print("\n" + "="*70)
    print("=== MULTILABEL EMOTION-AWARE AUGMENTATION + TRANSFER LEARNING ===")
    print("="*70)

    # ========================================
    # STEP 1: Load ESConv
    # ========================================
    print("\n[1/5] Loading ESConv data...")
    esconv_processor = MultilabelESConvProcessor(ESCONV_PATH)
    esconv_processor.load_data()

    # ========================================
    # STEP 2: Analyze ESConv (ALL 3 emotions) - MUST BE FIRST!
    # ========================================
    print("\n[2/5] Analyzing ESConv patterns (ALL 3 emotions)...")
    esconv_processor.analyze_multilabel_patterns(
        save_analysis=True,
        output_dir=OUTPUT_DIR
    )

    print(f"\n✅ ESConv analysis complete!")
    print(f"   Unique emotions: {len(esconv_processor.all_emotions)}")
    print(f"   Patterns: {len(esconv_processor.style_patterns)}")
    print(f"   Emotions: {sorted(esconv_processor.all_emotions)}")

    if len(esconv_processor.all_emotions) == 0:
        raise ValueError("❌ No emotions found in ESConv!")

    # ========================================
    # STEP 3: Filter MEISD
    # ========================================
    print("\n[3/5] Filtering MEISD data (ALL 3 emotions)...")
    meisd_raw = pd.read_csv(MEISD_PATH)

    meisd_filtered, meisd_report = filter_meisd_for_esconv_compatibility_FIXED(
        meisd_raw,
        esconv_processor,
        min_samples=5,
        allowed_intensities=[1.0, 2.0, 3.0],
        remove_incompatible_emotions=True
    )

    # Save filtered MEISD
    meisd_filtered_path = OUTPUT_DIR / "MEISD_filtered_ALL_EMOTIONS.csv"
    meisd_filtered.to_csv(meisd_filtered_path, index=False, encoding='utf-8')
    print(f"\n✅ Filtered MEISD saved: {meisd_filtered_path}")

    # Save report
    report_path = OUTPUT_DIR / "meisd_filtering_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MEISD FILTERING REPORT (ALL 3 EMOTIONS)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Original size: {meisd_report['original_size']}\n")
        f.write(f"Filtered size: {meisd_report['filtered_size']}\n")
        f.write(f"Removed: {meisd_report['removed_total']} "
                f"({meisd_report['removed_total'] / meisd_report['original_size'] * 100:.1f}%)\n\n")
        f.write(f"Removed by emotion: {meisd_report['removed_by_emotion']}\n")
        f.write(f"Removed by intensity: {meisd_report['removed_by_intensity']}\n")
        f.write(f"Removed by rarity: {meisd_report['removed_by_rarity']}\n\n")
        f.write(f"Compatible emotions ({len(meisd_report['compatible_emotions'])}):\n")
        f.write(f"  {', '.join(sorted(meisd_report['compatible_emotions']))}\n\n")
        f.write(f"Incompatible emotions ({len(meisd_report['incompatible_emotions'])}):\n")
        f.write(f"  {', '.join(sorted(meisd_report['incompatible_emotions']))}\n")

    print(f"✅ Report saved: {report_path}")

    # ========================================
    # STEP 4: Setup augmenter with LLM
    # ========================================
    print("\n[4/5] Setting up augmenter with transfer learning...")

    # Try to load LLM
    llm = None
    if LLAMA_PATH.exists():
        try:
            from llama_cpp import Llama
            llm = Llama(
                model_path=str(LLAMA_PATH),
                n_ctx=2048,
                n_threads=8,
                temperature=0.8
            )
            print("✅ LLM loaded successfully")
        except Exception as e:
            print(f"⚠️ LLM not loaded: {e}")
            print("   Will use EDA-only augmentation")
    else:
        print(f"⚠️ LLM not found at {LLAMA_PATH}")
        print("   Will use EDA-only augmentation")

    # Setup augmenter
    augmenter = MultilabelMEISDAugmenter(
        meisd_filtered_path,
        esconv_processor,
        llama_obj=llm
    )
    augmenter.setup()

    # ========================================
    # STEP 5: Test augmentation
    # ========================================
    print("\n[5/5] Testing augmentation (3 samples)...")

    mode = 'mixed' if llm else 'eda'
    df_aug = augmenter.augment_multilabel(
        num_samples=3,
        mode='llm',
        save_details=True
    )

    # Save augmented samples
    aug_output_path = OUTPUT_DIR / "MEISD_augmented_TEST.csv"
    df_aug.to_csv(aug_output_path, index=False, encoding='utf-8')
    print(f"✅ Augmented samples saved: {aug_output_path}")

    # Save quality summary
    summary = summarize_augmentation_quality(
        df_aug,
        mode,
        save_path=OUTPUT_DIR / "augmentation_quality_summary.csv"
    )

    print("\n" + "="*70)
    print("=== AUGMENTATION SUMMARY ===")
    print("="*70)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ========================================
    # OPTIONAL: Balance and expand dataset
    # ========================================
    print("\n" + "="*70)
    print("Optional: Balance and expand dataset?")
    print("  Uncomment code below to run full balancing")
    print("="*70)

    # Uncomment to run:
    # print("\n[BONUS] Balancing and expanding dataset...")
    # balanced_df = augmenter.balance_and_expand_multilabel(
    #     target_multiplier=1.5,
    #     mode=mode
    # )
    # balanced_path = OUTPUT_DIR / "MEISD_balanced_expanded.csv"
    # balanced_df.to_csv(balanced_path, index=False, encoding='utf-8')
    # print(f"✅ Balanced dataset saved: {balanced_path}")

    print("\n" + "="*70)
    print("✅ ALL PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  📊 ESConv emotions analyzed: {len(esconv_processor.all_emotions)}")
    print(f"  📊 ESConv patterns extracted: {len(esconv_processor.style_patterns)}")
    print(f"  📊 MEISD filtered: {len(meisd_raw)} → {len(meisd_filtered)} rows")
    print(f"  📊 Augmented samples: {len(df_aug)}")
    print(f"  📊 Average quality: {summary['avg_quality']}")
    print(f"\n📁 Output files:")
    print(f"  - {meisd_filtered_path}")
    print(f"  - {aug_output_path}")
    print(f"  - {report_path}")
    print(f"  - {OUTPUT_DIR / 'esconv_pattern_analysis_ALL_EMOTIONS.csv'}")