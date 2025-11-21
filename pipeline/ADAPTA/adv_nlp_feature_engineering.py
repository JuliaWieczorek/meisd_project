"""
===========================================================
FAZA 1: ADVANCED FEATURE ENGINEERING (PRODUCTION-READY)
===========================================================
Ulepszona wersja z obsługą błędów i optymalizacją
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================================
# DEPENDENCY CHECKER
# ===============================================
class DependencyChecker:
    """Sprawdza dostępność bibliotek i modeli"""

    @staticmethod
    def check_and_install_spacy():
        """Sprawdza i instaluje spacy model"""
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spacy model 'en_core_web_sm' loaded successfully")
                return nlp
            except OSError:
                logger.warning("⚠️ spacy model not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spacy model installed and loaded")
                return nlp
        except Exception as e:
            logger.error(f"❌ Could not load spacy: {e}")
            return None

    @staticmethod
    def check_transformers():
        """Sprawdza dostępność transformers"""
        try:
            from transformers import pipeline
            logger.info("✅ transformers library available")
            return True
        except ImportError:
            logger.error("❌ transformers not installed. Run: pip install transformers")
            return False

    @staticmethod
    def check_sentence_transformers():
        """Sprawdza sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("✅ sentence-transformers available")
            return True
        except ImportError:
            logger.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            return False

# ===============================================
# 1️⃣ LIGHTWEIGHT SENTIMENT & EMOTION ANALYSIS
# ===============================================
class EmotionAnalyzer:
    """Optimized emotion analyzer with fallback mechanisms"""

    def __init__(self, use_gpu=False):
        self.use_transformers = DependencyChecker.check_transformers()
        self.sentiment_analyzer = None
        self.emotion_detector = None

        if self.use_transformers:
            try:
                from transformers import pipeline
                device = 0 if use_gpu else -1

                # Sentiment (lighter model)
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device
                )

                # Emotion (optional - comment out if too slow)
                try:
                    self.emotion_detector = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        top_k=None,
                        device=device
                    )
                    logger.info("✅ Emotion detector loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Emotion detector not available: {e}")
                    self.emotion_detector = None

                logger.info("✅ EmotionAnalyzer initialized with transformers")
            except Exception as e:
                logger.error(f"❌ Could not initialize transformers: {e}")
                self.use_transformers = False

    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment and emotions with fallback to TextBlob"""
        if not text or len(text.strip()) == 0:
            return self._get_default_analysis()

        # Truncate long texts
        text = text[:512]

        result = {}

        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text)[0]
                result['sentiment_label'] = sentiment['label']
                result['sentiment_score'] = sentiment['score']
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
                result.update(self._textblob_sentiment(text))
        else:
            result.update(self._textblob_sentiment(text))

        # Emotion detection
        if self.emotion_detector:
            try:
                emotions = self.emotion_detector(text)[0]
                result['emotions'] = {e['label']: e['score'] for e in emotions}
            except Exception as e:
                logger.debug(f"Emotion detection failed: {e}")
                result['emotions'] = self._get_default_emotions()
        else:
            result['emotions'] = self._get_default_emotions()

        return result

    def _textblob_sentiment(self, text: str) -> Dict:
        """Fallback sentiment using TextBlob"""
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                label = 'POSITIVE'
            elif polarity < -0.1:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'

            return {
                'sentiment_label': label,
                'sentiment_score': abs(polarity)
            }
        except:
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict:
        return {
            'sentiment_label': 'NEUTRAL',
            'sentiment_score': 0.5,
            'emotions': self._get_default_emotions()
        }

    def _get_default_emotions(self) -> Dict:
        return {
            'joy': 0.0, 'sadness': 0.0, 'anger': 0.0,
            'fear': 0.0, 'surprise': 0.0, 'disgust': 0.0
        }

# ===============================================
# 2️⃣ LINGUISTIC FEATURES (SPACY or FALLBACK)
# ===============================================
class LinguisticFeatureExtractor:
    """Extract linguistic features with spacy or basic fallback"""

    def __init__(self):
        self.nlp = DependencyChecker.check_and_install_spacy()
        self.use_spacy = self.nlp is not None

        if not self.use_spacy:
            logger.warning("⚠️ Using fallback linguistic features (spacy not available)")

    def extract_features(self, text: str) -> Dict:
        if self.use_spacy:
            return self._extract_with_spacy(text)
        else:
            return self._extract_basic(text)

    def _extract_with_spacy(self, text: str) -> Dict:
        """Full feature extraction with spacy"""
        doc = self.nlp(text)

        words = [t for t in doc if not t.is_punct]
        sentences = list(doc.sents)

        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(t.text) for t in words]) if words else 0,
            'avg_sentence_length': len(doc) / max(len(sentences), 1),

            'noun_count': len([t for t in doc if t.pos_ == 'NOUN']),
            'verb_count': len([t for t in doc if t.pos_ == 'VERB']),
            'adj_count': len([t for t in doc if t.pos_ == 'ADJ']),
            'adv_count': len([t for t in doc if t.pos_ == 'ADV']),
            'pronoun_count': len([t for t in doc if t.pos_ == 'PRON']),

            'first_person_count': len([t for t in doc if t.text.lower() in
                                       ['i', 'me', 'my', 'mine', 'we', 'us', 'our']]),
            'second_person_count': len([t for t in doc if t.text.lower() in
                                        ['you', 'your', 'yours']]),

            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'entity_count': len(doc.ents),
        }

        return features

    def _extract_basic(self, text: str) -> Dict:
        """Basic features without spacy"""
        words = text.split()
        sentences = text.split('.')

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),

            # Basic counts (approximate)
            'first_person_count': sum(text.lower().count(w) for w in ['i ', 'me ', 'my ', 'we ']),
            'second_person_count': sum(text.lower().count(w) for w in ['you ', 'your ']),

            # Placeholders for spacy-dependent features
            'noun_count': 0, 'verb_count': 0, 'adj_count': 0,
            'adv_count': 0, 'pronoun_count': 0, 'entity_count': 0,
        }

# ===============================================
# 3️⃣ CONVERSATIONAL CONTEXT
# ===============================================
class ConversationalFeatureExtractor:
    """Extract conversation-level features"""

    @staticmethod
    def extract_conversation_features(dialog_history: List, current_turn_idx: int) -> Dict:
        supporter_turns = [t for t in dialog_history[:current_turn_idx] if t.get('speaker') == 'supporter']
        seeker_turns = [t for t in dialog_history[:current_turn_idx] if t.get('speaker') == 'seeker']

        strategies = [t.get('annotation', {}).get('strategy') for t in supporter_turns]

        features = {
            'turn_number': current_turn_idx,
            'conversation_length_so_far': len(dialog_history[:current_turn_idx]),
            'supporter_turns_count': len(supporter_turns),
            'seeker_turns_count': len(seeker_turns),

            'last_strategy': strategies[-1] if strategies else 'None',
            'strategy_diversity': len(set(strategies)),
            'strategy_switches': sum(1 for i in range(1, len(strategies)) if strategies[i] != strategies[i-1]),

            'supporter_avg_length': np.mean([len(t.get('content', '')) for t in supporter_turns]) if supporter_turns else 0,
            'seeker_avg_length': np.mean([len(t.get('content', '')) for t in seeker_turns]) if seeker_turns else 0,
        }

        return features

# ===============================================
# 4️⃣ SEMANTIC EMBEDDINGS (OPTIONAL)
# ===============================================
class SemanticEmbedder:
    """Generate semantic embeddings - optional for speed"""

    def __init__(self, enabled=False):
        self.enabled = enabled and DependencyChecker.check_sentence_transformers()
        self.model = None

        if self.enabled:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Semantic embedder loaded")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embeddings model: {e}")
                self.enabled = False

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if self.enabled and self.model:
            try:
                return self.model.encode(text)
            except:
                return None
        return None

# ===============================================
# 5️⃣ MAIN PIPELINE WITH BATCH PROCESSING
# ===============================================
def build_feature_dataset(
        esconv_data_path: str,
        output_path: str,
        batch_size: int = 100,
        use_embeddings: bool = False,
        checkpoint_path: str = "checkpoint.parquet"
):
    """
    Main pipeline with optimizations:
    - Batch processing
    - Progress tracking
    - Checkpoint saving (resume capability)
    - Error handling
    """

    logger.info("🚀 Starting feature engineering pipeline...")

    # Initialize components
    emotion_analyzer = EmotionAnalyzer(use_gpu=False)
    linguistic_extractor = LinguisticFeatureExtractor()
    conv_extractor = ConversationalFeatureExtractor()
    embedder = SemanticEmbedder(enabled=use_embeddings)

    # Load data
    logger.info(f"📂 Loading data from {esconv_data_path}")
    with open(esconv_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"✅ Loaded {len(data)} conversations")

    # Check for checkpoint
    start_idx = 0
    enriched_records = []

    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        logger.info(f"📥 Found checkpoint, loading...")
        checkpoint_df = pd.read_parquet(checkpoint_file)
        enriched_records = checkpoint_df.to_dict('records')
        start_idx = len(enriched_records)
        logger.info(f"✅ Resumed from record {start_idx}")

    # Process conversations
    total_turns = sum(len([t for t in conv.get('dialog', []) if t['speaker'] == 'supporter'])
                      for conv in data)

    with tqdm(total=total_turns, initial=start_idx, desc="Processing turns") as pbar:
        batch_records = []

        for conv_idx, conv in enumerate(data):
            emotion_type = conv.get('emotion_type')
            scores = conv.get('survey_score', {}).get('seeker', {})
            init_int = int(scores.get('initial_emotion_intensity', 0))
            fin_int = int(scores.get('final_emotion_intensity', 0))
            delta = init_int - fin_int

            dialog = conv.get('dialog', [])

            for turn_idx, turn in enumerate(dialog):
                if turn['speaker'] != 'supporter':
                    continue

                # Skip if already processed
                if len(enriched_records) + len(batch_records) < start_idx:
                    continue

                try:
                    content = turn.get('content', '')
                    strategy = turn.get('annotation', {}).get('strategy', 'None')

                    # Extract features
                    emotion_features = emotion_analyzer.analyze_text(content)
                    ling_features = linguistic_extractor.extract_features(content)
                    conv_features = conv_extractor.extract_conversation_features(dialog, turn_idx)

                    # Build record
                    record = {
                        'conversation_id': conv.get('conv_id'),
                        'turn_id': turn_idx,
                        'emotion_type': emotion_type,
                        'initial_intensity': init_int,
                        'final_intensity': fin_int,
                        'delta_intensity': delta,
                        'strategy': strategy,
                        'content': content,
                    }

                    # Add sentiment/emotion features
                    record['sent_label'] = emotion_features.get('sentiment_label')
                    record['sent_score'] = emotion_features.get('sentiment_score')

                    for emo_name, emo_score in emotion_features.get('emotions', {}).items():
                        record[f'emo_{emo_name}'] = emo_score

                    # Add linguistic features
                    for feat_name, feat_val in ling_features.items():
                        record[f'ling_{feat_name}'] = feat_val

                    # Add conversational features
                    for feat_name, feat_val in conv_features.items():
                        if not isinstance(feat_val, (list, dict)):
                            record[f'conv_{feat_name}'] = feat_val

                    # Embedding (optional)
                    if use_embeddings:
                        embedding = embedder.get_embedding(content)
                        if embedding is not None:
                            record['embedding'] = embedding.tolist()

                    batch_records.append(record)

                except Exception as e:
                    logger.error(f"Error processing turn {turn_idx} in conv {conv.get('conv_id')}: {e}")
                    continue

                finally:
                    pbar.update(1)

                # Save checkpoint every batch_size records
                if len(batch_records) >= batch_size:
                    enriched_records.extend(batch_records)
                    df_checkpoint = pd.DataFrame(enriched_records)
                    df_checkpoint.to_parquet(checkpoint_path, index=False)
                    batch_records = []
                    logger.info(f"💾 Checkpoint saved ({len(enriched_records)} records)")

        # Save remaining batch
        if batch_records:
            enriched_records.extend(batch_records)

    # Final save
    df = pd.DataFrame(enriched_records)
    df.to_parquet(output_path, index=False)

    logger.info(f"\n✅ Feature engineering completed!")
    logger.info(f"📊 Total records: {len(df)}")
    logger.info(f"🔢 Total features: {len(df.columns)}")
    logger.info(f"💾 Saved to: {output_path}")

    # Cleanup checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("🧹 Checkpoint file removed")

    return df

# ===============================================
# EXECUTION
# ===============================================
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "C:/Users/juwieczo/DataspellProjects/meisd_project/data/ESConv.json"
    OUTPUT_PATH = "esconv_enriched_features.parquet"

    # Run pipeline
    # Set use_embeddings=True only if you have enough RAM and time
    enriched_df = build_feature_dataset(
        esconv_data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        batch_size=100,
        use_embeddings=False  # Start with False for speed
    )

    # Quick stats
    logger.info("\n" + "="*60)
    logger.info("📈 DATASET STATISTICS")
    logger.info("="*60)
    logger.info(f"Shape: {enriched_df.shape}")
    logger.info(f"Strategies: {enriched_df['strategy'].nunique()}")
    logger.info(f"Emotions: {enriched_df['emotion_type'].nunique()}")
    logger.info(f"Mean Δ intensity: {enriched_df['delta_intensity'].mean():.2f}")
    logger.info("="*60)