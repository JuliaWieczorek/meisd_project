"""
===========================================================
FAZA 2: PREDICTIVE MODELING
===========================================================
Cel: Budowa modeli ML do rekomendacji optymalnych strategii
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================
# PROBLEM 1: STRATEGY RECOMMENDATION
# ===============================================
# Input: stan emocjonalny, kontekst, historia
# Output: najlepsza strategia do użycia

class StrategyRecommender:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_features(self, df):
        """
            Przygotowanie cech do modelu klasyfikacyjnego
        """
        # Target: strategia
        y = self.label_encoder.fit_transform(df['strategy'])

        # Features: wybierz kolumny według prefiksu
        feature_cols = [c for c in df.columns if c.startswith(('sent_', 'emo_', 'ling_', 'conv_'))
                        and not isinstance(df[c].iloc[0], (list, dict))]

        # KLUCZOWE: wybierz tylko kolumny numeryczne
        numeric_features = df[feature_cols].select_dtypes(include=[np.number])

        # Dodaj one-hot encoding dla emotion_type
        emotion_dummies = pd.get_dummies(df['emotion_type'], prefix='emotion')

        X = pd.concat([
            numeric_features,
            emotion_dummies,
            df[['initial_intensity', 'turn_id']]
        ], axis=1)

        self.feature_names = X.columns.tolist()

        # Upewnij się, że wszystkie wartości są numeryczne
        X = X.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train(self, df, model_type='xgboost'):
        """
        Trening modelu rekomendacji strategii
        """
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=len(np.unique(y)),
                random_state=42
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )

        self.model.fit(X_train, y_train)

        # Ewaluacja
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"✅ Model trained: {model_type}")
        print(f"📊 Accuracy: {accuracy:.3f}")
        print(f"📊 F1-score (weighted): {f1:.3f}")

        # Feature importance
        self.plot_feature_importance()

        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        return accuracy, f1

    def plot_feature_importance(self, top_n=20):
        """Wizualizacja ważności cech"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top Features for Strategy Recommendation')
            plt.tight_layout()
            plt.savefig('feature_importance_strategy.png')
            plt.close()

            print("✅ Feature importance plot saved")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Strategy Recommendation - Confusion Matrix')
        plt.ylabel('True Strategy')
        plt.xlabel('Predicted Strategy')
        plt.tight_layout()
        plt.savefig('confusion_matrix_strategy.png')
        plt.close()

        print("✅ Confusion matrix saved")

    def recommend_strategy(self, features):
        """
        Rekomendacja strategii dla nowego przypadku
        """
        features_scaled = self.scaler.transform([features])
        strategy_id = self.model.predict(features_scaled)[0]
        strategy_name = self.label_encoder.inverse_transform([strategy_id])[0]

        # Prawdopodobieństwa wszystkich strategii
        probs = self.model.predict_proba(features_scaled)[0]
        top_3 = np.argsort(probs)[-3:][::-1]

        recommendations = {
            self.label_encoder.inverse_transform([i])[0]: probs[i]
            for i in top_3
        }

        return strategy_name, recommendations

# ===============================================
# PROBLEM 2: EFFECTIVENESS PREDICTION
# ===============================================
# Input: strategia + kontekst + emocja
# Output: przewidywana zmiana intensywności (Δ)

class EffectPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.strategy_encoder = LabelEncoder()

    def prepare_features(self, df):
        """Przygotowanie cech do regresji"""
        # Target: delta_intensity
        y = df['delta_intensity'].values

        # Features: wybierz kolumny według prefiksu
        feature_cols = [c for c in df.columns if c.startswith(('sent_', 'emo_', 'ling_', 'conv_'))
                        and not isinstance(df[c].iloc[0], (list, dict))]

        # KLUCZOWE: wybierz tylko kolumny numeryczne
        numeric_features = df[feature_cols].select_dtypes(include=[np.number])

        # Encode strategy
        strategy_encoded = self.strategy_encoder.fit_transform(df['strategy'])

        # One-hot encoding dla emotion_type
        emotion_dummies = pd.get_dummies(df['emotion_type'], prefix='emotion')

        X = pd.concat([
            numeric_features,
            pd.DataFrame({'strategy_encoded': strategy_encoded}),
            emotion_dummies,
            df[['initial_intensity', 'turn_id']]
        ], axis=1)

        # Upewnij się, że wszystkie wartości są numeryczne
        X = X.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train(self, df):
        """Trening modelu predykcji efektywności"""
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Użyjmy GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Ewaluacja
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = self.model.score(X_test, y_test)

        print(f"✅ Effectiveness predictor trained")
        print(f"📊 MAE: {mae:.3f}")
        print(f"📊 R² score: {r2:.3f}")

        # Scatter plot: predicted vs actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Δ intensity')
        plt.ylabel('Predicted Δ intensity')
        plt.title('Effect Prediction: Actual vs Predicted')
        plt.tight_layout()
        plt.savefig('effect_prediction_scatter.png')
        plt.close()

        print("✅ Prediction plot saved")

        return mae, r2

    def predict_effect(self, strategy, features):
        """
        Przewidywanie efektywności danej strategii
        """
        features_scaled = self.scaler.transform([features])
        predicted_delta = self.model.predict(features_scaled)[0]

        return predicted_delta

# ===============================================
# PROBLEM 3: SEQUENCE OPTIMIZATION
# ===============================================
# Który ciąg strategii jest najbardziej efektywny?

# ===============================================
# POPRAWIONA WERSJA - SequenceOptimizer z diagnostyką
# ===============================================

# ===============================================
# POPRAWIONA WERSJA - SequenceOptimizer
# Rekonstrukcja conversation_id
# ===============================================

# ===============================================
# POPRAWIONA WERSJA - SequenceOptimizer
# Rekonstrukcja conversation_id
# ===============================================

class SequenceOptimizer:
    def __init__(self, df):
        self.df = df.copy()  # Kopia, żeby nie modyfikować oryginału
        self.transitions = None
        self._reconstruct_conversation_ids()

    def _reconstruct_conversation_ids(self):
        """
        Rekonstrukcja conversation_id na podstawie turn_id
        Założenie: gdy turn_id spada lub resetuje się do niskiej wartości,
        to zaczyna się nowa rozmowa
        """
        print("\n🔧 Reconstructing conversation IDs...")

        # Sprawdź, czy conversation_id jest None/null
        if self.df['conversation_id'].isna().all() or (self.df['conversation_id'] == None).all():
            print("⚠️  All conversation_id values are None/null")

            # Metoda 1: Wykryj reset turn_id (gdy turn_id spada)
            conversation_breaks = [0]  # Pierwsza rozmowa zaczyna się od indeksu 0

            for i in range(1, len(self.df)):
                prev_turn = self.df.iloc[i-1]['turn_id']
                curr_turn = self.df.iloc[i]['turn_id']

                # Jeśli turn_id spadł lub zresetował się do 1, to nowa rozmowa
                if curr_turn <= prev_turn or curr_turn == 1:
                    conversation_breaks.append(i)

            # Stwórz nowe conversation_id
            conv_ids = []
            current_conv = 0
            break_pointer = 1  # Wskaźnik na kolejny break

            for i in range(len(self.df)):
                # Sprawdź, czy dotarliśmy do następnego breakpoint
                if break_pointer < len(conversation_breaks) and i == conversation_breaks[break_pointer]:
                    current_conv += 1
                    break_pointer += 1
                conv_ids.append(f"conv_{current_conv}")

            self.df['conversation_id'] = conv_ids

            print(f"✅ Reconstructed {current_conv + 1} conversations")

            # Statystyki
            conv_sizes = self.df.groupby('conversation_id').size()
            print(f"📊 Conversation sizes - min: {conv_sizes.min()}, max: {conv_sizes.max()}, mean: {conv_sizes.mean():.1f}")
            print(f"📊 Conversations with 2+ turns: {(conv_sizes >= 2).sum()}")
        else:
            print(f"✅ Using existing conversation_id ({self.df['conversation_id'].nunique()} conversations)")

    def analyze_strategy_transitions(self):
        """
        Analiza efektywności sekwencji strategii
        """
        print("\n🔍 Analyzing Strategy Transitions...")

        conversations = self.df.groupby('conversation_id')

        transition_effects = []
        conversations_with_transitions = 0

        for conv_id, group in conversations:
            # Pomiń rozmowy z tylko 1 turą
            if len(group) < 2:
                continue

            # Sortuj według turn_id
            group_sorted = group.sort_values('turn_id')

            strategies = group_sorted['strategy'].tolist()
            deltas = group_sorted['delta_intensity'].tolist()

            # Twórz przejścia
            conv_had_transitions = False
            for i in range(len(strategies) - 1):
                if i+1 < len(deltas) and deltas[i+1] is not None and not pd.isna(deltas[i+1]):
                    transition_effects.append({
                        'from_strategy': strategies[i],
                        'to_strategy': strategies[i+1],
                        'effect': deltas[i+1],
                        'conversation_id': conv_id
                    })
                    conv_had_transitions = True

            if conv_had_transitions:
                conversations_with_transitions += 1

        print(f"\n📊 Analysis Results:")
        print(f"Total conversations: {len(conversations)}")
        print(f"Conversations with 2+ turns: {sum(1 for _, g in conversations if len(g) >= 2)}")
        print(f"Conversations with transitions: {conversations_with_transitions}")
        print(f"Total transitions found: {len(transition_effects)}")

        if not transition_effects:
            print("\n⚠️  No valid strategy transitions found!")
            print("\nSample data (first conversation):")
            first_conv = self.df.groupby('conversation_id').first()
            print(self.df[self.df['conversation_id'] == self.df['conversation_id'].iloc[0]].head(10))
            return pd.DataFrame()

        self.transitions = pd.DataFrame(transition_effects)

        # Najlepsze przejścia
        best_transitions = (
            self.transitions.groupby(['from_strategy', 'to_strategy'])['effect']
            .agg(['mean', 'count', 'std'])
            .sort_values('mean', ascending=False)
            .head(20)
        )

        print("\n🔍 Top 20 Most Effective Strategy Transitions:")
        print("=" * 80)
        print(f"{'From Strategy':<35} {'To Strategy':<35} {'Mean Δ':>8}")
        print("=" * 80)
        for (from_s, to_s), row in best_transitions.iterrows():
            print(f"{from_s:<35} {to_s:<35} {row['mean']:>8.3f}")
        print("=" * 80)

        # Najgorsze przejścia
        worst_transitions = (
            self.transitions.groupby(['from_strategy', 'to_strategy'])['effect']
            .agg(['mean', 'count'])
            .sort_values('mean', ascending=True)
            .head(10)
        )

        print("\n⚠️  Top 10 Least Effective Strategy Transitions:")
        print("=" * 80)
        for (from_s, to_s), row in worst_transitions.iterrows():
            print(f"{from_s:<35} {to_s:<35} {row['mean']:>8.3f}")
        print("=" * 80)

        print(f"\n✅ Analysis complete!")
        print(f"📊 Total unique strategy pairs: {len(best_transitions)}")

        return best_transitions

    def recommend_next_strategy(self, current_strategy, top_n=3):
        """
        Rekomendacja następnej strategii na podstawie historycznych sekwencji
        """
        if self.transitions is None or len(self.transitions) == 0:
            print("⚠️  No transitions available. Run analyze_strategy_transitions() first.")
            return None

        relevant = self.transitions[self.transitions['from_strategy'] == current_strategy]

        if len(relevant) == 0:
            print(f"⚠️  No historical data for strategy: {current_strategy}")
            return None

        next_strategies = (
            relevant.groupby('to_strategy')['effect']
            .agg(['mean', 'count'])
            .sort_values('mean', ascending=False)
            .head(top_n)
        )

        print(f"\n💡 Top {top_n} recommended strategies after '{current_strategy}':")
        print("=" * 60)
        for strategy, row in next_strategies.iterrows():
            print(f"  {strategy:<40} Δ={row['mean']:>6.3f} (n={int(row['count'])})")
        print("=" * 60)

        return next_strategies['mean'].to_dict()


# ===============================================
# MAIN EXECUTION
# ===============================================
if __name__ == "__main__":
    # Załaduj wzbogacony dataset
    df = pd.read_parquet('esconv_enriched_features.parquet')

    print("="*60)
    print("🤖 MODEL 1: STRATEGY RECOMMENDER")
    print("="*60)
    recommender = StrategyRecommender()
    recommender.train(df, model_type='xgboost')

    print("\n" + "="*60)
    print("📈 MODEL 2: EFFECTIVENESS PREDICTOR")
    print("="*60)
    predictor = EffectPredictor()
    predictor.train(df)

    print("\n" + "="*60)
    print("🔄 MODEL 3: SEQUENCE OPTIMIZER")
    print("="*60)
    optimizer = SequenceOptimizer(df)
    best_transitions = optimizer.analyze_strategy_transitions()

    print("\n✅ All models trained successfully!")
    print("📦 Models ready for deployment")