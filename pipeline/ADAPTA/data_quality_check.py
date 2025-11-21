"""
===========================================================
DATA QUALITY CHECK - Po feature engineering
===========================================================
Sprawdza jakość wygenerowanych cech przed modelowaniem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def check_data_quality(parquet_path: str):
    """Comprehensive data quality report"""

    print("="*70)
    print("📊 DATA QUALITY CHECK REPORT")
    print("="*70)

    # Load data
    df = pd.read_parquet(parquet_path)
    print(f"\n✅ Loaded {len(df)} records from {parquet_path}")

    # 1. Basic info
    print("\n" + "="*70)
    print("1️⃣ BASIC INFORMATION")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.dtypes.value_counts())

    # 2. Missing values
    print("\n" + "="*70)
    print("2️⃣ MISSING VALUES")
    print("="*70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)

    if len(missing_df) > 0:
        print("⚠️ Columns with missing values:")
        print(missing_df)
    else:
        print("✅ No missing values!")

    # 3. Target variable distribution
    print("\n" + "="*70)
    print("3️⃣ TARGET VARIABLE: delta_intensity")
    print("="*70)
    print(df['delta_intensity'].describe())

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    df['delta_intensity'].hist(bins=30, edgecolor='black')
    plt.xlabel('Delta Intensity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta Intensity')

    plt.subplot(1, 3, 2)
    df['initial_intensity'].hist(bins=10, alpha=0.7, label='Initial', edgecolor='black')
    df['final_intensity'].hist(bins=10, alpha=0.7, label='Final', edgecolor='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Initial vs Final Intensity')

    plt.subplot(1, 3, 3)
    df.groupby('emotion_type')['delta_intensity'].mean().plot(kind='bar')
    plt.xlabel('Emotion Type')
    plt.ylabel('Mean Delta')
    plt.title('Average Improvement by Emotion')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('data_quality_delta.png', dpi=150)
    print("📈 Saved: data_quality_delta.png")
    plt.close()

    # 4. Strategy distribution
    print("\n" + "="*70)
    print("4️⃣ STRATEGY DISTRIBUTION")
    print("="*70)
    strategy_counts = df['strategy'].value_counts()
    print(strategy_counts)

    if strategy_counts.min() < 20:
        print(f"\n⚠️ WARNING: Some strategies have very few samples (<20):")
        print(strategy_counts[strategy_counts < 20])
        print("Consider combining rare strategies or using stratified sampling")

    # 5. Feature correlations with target
    print("\n" + "="*70)
    print("5️⃣ FEATURE CORRELATIONS WITH TARGET")
    print("="*70)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in
                    ['conversation_id', 'turn_id', 'initial_intensity',
                     'final_intensity', 'delta_intensity']]

    if feature_cols:
        correlations = df[feature_cols + ['delta_intensity']].corr()['delta_intensity'].drop('delta_intensity')
        top_corr = correlations.abs().sort_values(ascending=False).head(15)

        print("Top 15 features most correlated with delta_intensity:")
        print(top_corr)

        plt.figure(figsize=(10, 6))
        top_corr.plot(kind='barh')
        plt.xlabel('Absolute Correlation with Delta Intensity')
        plt.title('Top Features by Correlation')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=150)
        print("📈 Saved: feature_correlations.png")
        plt.close()

    # 6. Check for outliers
    print("\n" + "="*70)
    print("6️⃣ OUTLIER DETECTION")
    print("="*70)

    Q1 = df['delta_intensity'].quantile(0.25)
    Q3 = df['delta_intensity'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['delta_intensity'] < Q1 - 1.5*IQR) |
                  (df['delta_intensity'] > Q3 + 1.5*IQR)]

    print(f"Outliers in delta_intensity: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    if len(outliers) > 0:
        print("\nOutlier examples:")
        print(outliers[['strategy', 'emotion_type', 'delta_intensity']].head(10))

    # 7. Feature value ranges
    print("\n" + "="*70)
    print("7️⃣ FEATURE VALUE RANGES")
    print("="*70)

    # Check sentiment features
    sent_cols = [c for c in df.columns if c.startswith('sent_')]
    if sent_cols:
        print("\nSentiment features:")
        print(df[sent_cols].describe())

    # Check emotion features
    emo_cols = [c for c in df.columns if c.startswith('emo_')]
    if emo_cols:
        print("\nEmotion features:")
        print(df[emo_cols].describe())

    # Check linguistic features
    ling_cols = [c for c in df.columns if c.startswith('ling_')]
    if ling_cols:
        print("\nLinguistic features (sample):")
        print(df[ling_cols[:5]].describe())

    # 8. Class balance for classification
    print("\n" + "="*70)
    print("8️⃣ CLASS BALANCE (for strategy classification)")
    print("="*70)

    class_balance = df['strategy'].value_counts(normalize=True) * 100
    imbalance_ratio = class_balance.max() / class_balance.min()

    print(f"Most common strategy: {class_balance.index[0]} ({class_balance.iloc[0]:.1f}%)")
    print(f"Least common strategy: {class_balance.index[-1]} ({class_balance.iloc[-1]:.1f}%)")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 10:
        print("\n⚠️ HIGH CLASS IMBALANCE detected!")
        print("Recommendations:")
        print("  - Use stratified sampling")
        print("  - Apply SMOTE or class weights")
        print("  - Consider combining rare classes")

    # 9. Sample data
    print("\n" + "="*70)
    print("9️⃣ SAMPLE RECORDS")
    print("="*70)
    print("\nFirst 3 records:")
    display_cols = ['strategy', 'emotion_type', 'delta_intensity',
                    'sent_label', 'sent_score', 'ling_word_count']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head(3))

    # 10. Final recommendations
    print("\n" + "="*70)
    print("🎯 RECOMMENDATIONS")
    print("="*70)

    recommendations = []

    if len(missing_df) > 0:
        recommendations.append("✓ Handle missing values (imputation or removal)")

    if imbalance_ratio > 10:
        recommendations.append("✓ Address class imbalance")

    if len(outliers) / len(df) > 0.05:
        recommendations.append("✓ Consider outlier treatment")

    if len(feature_cols) > 100:
        recommendations.append("✓ Consider feature selection/dimensionality reduction")

    if not recommendations:
        recommendations.append("✅ Data looks good! Ready for modeling.")

    for rec in recommendations:
        print(f"  {rec}")

    print("\n" + "="*70)
    print("✅ QUALITY CHECK COMPLETE")
    print("="*70)

    return df

if __name__ == "__main__":
    # Run quality check
    df = check_data_quality("esconv_enriched_features.parquet")

    print("\n💡 Next steps:")
    print("  1. Review the generated plots (data_quality_*.png)")
    print("  2. Address any warnings/issues identified above")
    print("  3. Proceed to PHASE 2: ML Modeling")