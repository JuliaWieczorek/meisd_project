import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Data from the confusion matrices
results_data = {
    'Classical_ESConv': {'TN': 218, 'FP': 61, 'FN': 28, 'TP': 183},
    'Classical_MEISD': {'TN': 351, 'FP': 123, 'FN': 124, 'TP': 350},
    'Mixed_ESConv': {'TN': 216, 'FP': 63, 'FN': 28, 'TP': 183},
    'Mixed_MEISD': {'TN': 386, 'FP': 88, 'FN': 71, 'TP': 403},
    'LLM_ESConv': {'TN': 211, 'FP': 68, 'FN': 20, 'TP': 191},
    'LLM_MEISD': {'TN': 419, 'FP': 55, 'FN': 57, 'TP': 417},
    'NLP_ESConv': {'TN': 197, 'FP': 82, 'FN': 36, 'TP': 175},
    'NLP_MEISD': {'TN': 287, 'FP': 187, 'FN': 136, 'TP': 338},
    'LLM_NLP_ESConv': {'TN': 203, 'FP': 76, 'FN': 24, 'TP': 187},
    'LLM_NLP_MEISD': {'TN': 311, 'FP': 163, 'FN': 103, 'TP': 371}
}

def calculate_metrics(tn, fp, fn, tp):
    """Calculate performance metrics from confusion matrix"""
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

def mcnemar_test(tn1, fp1, fn1, tp1, tn2, fp2, fn2, tp2):
    """Perform McNemar test between two classifiers"""
    # Create 2x2 contingency table for disagreements
    # Disagreement cases: classifier1 correct & classifier2 wrong, classifier1 wrong & classifier2 correct
    correct1 = tp1 + tn1
    correct2 = tp2 + tn2
    total = tp1 + tn1 + fp1 + fn1

    # For McNemar test, we need cases where classifiers disagree
    # This is an approximation since we only have confusion matrices
    # We'll use Chi-square test for independence instead

    # Create contingency table
    contingency = np.array([[tp1, fp1], [fn1, tn1]])
    contingency2 = np.array([[tp2, fp2], [fn2, tn2]])

    # Chi-square test
    chi2_1, p_val_1 = stats.chisquare([tp1, tn1, fp1, fn1])
    chi2_2, p_val_2 = stats.chisquare([tp2, tn2, fp2, fn2])

    return chi2_1, p_val_1, chi2_2, p_val_2

def compare_proportions(n1, success1, n2, success2):
    """Two-proportion z-test"""
    p1 = success1 / n1
    p2 = success2 / n2
    p_pool = (success1 + success2) / (n1 + n2)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_val, p1, p2

# Calculate metrics for all conditions
metrics_df = []
for condition, data in results_data.items():
    acc, prec, rec, f1 = calculate_metrics(data['TN'], data['FP'], data['FN'], data['TP'])
    method, dataset = condition.split('_', 1)
    metrics_df.append({
        'Method': method,
        'Dataset': dataset,
        'Condition': condition,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'TN': data['TN'],
        'FP': data['FP'],
        'FN': data['FN'],
        'TP': data['TP']
    })

metrics_df = pd.DataFrame(metrics_df)
print("=== PERFORMANCE METRICS SUMMARY ===")
print(metrics_df[['Method', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1']].round(4))

print("\n=== STATISTICAL COMPARISONS ===")

# Compare methods within each dataset
datasets = ['ESConv', 'MEISD']
methods = ['Classical', 'Mixed', 'LLM', 'NLP', 'LLM_NLP']

for dataset in datasets:
    print(f"\n--- {dataset} Dataset Comparisons ---")
    dataset_results = metrics_df[metrics_df['Dataset'] == dataset]

    # Pairwise comparisons of accuracy
    print(f"\nAccuracy Comparisons (Two-proportion z-tests):")
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            try:
                data1 = dataset_results[dataset_results['Method'] == method1].iloc[0]
                data2 = dataset_results[dataset_results['Method'] == method2].iloc[0]

                n = 4738  # Total samples
                success1 = data1['TP'] + data1['TN']
                success2 = data2['TP'] + data2['TN']

                z, p_val, p1, p2 = compare_proportions(n, success1, n, success2)

                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"{method1} vs {method2}: z={z:.3f}, p={p_val:.4f} {significance} (acc: {p1:.4f} vs {p2:.4f})")

            except Exception as e:
                print(f"Error comparing {method1} vs {method2}: {e}")

# Compare same method across datasets
print(f"\n=== CROSS-DATASET COMPARISONS ===")
for method in methods:
    try:
        esconv_data = metrics_df[(metrics_df['Method'] == method) & (metrics_df['Dataset'] == 'ESConv')].iloc[0]
        meisd_data = metrics_df[(metrics_df['Method'] == method) & (metrics_df['Dataset'] == 'MEISD')].iloc[0]

        n = 4738
        success1 = esconv_data['TP'] + esconv_data['TN']
        success2 = meisd_data['TP'] + meisd_data['TN']

        z, p_val, p1, p2 = compare_proportions(n, success1, n, success2)

        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{method} (ESConv vs MEISD): z={z:.3f}, p={p_val:.4f} {significance} (acc: {p1:.4f} vs {p2:.4f})")

    except Exception as e:
        print(f"Error comparing {method} across datasets: {e}")

# Effect sizes (Cohen's h for proportions)
print(f"\n=== EFFECT SIZES (Cohen's h) ===")
print("Effect size interpretation: small (0.2), medium (0.5), large (0.8)")

def cohens_h(p1, p2):
    """Calculate Cohen's h for effect size between two proportions"""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

# Calculate effect sizes for key comparisons
print("\nKey Method Comparisons on MEISD (largest performance differences):")
meisd_results = metrics_df[metrics_df['Dataset'] == 'MEISD']
llm_acc = meisd_results[meisd_results['Method'] == 'LLM']['Accuracy'].iloc[0]
nlp_acc = meisd_results[meisd_results['Method'] == 'NLP']['Accuracy'].iloc[0]
h_effect = cohens_h(llm_acc, nlp_acc)
print(f"LLM vs NLP: Cohen's h = {h_effect:.3f} ({'large' if abs(h_effect) > 0.8 else 'medium' if abs(h_effect) > 0.5 else 'small'} effect)")

# Confidence intervals for F1 scores (bootstrap approximation)
print(f"\n=== F1 SCORE CONFIDENCE INTERVALS (approximate) ===")
for condition, data in results_data.items():
    tp, tn, fp, fn = data['TP'], data['TN'], data['FP'], data['FN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Approximate 95% CI using Wilson score interval for F1
    n = tp + tn + fp + fn
    # This is a rough approximation - proper bootstrap would be better
    se_approx = np.sqrt(f1 * (1 - f1) / n)
    ci_lower = max(0, f1 - 1.96 * se_approx)
    ci_upper = min(1, f1 + 1.96 * se_approx)

    method, dataset = condition.split('_', 1)
    print(f"{method} {dataset}: F1 = {f1:.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")