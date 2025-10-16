import os
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd
from itertools import combinations
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
# PROFESSIONAL SCIENTIFIC STYLE SETUP
# Remove seaborn style for cleaner, more professional look
plt.style.use('default')
# Set scientific color palette - grayscale and blue tones
scientific_colors = ['#2C3E50', '#34495E', '#5D6D7E', '#85929E', '#AEB6BF', '#D5DBDB']
blue_palette = ['#1B4F72', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Professional matplotlib parameters
plt.rcParams.update({
    'font.family': 'serif',           # Scientific serif font
    'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 11,                  # Standard scientific font size
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,            # Thinner axes
    'grid.linewidth': 0.5,            # Subtle grid
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.color': '#CCCCCC',
    'axes.axisbelow': True,           # Grid behind data
    'axes.spines.top': False,         # Remove top spine
    'axes.spines.right': False,       # Remove right spine
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,              # High resolution
    'savefig.bbox': 'tight',         # Tight bounding box
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,              # True Type fonts in PDF
})
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
    'LLM_NLP_MEISD': {'TN': 311, 'FP': 163, 'FN': 103, 'TP': 371},
    'Original_ESConv': {'TN': 180, 'FP': 99, 'FN': 52, 'TP': 159},
    'Original_MEISD': {'TN': 18, 'FP': 45, 'FN': 31, 'TP': 123}
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

# Fixed method parsing - handle multi-underscore method names
def parse_condition(condition):
    """Parse condition string to extract method and dataset"""
    if condition.startswith('LLM_NLP_'):
        return 'LLM_NLP', condition[8:]  # Remove 'LLM_NLP_' prefix
    else:
        parts = condition.split('_')
        if len(parts) >= 2:
            return parts[0], '_'.join(parts[1:])
        return condition, ''

# Calculate metrics for all conditions
metrics_df = []
for condition, data in results_data.items():
    acc, prec, rec, f1 = calculate_metrics(data['TN'], data['FP'], data['FN'], data['TP'])
    method, dataset = parse_condition(condition)
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

    pivot_df = metrics_df.pivot(index="Method", columns="Dataset", values="F1").reindex(['Original', 'Classical', 'Mixed', 'LLM', 'NLP', 'LLM_NLP'])

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Data from your results
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
    'LLM_NLP_MEISD': {'TN': 311, 'FP': 163, 'FN': 103, 'TP': 371},
    'Original_ESConv': {'TN': 180, 'FP': 99, 'FN': 52, 'TP': 159},
    'Original_MEISD': {'TN': 18, 'FP': 45, 'FN': 31, 'TP': 123}
}

def calculate_metrics(tn, fp, fn, tp):
    """Calculate performance metrics from confusion matrix"""
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

# Setup output directory
output_dir = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/intensity_classification/outputs/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Professional Performance Comparison
def create_performance_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]

        # Get data
        esconv_data = metrics_df[metrics_df['Dataset'] == 'ESConv']
        meisd_data = metrics_df[metrics_df['Dataset'] == 'MEISD']

        x = np.arange(len(esconv_data['Method']))
        width = 0.35

        # Professional colors
        bars1 = ax.bar(x - width/2, esconv_data[metric.replace('-Score', '')], width,
                       label='ESConv', color='#34495E', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, meisd_data[metric.replace('-Score', '')], width,
                       label='MEISD', color='#5D6D7E', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Augmentation Method')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(esconv_data['Method'], rotation=45, ha='right')

        # Professional legend
        if i == 0:  # Only show legend on first subplot
            ax.legend(frameon=True, fancybox=False, shadow=False,
                      framealpha=1, edgecolor='black', loc='upper right')

        # Add value labels (smaller, less intrusive)
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='#2C3E50')

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='#2C3E50')

        # Set y-axis limits for better readability
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    return fig

# 2. Professional F1 Score Analysis
def create_f1_focus_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate F1 data
    f1_data = []
    for condition, data in results_data.items():
        tp, tn, fp, fn = data['TP'], data['TN'], data['FP'], data['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        method, dataset = condition.split('_', 1)
        n = tp + tn + fp + fn
        se_approx = np.sqrt(f1 * (1 - f1) / n)
        ci_error = 1.96 * se_approx

        f1_data.append({
            'Method': method,
            'Dataset': dataset,
            'F1': f1,
            'CI_Error': ci_error
        })

    f1_df = pd.DataFrame(f1_data)

    # ESConv dataset
    esconv_f1 = f1_df[f1_df['Dataset'] == 'ESConv']
    bars1 = ax1.bar(range(len(esconv_f1)), esconv_f1['F1'],
                    yerr=esconv_f1['CI_Error'], capsize=3, alpha=0.8,
                    color='#34495E', edgecolor='black', linewidth=0.5,
                    error_kw={'linewidth': 1, 'capthick': 1})

    ax1.set_title('ESConv Dataset', fontweight='bold')
    ax1.set_xlabel('Augmentation Method')
    ax1.set_ylabel('F1-Score')
    ax1.set_xticks(range(len(esconv_f1)))
    ax1.set_xticklabels(esconv_f1['Method'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)

    # Add value labels
    for i, (bar, f1_val) in enumerate(zip(bars1, esconv_f1['F1'])):
        ax1.text(bar.get_x() + bar.get_width()/2., f1_val + 0.03,
                 f'{f1_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # MEISD dataset
    meisd_f1 = f1_df[f1_df['Dataset'] == 'MEISD']
    bars2 = ax2.bar(range(len(meisd_f1)), meisd_f1['F1'],
                    yerr=meisd_f1['CI_Error'], capsize=3, alpha=0.8,
                    color='#5D6D7E', edgecolor='black', linewidth=0.5,
                    error_kw={'linewidth': 1, 'capthick': 1})

    ax2.set_title('MEISD Dataset', fontweight='bold')
    ax2.set_xlabel('Augmentation Method')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(range(len(meisd_f1)))
    ax2.set_xticklabels(meisd_f1['Method'], rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)

    # Highlight best performer
    best_idx = meisd_f1['F1'].idxmax()
    best_bar_idx = list(meisd_f1.index).index(best_idx)
    bars2[best_bar_idx].set_color('#2C3E50')

    # Add value labels
    for i, (bar, f1_val) in enumerate(zip(bars2, meisd_f1['F1'])):
        color = 'white' if i == best_bar_idx else '#2C3E50'
        ax2.text(bar.get_x() + bar.get_width()/2., f1_val + 0.03,
                 f'{f1_val:.3f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=9, color=color)

    plt.tight_layout()
    return fig

# 3. Professional Heatmap
def create_performance_heatmap():
    # Create pivot table
    heatmap_data = metrics_df.pivot_table(
        index='Method',
        columns='Dataset',
        values=['Accuracy', 'Precision', 'Recall', 'F1']
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        data = heatmap_data[metric]

        # Professional grayscale colormap
        im = ax.imshow(data, cmap='Greys', aspect='auto', vmin=data.min().min(), vmax=data.max().max())

        # Add text annotations
        for j in range(len(data.index)):
            for k in range(len(data.columns)):
                text = ax.text(k, j, f'{data.iloc[j, k]:.3f}',
                               ha='center', va='center', fontweight='bold', fontsize=10)

        ax.set_title(f'{metric}', fontweight='bold')
        ax.set_xticks(range(len(data.columns)))
        ax.set_yticks(range(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Method')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig

# 4. Professional Transfer Learning Plot
def create_transfer_learning_plot():
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = metrics_df['Method'].unique()
    x_pos = np.arange(len(methods))

    esconv_f1 = []
    meisd_f1 = []

    for method in methods:
        esconv_score = metrics_df[(metrics_df['Method'] == method) &
                                  (metrics_df['Dataset'] == 'ESConv')]['F1'].iloc[0]
        meisd_score = metrics_df[(metrics_df['Method'] == method) &
                                 (metrics_df['Dataset'] == 'MEISD')]['F1'].iloc[0]
        esconv_f1.append(esconv_score)
        meisd_f1.append(meisd_score)

    # Professional line plot
    line1 = ax.plot(x_pos, esconv_f1, 'o-', linewidth=2.5, markersize=7,
                    label='ESConv (Target)', color='#2C3E50', markerfacecolor='white',
                    markeredgewidth=2, markeredgecolor='#2C3E50')
    line2 = ax.plot(x_pos, meisd_f1, 's-', linewidth=2.5, markersize=7,
                    label='MEISD (Source)', color='#5D6D7E', markerfacecolor='white',
                    markeredgewidth=2, markeredgecolor='#5D6D7E')

    # Add difference annotations
    for i, (method, e_f1, m_f1) in enumerate(zip(methods, esconv_f1, meisd_f1)):
        # Subtle connection lines
        ax.plot([i, i], [e_f1, m_f1], '--', color='#BDC3C7', alpha=0.5, linewidth=1)

        # Difference annotation
        diff = e_f1 - m_f1
        ax.text(i + 0.05, (e_f1 + m_f1)/2, f'{diff:+.3f}',
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#BDC3C7', alpha=0.8))

    ax.set_xlabel('Augmentation Method')
    ax.set_ylabel('F1-Score')
    ax.set_title('Cross-Dataset Performance Analysis', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')

    # Professional legend
    ax.legend(frameon=True, fancybox=False, shadow=False,
              framealpha=1, edgecolor='black', loc='lower right')

    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    return fig

# 5. Professional Confusion Matrices
def create_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    best_esconv = 'LLM_ESConv'
    best_meisd = 'LLM_MEISD'

    for i, (condition, title) in enumerate([(best_esconv, 'LLM (ESConv)'),
                                            (best_meisd, 'LLM (MEISD)')]):
        data = results_data[condition]
        confusion_matrix = np.array([[data['TP'], data['FN']],
                                     [data['FP'], data['TN']]])

        # Professional grayscale heatmap
        im = axes[i].imshow(confusion_matrix, cmap='Greys', aspect='auto')

        # Add text annotations
        for j in range(2):
            for k in range(2):
                text = axes[i].text(k, j, confusion_matrix[j, k],
                                    ha='center', va='center', fontweight='bold',
                                    fontsize=14, color='white' if confusion_matrix[j, k] > confusion_matrix.max()/2 else 'black')

        axes[i].set_title(f'{title}', fontweight='bold')
        axes[i].set_xlabel('Predicted Class')
        axes[i].set_ylabel('True Class')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['High', 'Low'])
        axes[i].set_yticklabels(['High', 'Low'])

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig

# 6. Professional Summary Plot
def create_summary_plot():
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = metrics_df['Method'].unique()
    esconv_f1 = []
    meisd_f1 = []

    for method in methods:
        esconv_score = metrics_df[(metrics_df['Method'] == method) &
                                  (metrics_df['Dataset'] == 'ESConv')]['F1'].iloc[0]
        meisd_score = metrics_df[(metrics_df['Method'] == method) &
                                 (metrics_df['Dataset'] == 'MEISD')]['F1'].iloc[0]
        esconv_f1.append(esconv_score)
        meisd_f1.append(meisd_score)

    x = np.arange(len(methods))
    width = 0.35

    # Professional bars
    bars1 = ax.bar(x - width/2, esconv_f1, width, label='ESConv',
                   color='#34495E', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, meisd_f1, width, label='MEISD',
                   color='#5D6D7E', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Highlight best performers
    max_esconv_idx = np.argmax(esconv_f1)
    max_meisd_idx = np.argmax(meisd_f1)

    bars1[max_esconv_idx].set_color('#2C3E50')
    bars2[max_meisd_idx].set_color('#2C3E50')

    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        color1 = 'white' if i == max_esconv_idx else '#2C3E50'
        color2 = 'white' if i == max_meisd_idx else '#2C3E50'

        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
                f'{height1:.3f}', ha='center', va='bottom',
                fontweight='bold' if i == max_esconv_idx else 'normal',
                fontsize=9, color=color1)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
                f'{height2:.3f}', ha='center', va='bottom',
                fontweight='bold' if i == max_meisd_idx else 'normal',
                fontsize=9, color=color2)

    ax.set_xlabel('Augmentation Method')
    ax.set_ylabel('F1-Score')
    ax.set_title('Performance Summary: F1-Scores Across Methods', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')

    # Professional legend
    ax.legend(frameon=True, fancybox=False, shadow=False,
              framealpha=1, edgecolor='black', loc='upper right')

    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    return fig

# Create and save all visualizations
print("Creating professional scientific visualizations...")

plots = [
    ("summary_performance", create_summary_plot, "Performance Summary"),
    ("performance_comparison", create_performance_comparison, "Performance Comparison"),
    ("f1_analysis", create_f1_focus_plot, "F1-Score Analysis"),
    ("performance_heatmap", create_performance_heatmap, "Performance Heatmap"),
    ("transfer_analysis", create_transfer_learning_plot, "Transfer Learning Analysis"),
    ("confusion_matrices", create_confusion_matrices, "Confusion Matrices")
]

for i, (filename, create_func, description) in enumerate(plots):
    try:
        print(f"{i+1}. {description}")
        fig = create_func()

        # Save in both formats with high quality
        fig.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        fig.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        plt.close(fig)
        print(f"   ✓ Saved: {filename}.png and {filename}.pdf")

    except Exception as e:
        print(f"   ✗ Error creating {description}: {str(e)}")

print(f"\nAll professional visualizations saved to '{output_dir}/'")
print("Ready for scientific publication!")


# data = {
#     "Method": ["Classical", "Mixed", "LLM", "NLP", "LLM-NLP"],
#     "BLEU": [0.194, 0.267, 0.309, 0.139, 0.182],
#     "CHRF": [0.348, 0.417, 0.492, 0.299, 0.336],
#     "SelfBLEU": [0.291, 0.253, 0.226, 0.187, 0.195],
#     "PPL": [38.7, 30.2, 25.3, 42.6, 35.1],
#     "F1_MEISD": [0.7392, 0.8352, 0.8816, 0.6767, 0.7361],
#     "F1_ESConv": [0.8044, 0.8009, 0.8128, 0.7479, 0.789]
# }


data = {
    "Method": ["Classical", "Mixed", "LLM", "NLP", "LLM-NLP"],
    "BLEU": [0.9798, 0.9936, 0.9989, 0.9687, 0.9994],
    "Self-BLEU": [0.8671, 0.7484, 0.7989, 0.6381, 0.8314],
    "RWORDS": [0.9938, 0.9948, 0.9950, 0.9950, 0.9933],
    "Misspelled_Words_Ratio": [0.0026, 0.0030, 0.0029, 0.0028, 0.0017],
    "Misspelled_Chars_Ratio": [0.0016, 0.0022, 0.0021, 0.0017, 0.0006],
    "PPL": [38.94, 35.30, 35.55, 47.53, 33.44],
    "F1_MEISD": [0.7392, 0.8352, 0.8816, 0.6767, 0.7361],
    "F1_ESConv": [0.8044, 0.8009, 0.8128, 0.7479, 0.789]
}


df = pd.DataFrame(data)

# Korelacje dla MEISD
print(df.corr(numeric_only=True)["F1_MEISD"].sort_values(ascending=False))

# Korelacje dla ESConv
print(df.corr(numeric_only=True)["F1_ESConv"].sort_values(ascending=False))
