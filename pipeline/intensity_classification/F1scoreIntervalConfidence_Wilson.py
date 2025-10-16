import numpy as np
from scipy import stats
import pandas as pd

def wilson_score_interval(successes, n, confidence=0.95):
    """
    Wilson score interval for a single proportion
    More accurate than normal approximation, especially for small samples or extreme proportions
    """
    if n == 0:
        return 0, 0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Critical value
    p = successes / n  # Sample proportion

    # Wilson score interval formula
    denominator = 1 + z**2 / n
    centre_adjusted = p + z**2 / (2 * n)
    margin_of_error = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

    lower_bound = (centre_adjusted - margin_of_error) / denominator
    upper_bound = (centre_adjusted + margin_of_error) / denominator

    return max(0, lower_bound), min(1, upper_bound)

def wilson_direct_f1_ci(tp, tn, fp, fn, confidence=0.95):
    """
    Wilson Direct Method for F1-score CI

    Treats F1-score as if it were a proportion and applies Wilson interval directly.
    This is a heuristic approach - F1 is not actually a proportion but this can
    provide reasonable bounds.

    Args:
        tp, tn, fp, fn: confusion matrix values
        confidence: confidence level (default 0.95)

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    # Calculate F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if f1 == 0:
        return 0, 0

    # Total sample size
    n = tp + tn + fp + fn

    # Treat F1 as a "success rate" and apply Wilson interval
    # This is the heuristic part - we're treating n*f1 as "successes"
    pseudo_successes = n * f1

    return wilson_score_interval(pseudo_successes, n, confidence)

def wilson_indirect_f1_ci(tp, tn, fp, fn, confidence=0.95):
    """
    Wilson Indirect Method for F1-score CI

    Calculates Wilson intervals for precision and recall separately,
    then uses these to derive bounds for F1-score using interval arithmetic.
    This is more theoretically sound than the direct method.

    Args:
        tp, tn, fp, fn: confusion matrix values
        confidence: confidence level (default 0.95)

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    # Calculate point estimates
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    if f1 == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        return 0, 0

    # Wilson intervals for precision and recall
    prec_lower, prec_upper = wilson_score_interval(tp, tp + fp, confidence)
    rec_lower, rec_upper = wilson_score_interval(tp, tp + fn, confidence)

    # Calculate F1 bounds using interval arithmetic
    # F1 = 2PR/(P+R), so we need to find min and max of this expression

    # Helper function to calculate F1 from precision and recall
    def calc_f1(p, r):
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)

    # Test all combinations of bounds to find F1 range
    # Due to the non-linear nature of F1, we need to check multiple combinations
    f1_candidates = []

    # Corner cases
    combinations = [
        (prec_lower, rec_lower),
        (prec_lower, rec_upper),
        (prec_upper, rec_lower),
        (prec_upper, rec_upper)
    ]

    for p, r in combinations:
        if p >= 0 and r >= 0:  # Valid precision and recall values
            f1_candidates.append(calc_f1(p, r))

    # Also check if the extrema occur at the boundaries
    # For F1 = 2PR/(P+R), taking partial derivatives:
    # ∂F1/∂P = 2R²/(P+R)² > 0 when R > 0
    # ∂F1/∂R = 2P²/(P+R)² > 0 when P > 0
    # So F1 is increasing in both P and R when both are positive

    if f1_candidates:
        f1_lower = min(f1_candidates)
        f1_upper = max(f1_candidates)
    else:
        f1_lower = f1_upper = 0

    return max(0, f1_lower), min(1, f1_upper)

def compare_wilson_methods(results_data, confidence=0.95):
    """Compare Wilson direct and indirect methods"""

    print(f"=== WILSON METHODS COMPARISON FOR F1-SCORE ({confidence*100}% CI) ===")
    print("Method\t\t\tF1-Score\tDirect Method CI\tIndirect Method CI\tCI Width Comparison")
    print("-" * 95)

    for condition, data in results_data.items():
        tp, tn, fp, fn = data['TP'], data['TN'], data['FP'], data['FN']

        # Calculate F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Wilson Direct Method
        direct_lower, direct_upper = wilson_direct_f1_ci(tp, tn, fp, fn, confidence)
        direct_width = direct_upper - direct_lower

        # Wilson Indirect Method
        indirect_lower, indirect_upper = wilson_indirect_f1_ci(tp, tn, fp, fn, confidence)
        indirect_width = indirect_upper - indirect_lower

        # Width comparison
        width_diff = indirect_width - direct_width
        width_comparison = f"Indirect {'wider' if width_diff > 0 else 'narrower'} by {abs(width_diff):.4f}"

        print(f"{condition:<15}\t{f1:.4f}\t\t[{direct_lower:.4f}, {direct_upper:.4f}]\t\t"
              f"[{indirect_lower:.4f}, {indirect_upper:.4f}]\t{width_comparison}")

# Your data
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

# Run comparison
compare_wilson_methods(results_data)

print("\n=== METHOD DETAILS ===")
print("\n1. WILSON DIRECT METHOD:")
print("   - Treats F1-score as if it were a simple proportion")
print("   - Applies Wilson score interval directly to F1")
print("   - Fast and simple, but theoretically questionable")
print("   - Good heuristic approximation")

print("\n2. WILSON INDIRECT METHOD:")
print("   - Calculates Wilson intervals for precision and recall separately")
print("   - Derives F1 bounds using interval arithmetic")
print("   - More theoretically sound")
print("   - Usually gives wider (more conservative) intervals")

print("\n3. RECOMMENDATIONS:")
print("   - For publication: Use Wilson Indirect Method (more rigorous)")
print("   - For quick analysis: Wilson Direct Method is acceptable")
print("   - Both methods are better than treating F1 as normal distribution")

print("\n=== EXAMPLE USAGE FOR YOUR PAPER ===")
# Example for one method
condition = 'LLM_MEISD'
data = results_data[condition]
tp, tn, fp, fn = data['TP'], data['TN'], data['FP'], data['FN']

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

indirect_lower, indirect_upper = wilson_indirect_f1_ci(tp, tn, fp, fn)

print(f"\nExample reporting:")
print(f"The LLM method achieved an F1-score of {f1:.3f} (95% CI: [{indirect_lower:.3f}, {indirect_upper:.3f}]) on the MEISD dataset.")