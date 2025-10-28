import pandas as pd
import numpy as np
import re
from collections import Counter

# === 🔧 Podaj ścieżki do plików ===
meisd_path = "C:/Users/juwieczo/DataspellProjects/meisd_project/data/MEISD_DA_ready.csv"
esconv_path = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/data_preparation/balanced datasets/esconv_both_parts.csv"
output_path = "domain_comparison.csv"

# === 🔧 Pomocnicze funkcje ===
def read_any(path):
    return pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)

def find_text_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["text", "utterance", "content", "message", "conversation"]):
            return c
    return df.columns[0]

def find_label_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["intensity", "label", "emotion", "max_intensity"]):
            return c
    return None

def tokenize_words(text):
    return re.findall(r"\w+", str(text).lower())

def basic_stats(df, name):
    text_col = find_text_col(df)
    label_col = find_label_col(df)

    texts = df[text_col].dropna().astype(str).tolist()
    lengths = [len(tokenize_words(t)) for t in texts]
    vocab = set([w for t in texts for w in tokenize_words(t)])
    ttr = len(vocab) / np.sum(lengths) if np.sum(lengths) > 0 else 0.0
    pronouns = ["i", "me", "my", "we", "us", "our", "you", "your"]
    pronoun_freq = Counter([w for t in texts for w in tokenize_words(t) if w in pronouns])
    top_pronouns = ", ".join([p for p, _ in pronoun_freq.most_common(3)]) if pronoun_freq else "—"

    # Średnia liczba zdań w wypowiedzi
    sent_counts = [len(re.split(r'[.!?]+', t)) for t in texts]
    avg_sentences = np.mean(sent_counts) if sent_counts else 0.0

    # Class balance (jeśli istnieje)
    if label_col and label_col in df.columns:
        dist = df[label_col].value_counts(normalize=True).to_dict()
        balance = f"{dist.get('low', 0):.2f} / {dist.get('high', 0):.2f}"
    else:
        balance = "—"

    return {
        "Dataset": name,
        "Samples": len(texts),
        "Avg length (words)": round(np.mean(lengths), 2),
        "Std length": round(np.std(lengths), 2),
        "Avg sentences per text": round(avg_sentences, 2),
        "Vocab size": len(vocab),
        "TTR": round(ttr, 4),
        "Top pronouns": top_pronouns,
        "Class balance (low/high)": balance
    }

# === 🔍 Wczytanie i analiza ===
df_meisd = read_any(meisd_path)
df_esconv = read_any(esconv_path)

s_meisd = basic_stats(df_meisd, "MEISD (source)")
s_esconv = basic_stats(df_esconv, "ESConv (target)")

# === 📊 Wyniki ===
df_out = pd.DataFrame([s_meisd, s_esconv])
print("\n=== Domain Comparison (MEISD vs ESConv) ===\n")
print(df_out.to_string(index=False))

# === 💾 Zapisz do CSV ===
df_out.to_csv(output_path, index=False)
print(f"\n[INFO] Zapisano do pliku: {output_path}")
