import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
csv_path = "C:/Users/juwieczo/DataspellProjects/meisd_project/pipeline/EMOTIA/EMOTIA-DA/outputs/multilabel_augmented_onehot.csv"  # <- podaj ścieżkę

# === LOAD DATA ===
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {len(df)} samples\n")

# === AUTO-DETECT EMOTION COLUMNS ===
emotion_cols = [c for c in df.columns if c.startswith("emotion__")]
intensity_cols = [c for c in df.columns if c.startswith("intensity__")]

emotions = [c.replace("emotion__", "") for c in emotion_cols]
print(f"Detected emotions: {emotions}\n")

# === SENTIMENT DISTRIBUTION ===
if "sentiment" in df.columns:
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts(), "\n")

# === EMOTION PRESENCE DISTRIBUTION ===
emotion_counts = df[emotion_cols].sum().sort_values(ascending=False)
print("Emotion presence counts:")
print(emotion_counts)

# Multi-label analysis
df["num_emotions"] = df[emotion_cols].sum(axis=1)
multi_ratio = (df["num_emotions"] > 1).mean() * 100
print(f"\nAverage number of emotions per sample: {df['num_emotions'].mean():.2f}")
print(f"Samples with >1 emotion (multi-label candidates): {multi_ratio:.2f}%")

# === INTENSITY DISTRIBUTION PER EMOTION ===
print("\nEmotion × Intensity distribution:")
intensity_summary = {}

for emo in emotions:
    emo_mask = df[f"emotion__{emo}"] == 1
    counts = df.loc[emo_mask, f"intensity__{emo}"].value_counts().sort_index()
    intensity_summary[emo] = counts.to_dict()

intensity_df = pd.DataFrame(intensity_summary).fillna(0).astype(int)
print(intensity_df)

# === TOTAL INTENSITY COUNTS (all emotions) ===
all_intensities = []
for emo in emotions:
    mask = df[f"emotion__{emo}"] == 1
    all_intensities.extend(df.loc[mask, f"intensity__{emo}"].values)
all_intensity_counts = pd.Series(all_intensities).value_counts().sort_index()

print("\nOverall intensity counts (all emotions combined):")
print(all_intensity_counts)

# === SAVE REPORT ===
output_path = "emotion_intensity_report.csv"
intensity_df.to_csv(output_path)
print(f"\nReport saved to: {output_path}")

# === OPTIONAL: SIMPLE PLOTS ===
plt.figure(figsize=(8,4))
emotion_counts.plot(kind="bar", color="steelblue", title="Emotion presence counts")
plt.ylabel("Number of samples")
plt.show()

plt.figure(figsize=(6,4))
all_intensity_counts.plot(kind="bar", color="seagreen", title="Overall intensity distribution")
plt.ylabel("Count")
plt.show()
