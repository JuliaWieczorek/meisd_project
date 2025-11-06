"""
===========================================================
EMO-STRAT ANALYZER – PEŁNY RAPORT ANALIZY
===========================================================

Cel:
- Pełna analiza relacji między strategiami i emocjami w zbiorze ESConv
- Zapis wyników, wykresów, korelacji i sekwencji strategii do plików
- Generowanie raportu tekstowego z wnioskami

"""

import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

# ===============================================
# KONFIGURACJA ŚCIEŻEK
# ===============================================
DATA_PATH = "C:/Users/juwieczo/DataspellProjects/meisd_project/data/ESConv.json"
OUTPUT_EXCEL = "emo_strat_results_full.xlsx"
OUTPUT_SUMMARY = "emo_strat_summary_full.txt"
PLOT_DIR = "plots_full"
os.makedirs(PLOT_DIR, exist_ok=True)

# ===============================================
# 1️⃣ Wczytanie i przygotowanie danych
# ===============================================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for conv in data:
    emotion = conv.get("emotion_type")
    seeker_scores = conv.get("survey_score", {}).get("seeker", {})
    init_int = int(seeker_scores.get("initial_emotion_intensity", 0))
    fin_int = int(seeker_scores.get("final_emotion_intensity", 0))
    delta_int = init_int - fin_int

    for turn in conv.get("dialog", []):
        if turn["speaker"] == "supporter":
            strategy = turn["annotation"].get("strategy", "None")
            content = turn.get("content", "")
            records.append({
                "conversation_id": conv.get("conv_id", None),
                "emotion_type": emotion,
                "initial_intensity": init_int,
                "final_intensity": fin_int,
                "delta_intensity": delta_int,
                "strategy": strategy,
                "content": content
            })

df = pd.DataFrame(records)
print(f"Załadowano {len(df)} wypowiedzi wspierających z {len(data)} rozmów.")

# ===============================================
# 2️⃣ Statystyki ogólne
# ===============================================
emotion_counts = df["emotion_type"].value_counts()
strategy_counts = df["strategy"].value_counts()
mean_delta = df.groupby("strategy")["delta_intensity"].mean().sort_values(ascending=False)

# ===============================================
# 3️⃣ Wizualizacje podstawowe
# ===============================================
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="strategy", hue="emotion_type", order=df["strategy"].value_counts().index)
plt.xticks(rotation=75)
plt.title("Częstość użycia strategii względem emocji rozmówcy")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/strategies_by_emotion.png")
plt.close()

impact = df.groupby(["strategy", "emotion_type"])["delta_intensity"].mean().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
sns.heatmap(impact, annot=True, cmap="coolwarm", center=0)
plt.title("Wpływ strategii na zmianę intensywności emocji (średni Δ)")
plt.xlabel("Emocja rozmówcy")
plt.ylabel("Strategia wspierającego")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/impact_heatmap.png")
plt.close()

plt.figure(figsize=(10, 4))
mean_delta.plot(kind="bar", color="skyblue")
plt.title("Średnia poprawa emocji wg strategii (Δ intensywności)")
plt.ylabel("Średnia zmiana intensywności (init - final)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/mean_delta_bar.png")
plt.close()

# ===============================================
# 4️⃣ Analiza współwystępowania strategii i emocji
# ===============================================
cross_tab = pd.crosstab(df["strategy"], df["emotion_type"], normalize="columns") * 100
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Procentowy udział strategii w każdej emocji")
plt.xlabel("Emocja rozmówcy")
plt.ylabel("Strategia")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/strategy_emotion_heatmap.png")
plt.close()

# ===============================================
# 5️⃣ Sekwencje strategii – analiza przepływu (Sankey)
# ===============================================
sequence_data = []
for conv in data:
    seq = [turn["annotation"].get("strategy", "None") for turn in conv["dialog"] if turn["speaker"] == "supporter"]
    for i in range(len(seq) - 1):
        sequence_data.append((seq[i], seq[i + 1]))

seq_df = pd.DataFrame(sequence_data, columns=["prev_strategy", "next_strategy"])
seq_counts = seq_df.value_counts().reset_index(name="count")

unique_strategies = list(set(seq_df["prev_strategy"]) | set(seq_df["next_strategy"]))
strategy_to_id = {s: i for i, s in enumerate(unique_strategies)}

sources = [strategy_to_id[s] for s in seq_counts["prev_strategy"]]
targets = [strategy_to_id[t] for t in seq_counts["next_strategy"]]
values = seq_counts["count"].tolist()

fig = go.Figure(data=[go.Sankey(
    node=dict(pad=20, thickness=20, line=dict(color="black", width=0.5), label=unique_strategies),
    link=dict(source=sources, target=targets, value=values)
)])
fig.update_layout(title_text="Przepływ strategii w rozmowach (Sankey)", font_size=10)
fig.write_html(f"{PLOT_DIR}/strategy_flow_sankey.html")

# ===============================================
# 6️⃣ Analiza słów kluczowych wg emocji
# ===============================================
def get_top_keywords_per_emotion(df, n=10):
    vec = CountVectorizer(stop_words="english", max_features=2000)
    top_keywords = {}
    for emo in df["emotion_type"].unique():
        texts = df[df["emotion_type"] == emo]["content"]
        if len(texts) > 0:
            X = vec.fit_transform(texts)
            freqs = np.asarray(X.sum(axis=0)).ravel()
            top_idx = np.argsort(freqs)[::-1][:n]
            top_keywords[emo] = [vec.get_feature_names_out()[i] for i in top_idx]
    return top_keywords

keywords_by_emotion = get_top_keywords_per_emotion(df)

# ===============================================
# 7️⃣ Korelacja strategii i skuteczności emocjonalnej
# ===============================================
df_corr = df.copy()
strategy_effect = df_corr.groupby("strategy")["delta_intensity"].mean().to_dict()
df_corr["strategy_effect"] = df_corr["strategy"].map(strategy_effect)

plt.figure(figsize=(8,6))
sns.boxplot(data=df_corr, x="emotion_type", y="strategy_effect")
plt.title("Rozkład skuteczności strategii w zależności od emocji")
plt.ylabel("Średni efekt strategii (Δ intensywności)")
plt.xlabel("Emocja rozmówcy")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/strategy_effect_by_emotion.png")
plt.close()

# ==========================================================
# 8 PORÓWNANIE STRATEGII W EMOCJACH WYSOKIEJ VS NISKIEJ INTENSYWNOŚCI
# ==========================================================

# Ustalmy próg wysokiej intensywności (np. mediana wartości początkowych)
median_intensity = df["initial_intensity"].median()

df["intensity_level"] = np.where(df["initial_intensity"] >= median_intensity, "High", "Low")

# Obliczamy średni spadek intensywności (Δ) osobno dla obu grup
intensity_comparison = (
    df.groupby(["strategy", "intensity_level"])["delta_intensity"]
    .mean()
    .unstack(fill_value=0)
    .sort_values(by="High", ascending=False)
)

# Wykres porównawczy
plt.figure(figsize=(10, 6))
intensity_comparison.plot(kind="bar", figsize=(10, 6))
plt.title("Porównanie skuteczności strategii w emocjach wysokiej vs. niskiej intensywności")
plt.xlabel("Strategia")
plt.ylabel("Średni spadek intensywności emocji (Δ)")
plt.xticks(rotation=75)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/strategy_effect_high_vs_low.png")
plt.close()

# Interpretacja: różnica między skutecznością w emocjach silnych i słabych
intensity_comparison["difference"] = intensity_comparison["High"] - intensity_comparison["Low"]
top_diff = intensity_comparison["difference"].abs().sort_values(ascending=False).head(5)

# Zapis wniosków do raportu
with open(OUTPUT_SUMMARY, "a", encoding="utf-8") as f:
    f.write("\n=====================================================\n")
    f.write("🔎 ANALIZA: Wpływ poziomu intensywności emocji na skuteczność strategii\n")
    f.write("=====================================================\n\n")
    f.write(f"Średnia intensywność graniczna (mediana): {median_intensity}\n\n")

    f.write("Strategie, które najbardziej różnią się skutecznością między emocjami wysokiej i niskiej intensywności:\n")
    for strat, diff in top_diff.items():
        sign = "↑" if diff > 0 else "↓"
        f.write(f"  - {strat}: różnica {diff:.2f} ({'lepsza przy wysokich emocjach' if diff > 0 else 'lepsza przy niskich emocjach'}) {sign}\n")

    f.write("\n📊 Pełne wyniki w arkuszu: High_vs_Low_Intensity\n")
    f.write(f"📈 Wykres: {PLOT_DIR}/strategy_effect_high_vs_low.png\n")
    f.write("=====================================================\n")

print("✅ Analiza porównania strategii przy wysokich i niskich emocjach zakończona.")


# ===============================================
# 8️⃣ Zapis wyników do pliku Excel
# ===============================================
with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
    df.to_excel(writer, sheet_name="Dane_surowe", index=False)
    emotion_counts.to_excel(writer, sheet_name="Emocje")
    strategy_counts.to_excel(writer, sheet_name="Strategie")
    mean_delta.to_excel(writer, sheet_name="Delta_intensywnosci")
    cross_tab.to_excel(writer, sheet_name="Strategia_vs_Emocja")
    seq_counts.to_excel(writer, sheet_name="Sekwencje_strategii")
    kw_df = pd.DataFrame({
        "Emotion": list(keywords_by_emotion.keys()),
        "Top_Keywords": [", ".join(v) for v in keywords_by_emotion.values()]
    })
    kw_df.to_excel(writer, sheet_name="Slowa_kluczowe", index=False)
    intensity_comparison.to_excel(writer, sheet_name="High_vs_Low_Intensity")


print(f"✅ Wyniki zapisano do: {OUTPUT_EXCEL}")

# ===============================================
# 9️⃣ Generowanie automatycznych wniosków
# ===============================================
best_strat = mean_delta.idxmax()
worst_strat = mean_delta.idxmin()
dominant_by_emotion = df.groupby("emotion_type")["strategy"].agg(lambda x: x.value_counts().index[0])

# ===============================================
# 🔟 Raport tekstowy
# ===============================================
with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("=====================================================\n")
    f.write("EMO-STRAT ANALYSIS – PEŁNY RAPORT\n")
    f.write("=====================================================\n\n")

    f.write("📈 Najbardziej skuteczna strategia:\n")
    f.write(f"  • {best_strat} (średni spadek intensywności emocji: {mean_delta.max():.2f})\n\n")
    f.write("📉 Najmniej skuteczna strategia:\n")
    f.write(f"  • {worst_strat} (średni spadek intensywności emocji: {mean_delta.min():.2f})\n\n")

    f.write("🔥 Dominująca strategia w każdej emocji:\n")
    for emo, strat in dominant_by_emotion.items():
        f.write(f"  - {emo}: {strat}\n")

    f.write("\n🗝️ Najczęstsze słowa kluczowe wg emocji:\n")
    for emo, words in keywords_by_emotion.items():
        f.write(f"  - {emo}: {', '.join(words)}\n")

    f.write("\n📊 Dodatkowe pliki:\n")
    f.write(f"  - Dane i tabele: {OUTPUT_EXCEL}\n")
    f.write(f"  - Wykresy: folder {PLOT_DIR}/\n")
    f.write(f"  - Sankey diagram: {PLOT_DIR}/strategy_flow_sankey.html\n")
    f.write("\n=====================================================\n")

print(f"📝 Raport zapisano do: {OUTPUT_SUMMARY}")
print("Analiza zakończona pomyślnie ✅")
