import json
import csv
import math
import os
from collections import Counter

# --- Pomocnicze mapowanie emocji do sentymentu ---
SENTIMENT_MAP = {
    'joy': 'positive', 'love': 'positive', 'gratitude': 'positive', 'relief': 'positive', 'pride': 'positive',
    'anger': 'negative', 'fear': 'negative', 'sadness': 'negative', 'disgust': 'negative', 'anxiety': 'negative',
    'guilt': 'negative', 'shame': 'negative', 'neutral': 'neutral', 'acceptance': 'neutral', 'surprise': 'neutral'
}

# Sprawdzenie skali intensywności
def inspect_intensity_scale(json_path):
    """Analizuje rozkład intensywności emocji w pliku ESConv.json"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    init_values, final_values = [], []
    for convo in data:
        seeker = convo.get("survey_score", {}).get("seeker", {})
        if seeker.get("initial_emotion_intensity"):
            init_values.append(seeker.get("initial_emotion_intensity"))
        if seeker.get("final_emotion_intensity"):
            final_values.append(seeker.get("final_emotion_intensity"))

    all_vals = [v for v in init_values + final_values if v not in [None, ""]]
    counts = Counter(all_vals)

    print(f"\nLiczba wszystkich próbek: {len(all_vals)}")
    print(f"🔹 Unikalne wartości intensywności: {sorted(set(all_vals))}")

    print("\nRozkład intensywności:")
    for val, cnt in sorted(counts.items(), key=lambda x: int(x[0])):
        print(f"  wartość {val}: {cnt} wystąpień")

    try:
        nums = [int(v) for v in all_vals]
        min_v, max_v = min(nums), max(nums)
        print(f"\nZakres intensywności: {min_v}–{max_v}")
        if max_v <= 3:
            print("Wygląda na to, że skala już jest 1–3 — nie wymaga normalizacji.")
        elif max_v == 5:
            print("Skala 1–5 — zalecana normalizacja do 1–3 (1–2→1, 3→2, 4–5→3).")
        elif max_v == 4:
            print("Skala 1–4 — można przeskalować do 1–3 proporcjonalnie.")
        else:
            print("Nietypowa skala — sprawdź dane ręcznie.")
    except:
        print("Nie udało się sparsować wszystkich wartości jako liczb.")


# Automatyczna normalizacja do skali 1–3
def normalize_esconv_intensity(json_in_path, json_out_path):
    """Automatycznie normalizuje intensywności emocji (1–5 → 1–3) i zapisuje nowy plik JSON."""
    with open(json_in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_vals = []
    for convo in data:
        seeker = convo.get("survey_score", {}).get("seeker", {})
        for key in ["initial_emotion_intensity", "final_emotion_intensity"]:
            val = seeker.get(key)
            if val not in [None, ""]:
                try:
                    all_vals.append(int(val))
                except:
                    pass

    if not all_vals:
        print("Nie znaleziono wartości liczbowych intensywności.")
        return

    max_v = max(all_vals)
    print(f"Zakres intensywności w pliku: {min(all_vals)}–{max_v}")

    if max_v <= 3:
        print("Skala już 1–3 — nie wymaga normalizacji.")
        return

    def normalize_intensity(value):
        try:
            val = int(value)
            if val <= 2:
                return 1
            elif val == 3:
                return 2
            elif val >= 4:
                return 3
        except:
            return value
        return value

    # przeskalowanie
    for convo in data:
        seeker = convo.get("survey_score", {}).get("seeker", {})
        for key in ["initial_emotion_intensity", "final_emotion_intensity"]:
            if key in seeker and seeker[key] not in [None, ""]:
                seeker[key] = normalize_intensity(seeker[key])

    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Zapisano znormalizowany plik: {json_out_path}")


# 🔹 3️⃣ Pomocnicze funkcje konwersji
def dialog_to_text(dialog_turns):
    seeker_texts = [turn.get("content", "").strip() for turn in dialog_turns if turn.get("speaker") == "seeker"]
    return " ".join([t for t in seeker_texts if t])


def split_dialog(dialog):
    """Dzieli dialog na dwie połowy (początek i koniec rozmowy) — tylko wypowiedzi seekera"""
    seeker_turns = [turn for turn in dialog if turn.get("speaker") == "seeker"]
    total = len(seeker_turns)
    if total == 0:
        return [], []
    mid = math.ceil(total / 2)
    return seeker_turns[:mid], seeker_turns[mid:]

def emotion_to_sentiment(emo):
    if not emo:
        return 'neutral'
    return SENTIMENT_MAP.get(emo.lower().strip(), 'neutral')


# Konwersja JSON → CSV kompatybilny z DA
def convert_esconv_to_da_format(json_path, output_csv='esconv_da_ready.csv'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []

    for convo in data:
        emotion = convo.get("emotion_type", "").strip()
        init_int = convo.get("survey_score", {}).get("seeker", {}).get("initial_emotion_intensity", "")
        fin_int = convo.get("survey_score", {}).get("seeker", {}).get("final_emotion_intensity", "")
        problem = convo.get("problem_type", "")
        situation = convo.get("situation", "")
        dialog = convo.get("dialog", [])

        first_half, second_half = split_dialog(dialog)
        first_text = dialog_to_text(first_half)
        second_text = dialog_to_text(second_half)

        sent = emotion_to_sentiment(emotion)

        def normalize_intensity(value):
            try:
                val = int(value)
                if val <= 2:
                    return 1
                elif val == 3:
                    return 2
                elif val >= 4:
                    return 3
            except:
                return ""
            return ""

        if first_text:
            rows.append({
                "Utterances": f"{first_text}".strip(),
                "sentiment": sent,
                "emotion1": emotion if emotion else "neutral",
                "intensity1": normalize_intensity(init_int),
                "emotion2": "",
                "intensity2": "",
                "emotion3": "",
                "intensity3": "",
                "segment": "start",
                "problem_type": problem
            })

        if second_text:
            rows.append({
                "Utterances": f"{second_text}".strip(),
                "sentiment": sent,
                "emotion1": emotion if emotion else "neutral",
                "intensity1": normalize_intensity(fin_int),
                "emotion2": "",
                "intensity2": "",
                "emotion3": "",
                "intensity3": "",
                "segment": "end",
                "problem_type": problem
            })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "Utterances", "sentiment",
            "emotion1", "intensity1",
            "emotion2", "intensity2",
            "emotion3", "intensity3",
            "segment", "problem_type"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Utworzono {output_csv} z {len(rows)} segmentami (początek + koniec rozmów).")


# Uruchomienie sekwencji
if __name__ == "__main__":
    ESCONV_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\ESConv.json"
    NORMALIZED_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\ESConv_normalized.json"
    OUTPUT_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\ESConv_DA_ready.csv"

    # 1. Sprawdzenie skali
    inspect_intensity_scale(ESCONV_PATH)

    # 2. Automatyczna normalizacja (jeśli 1–5)
    normalize_esconv_intensity(ESCONV_PATH, NORMALIZED_PATH)

    # 3. Konwersja do CSV
    convert_esconv_to_da_format(NORMALIZED_PATH, OUTPUT_PATH)


# def load_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
# def dialog_to_text(dialog_turns):
#     """Konwertuje tylko wypowiedzi seekera na jeden tekst"""
#     seeker_texts = []
#     for turn in dialog_turns:
#         if turn.get("speaker") == "seeker":
#             content = turn.get("content", "").strip()
#             if content:  # Tylko jeśli content nie jest pusty
#                 seeker_texts.append(content)
#     return " ".join(seeker_texts)
#
# def split_dialog(dialog):
#     """Dzieli dialog na pierwsze 50% i ostatnie 50% wypowiedzi seekera"""
#     # Filtrujemy tylko wypowiedzi seekera
#     seeker_turns = [turn for turn in dialog if turn.get("speaker") == "seeker"]
#
#     total_turns = len(seeker_turns)
#     if total_turns == 0:
#         return [], []
#
#     mid_point = math.ceil(total_turns / 2)
#
#     first_half = seeker_turns[:mid_point]
#     second_half = seeker_turns[mid_point:]
#
#     return first_half, second_half
#
# def convert_esconv_to_csv(json_path, output_first_half='esconv_first_half.csv', output_second_half='esconv_second_half.csv'):
#     data = load_data(json_path)
#
#     # Przygotowanie danych dla pierwszej połowy
#     first_half_rows = []
#     second_half_rows = []
#
#     for convo in data:
#         dialog = convo.get("dialog", [])
#
#         if len(dialog) == 0:
#             continue
#
#         # Pobieranie labelek
#         initial_intensity = convo.get("survey_score", {}).get("seeker", {}).get("initial_emotion_intensity", "")
#         final_intensity = convo.get("survey_score", {}).get("seeker", {}).get("final_emotion_intensity", "")
#
#         # Dzielenie dialogu
#         first_half, second_half = split_dialog(dialog)
#
#         # Konwersja na tekst
#         first_half_text = dialog_to_text(first_half)
#         second_half_text = dialog_to_text(second_half)
#
#         # Dodawanie do list
#         if first_half_text and initial_intensity:
#             first_half_rows.append({
#                 'conversation': first_half_text,
#                 'label': initial_intensity
#             })
#
#         if second_half_text and final_intensity:
#             second_half_rows.append({
#                 'conversation': second_half_text,
#                 'label': final_intensity
#             })
#
#     # Zapisywanie pierwszej połowy
#     with open(output_first_half, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['conversation', 'label']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(first_half_rows)
#
#     # Zapisywanie drugiej połowy
#     with open(output_second_half, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['conversation', 'label']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(second_half_rows)
#
#     print(f"Utworzono {output_first_half} z {len(first_half_rows)} wierszami")
#     print(f"Utworzono {output_second_half} z {len(second_half_rows)} wierszami")
#
# # Przykład użycia:
# if __name__ == "__main__":
#     # Zastąp 'your_file.json' ścieżką do twojego pliku ESConv
#     ESCONV_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/ESConv.json'
#     convert_esconv_to_csv(ESCONV_PATH)