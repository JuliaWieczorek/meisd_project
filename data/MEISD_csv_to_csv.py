#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_to_da_halves.py
Konwertuje CSV (wiersze = utterance z kolumnami takimi jak: dialog_ids, uttr_ids, Utterances, sentiment, emotion, intensity, emotion2, intensity2, emotion3, intensity3)
na CSV gdzie każdy wiersz to połowa konwersacji (segment=start/end) z agregowanymi etykietami:
Utterances,sentiment,emotion1,intensity1,emotion2,intensity2,emotion3,intensity3,segment,problem_type

agg_emotion(col) -> bierze najczęściej występującą emocję w danej połowie rozmowy
agg_intensity(col) -> liczy średnią intensywność (per kolumna), zaokrągla i normalizuje do skali 1–3 (jeśli trzeba).

bierze średnią z wartości liczbowych dla danej kolumny (np. intensity).
zaokrągla do najbliższej liczby całkowitej.
jeśli dane są np. w skali 1–5, to normalizuje do 1–3.

W efekcie koncowym:
Każda połowa (segment = start lub end) ma więc:

    -Utterances = wszystkie wypowiedzi sklecone w jedno.
    -sentiment = dominujący lub wyliczony.
    -emotion1/intensity1 = dominująca emocja + średnia intensywność z kolumny intensity.
    -emotion2/intensity2 = dominująca emocja + średnia z intensity2 (jeśli w ogóle występuje).
    -emotion3/intensity3 = analogicznie.
    -problem_type = puste (brak w danych).
"""

import csv
import math
import os
from collections import Counter, defaultdict
from statistics import mean

# --- Mapowanie emocji -> sentyment (jak w Twoim wcześniejszym kodzie) ---
SENTIMENT_MAP = {
    'joy': 'positive', 'love': 'positive', 'gratitude': 'positive', 'relief': 'positive', 'pride': 'positive',
    'anger': 'negative', 'fear': 'negative', 'sadness': 'negative', 'disgust': 'negative', 'anxiety': 'negative',
    'guilt': 'negative', 'shame': 'negative', 'neutral': 'neutral', 'acceptance': 'neutral', 'surprise': 'neutral'
}

def read_csv(input_csv, encoding='utf-8'):
    with open(input_csv, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows

EMOTION_CLEAN_MAP = {
    "digust": "disgust",
    "df": "disgust",
    "sadnes": "sadness",
    "asadness": "sadness",
    "faer": "fear",
    "fera": "fear",
    "anger1": "anger",
    "sur": "surprise",
    "an": "anger",
    "like": "joy",
    "l": "joy"
}

VALID_EMOTIONS = set([
    "joy", "love", "gratitude", "relief", "pride",
    "anger", "fear", "sadness", "disgust", "anxiety",
    "guilt", "shame", "neutral", "acceptance", "surprise"
])

def clean_emotion(e):
    if not e:
        return ""
    e = e.strip().lower()
    if e in EMOTION_CLEAN_MAP:
        return EMOTION_CLEAN_MAP[e]
    if e in VALID_EMOTIONS:
        return e
    # wszystko inne uznajemy za neutralne / do usunięcia
    return ""

def inspect_intensity_and_emotions(rows):
    intensities = []
    emotions = Counter()

    for r in rows:
        for col in ('intensity', 'intensity2', 'intensity3'):
            v = r.get(col, "")
            if v and str(v).strip().isdigit():
                intensities.append(int(v.strip()))

        for col in ('emotion', 'emotion2', 'emotion3'):
            e = clean_emotion(r.get(col, ""))
            if e:
                r[col] = e  # <- nadpisz wiersz oczyszczoną wartością
                emotions[e] += 1
            else:
                r[col] = ""  # usuń brudną wartość

    print("=== INSPEKCJA DANYCH ===")
    if intensities:
        print(f"Zakres intensywności: {min(intensities)}–{max(intensities)}")
        print("Rozkład:")
        for val, cnt in sorted(Counter(intensities).items()):
            print(f"  {val}: {cnt}")
    else:
        print("Nie znaleziono wartości liczbowych intensywności.")

    print("\nUnikalne emocje (po czyszczeniu):")
    for emo, cnt in emotions.most_common():
        print(f"  {emo}: {cnt}")
    print("========================\n")

    return intensities, list(emotions.keys())

def normalize_intensity_value(val, detected_max=None):
    """Normalizuje pojedynczą wartość intensywności do skali 1-3,
       jeśli wykryto, że dane są w 1-5. Jeśli detected_max is None, zachowaj oryginał jeśli <=3.
       Zwraca None jeśli nie da się sparsować."""
    if val is None or val == "":
        return None
    try:
        v = int(str(val).strip())
    except:
        return None

    if detected_max is None:
        # nie znamy zakresu — jeśli v<=3 zostawiamy, jeśli v>3 stosujemy mapę 1-5->1-3
        if v <= 3:
            return v
        # założenie mapowania 1-5 -> 1-3
        if v <= 2:
            return 1
        if v == 3:
            return 2
        return 3
    else:
        max_v = detected_max
        if max_v <= 3:
            return v
        if max_v == 5:
            if v <= 2:
                return 1
            if v == 3:
                return 2
            return 3
        if max_v == 4:
            # proporcjonalne przeskalowanie 1..4 -> 1..3
            # mapujemy: 1->1, 2->1, 3->2, 4->3 (można też zrobić round((v-1)*(2/3))+1)
            if v <= 2:
                return 1
            if v == 3:
                return 2
            return 3
        # nietypowa skala — zfallbackuj do mapowania 1-5->1-3
        if v <= 2:
            return 1
        if v == 3:
            return 2
        return 3

def most_common_or_none(seq):
    seq = [s for s in seq if s is not None and str(s).strip() != ""]
    if not seq:
        return ""
    return Counter(seq).most_common(1)[0][0]

def majority_sentiment_from_emotions(emotion_list):
    mapped = []
    for e in emotion_list:
        if not e:
            continue
        me = str(e).strip().lower()
        if me in SENTIMENT_MAP:
            mapped.append(SENTIMENT_MAP[me])
    if not mapped:
        return ""
    return Counter(mapped).most_common(1)[0][0]

def aggregate_half(rows_half, detected_max_intensity):
    """Dla danej listy wierszy (połowa dialogu) zwraca:
       - concatenated text (Utterances)
       - sentiment (mode lub wyliczony z emocji)
       - emotion1,intensity1, emotion2,intensity2, emotion3,intensity3 (mode dla emocji, avg/rounded dla intensyw)
    """
    # Utterances concatenation (preserve order)
    texts = [r.get('Utterances','').strip() for r in rows_half if r.get('Utterances','').strip() != ""]
    joined_text = " ".join(texts).strip()

    # sentiment: try mode of 'sentiment' column
    sentiment_vals = [ (r.get('sentiment') or "").strip().lower() for r in rows_half if (r.get('sentiment') or "").strip() != "" ]
    sentiment_mode = most_common_or_none(sentiment_vals)
    if not sentiment_mode:
        # spróbuj z emocji mapując
        emotion_vals_all = []
        for c in ('emotion','emotion2','emotion3'):
            emotion_vals_all += [ (r.get(c) or "").strip().lower() for r in rows_half if (r.get(c) or "").strip() != "" ]
        sentiment_mode = majority_sentiment_from_emotions(emotion_vals_all)
    if not sentiment_mode:
        sentiment_mode = "neutral"

    # emotions: mode per column
    e1 = most_common_or_none([ (r.get('emotion') or "").strip().lower() for r in rows_half ])
    e2 = most_common_or_none([ (r.get('emotion2') or "").strip().lower() for r in rows_half ])
    e3 = most_common_or_none([ (r.get('emotion3') or "").strip().lower() for r in rows_half ])

    # intensities: compute mean (rounded) and then normalize to 1-3 if needed
    def agg_intensity_for_col(colname):
        vals = []
        for r in rows_half:
            v = r.get(colname, "")
            if v is None or str(v).strip() == "":
                continue
            try:
                vals.append(int(str(v).strip()))
            except:
                pass
        if not vals:
            return ""
        avg = mean(vals)
        # round to nearest integer
        avg_rounded = int(math.floor(avg + 0.5))
        normalized = normalize_intensity_value(avg_rounded, detected_max=detected_max_intensity)
        return normalized if normalized is not None else ""

    import math
    i1 = agg_intensity_for_col('intensity')
    i2 = agg_intensity_for_col('intensity2')
    i3 = agg_intensity_for_col('intensity3')

    # return with empty strings for missing values (matching target CSV)
    return {
        'Utterances': joined_text,
        'sentiment': sentiment_mode,
        'emotion1': e1 if e1 else "neutral",
        'intensity1': i1 if i1 != "" else "",
        'emotion2': e2 if e2 else "",
        'intensity2': i2 if i2 != "" else "",
        'emotion3': e3 if e3 else "",
        'intensity3': i3 if i3 != "" else ""
    }

def convert_csv_to_da_halves(input_csv, output_csv):
    rows = read_csv(input_csv)
    if not rows:
        print("Brak wierszy w wejściowym CSV.")
        return

    # 1) inspect intensities & emotions
    intensities, emotions = inspect_intensity_and_emotions(rows)
    detected_max = max(intensities) if intensities else None

    # 2) group by dialog_ids
    grouped = defaultdict(list)
    for r in rows:
        dialog_id = r.get('dialog_ids') or r.get('dialog_id') or r.get('dialog') or ""
        grouped[dialog_id].append(r)

    # 3) ensure order within each dialog: try uttr_ids numeric, otherwise keep file order
    def sort_dialog_list(lst):
        # try uttr_ids
        try:
            return sorted(lst, key=lambda x: int(x.get('uttr_ids') if x.get('uttr_ids') not in (None,"") else float('inf')))
        except:
            # fallback: return as-is
            return lst

    rows_out = []
    for dialog_id, items in grouped.items():
        items_sorted = sort_dialog_list(items)
        total = len(items_sorted)
        if total == 0:
            continue
        # split into two halves: ceil for first half (like Twój wcześniejszy kod)
        mid = math.ceil(total / 2)
        first_half = items_sorted[:mid]
        second_half = items_sorted[mid:]

        # aggregate
        agg1 = aggregate_half(first_half, detected_max)
        agg1['segment'] = 'start'
        agg1['problem_type'] = ""  # brak info w CSV — możesz zmienić jeśli masz inną kolumnę
        rows_out.append(agg1)

        if second_half:
            agg2 = aggregate_half(second_half, detected_max)
            agg2['segment'] = 'end'
            agg2['problem_type'] = ""
            rows_out.append(agg2)

    # 4) write output CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    fieldnames = [
        "Utterances", "sentiment",
        "emotion1", "intensity1",
        "emotion2", "intensity2",
        "emotion3", "intensity3",
        "segment", "problem_type"
    ]
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            # Ensure values are strings
            out = {k: ("" if r.get(k) is None else str(r.get(k))) for k in fieldnames}
            writer.writerow(out)

    print(f"Utworzono plik: {output_csv}  — liczba segmentów: {len(rows_out)}")
    print("Gotowe.")

MEISD_PATH = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\MEISD_text.csv"
OUTPUT_CSV = r"C:\Users\juwieczo\DataspellProjects\meisd_project\data\MEISD_DA_ready.csv"

if __name__ == "__main__":
    convert_csv_to_da_halves(MEISD_PATH, OUTPUT_CSV)