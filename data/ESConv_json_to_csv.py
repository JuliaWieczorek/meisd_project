import json
import csv
import math

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def dialog_to_text(dialog_turns):
    """Konwertuje tylko wypowiedzi seekera na jeden tekst"""
    seeker_texts = []
    for turn in dialog_turns:
        if turn.get("speaker") == "seeker":
            content = turn.get("content", "").strip()
            if content:  # Tylko jeśli content nie jest pusty
                seeker_texts.append(content)
    return " ".join(seeker_texts)

def split_dialog(dialog):
    """Dzieli dialog na pierwsze 50% i ostatnie 50% wypowiedzi seekera"""
    # Filtrujemy tylko wypowiedzi seekera
    seeker_turns = [turn for turn in dialog if turn.get("speaker") == "seeker"]

    total_turns = len(seeker_turns)
    if total_turns == 0:
        return [], []

    mid_point = math.ceil(total_turns / 2)

    first_half = seeker_turns[:mid_point]
    second_half = seeker_turns[mid_point:]

    return first_half, second_half

def convert_esconv_to_csv(json_path, output_first_half='esconv_first_half.csv', output_second_half='esconv_second_half.csv'):
    data = load_data(json_path)

    # Przygotowanie danych dla pierwszej połowy
    first_half_rows = []
    second_half_rows = []

    for convo in data:
        dialog = convo.get("dialog", [])

        if len(dialog) == 0:
            continue

        # Pobieranie labelek
        initial_intensity = convo.get("survey_score", {}).get("seeker", {}).get("initial_emotion_intensity", "")
        final_intensity = convo.get("survey_score", {}).get("seeker", {}).get("final_emotion_intensity", "")

        # Dzielenie dialogu
        first_half, second_half = split_dialog(dialog)

        # Konwersja na tekst
        first_half_text = dialog_to_text(first_half)
        second_half_text = dialog_to_text(second_half)

        # Dodawanie do list
        if first_half_text and initial_intensity:
            first_half_rows.append({
                'conversation': first_half_text,
                'label': initial_intensity
            })

        if second_half_text and final_intensity:
            second_half_rows.append({
                'conversation': second_half_text,
                'label': final_intensity
            })

    # Zapisywanie pierwszej połowy
    with open(output_first_half, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['conversation', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(first_half_rows)

    # Zapisywanie drugiej połowy
    with open(output_second_half, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['conversation', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(second_half_rows)

    print(f"Utworzono {output_first_half} z {len(first_half_rows)} wierszami")
    print(f"Utworzono {output_second_half} z {len(second_half_rows)} wierszami")

# Przykład użycia:
if __name__ == "__main__":
    # Zastąp 'your_file.json' ścieżką do twojego pliku ESConv
    ESCONV_PATH = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/ESConv.json'
    convert_esconv_to_csv(ESCONV_PATH)