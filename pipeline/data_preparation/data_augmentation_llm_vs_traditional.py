import pandas as pd
import json
import random
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from llama_cpp import Llama
from deep_translator import GoogleTranslator
from nltk.corpus import wordnet

# -------------------------
# CONFIG
# -------------------------
MEISD_PATH = '/data/filtered_negative_MEISD_intensity_max_first_25_conv.csv'
ESCONV_PATH = '/data/ESConv.json'
LLAMA_PATH = '/chatbot/llama-2-7b-chat.Q4_K_M.gguf'

# -------------------------
# 1. Load MEISD data and label it binary
# -------------------------
df_data = pd.read_csv(MEISD_PATH)
df_data['label'] = (df_data['max_intensity'] == 2).astype(int)
df = df_data[['Utterances', 'label']].copy()
original_df = df.copy()

# -------------------------
# 2. Load ESConv data & extract seeker utterances for style
# -------------------------
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_seeker_utterances(json_path, max_dialogs=20, max_examples=5):
    data = load_data(json_path)
    seeker_lines = []
    for convo in data[:max_dialogs]:
        for turn in convo.get("dialog", []):
            if turn.get("speaker") == "seeker":
                content = turn.get("content", "").strip()
                if 20 < len(content) < 150:
                    seeker_lines.append(content)
    random.shuffle(seeker_lines)
    selected = seeker_lines[:max_examples]
    return "\n".join([f"- {line}" for line in selected])

style_examples = extract_seeker_utterances(ESCONV_PATH, max_examples=3)

# -------------------------
# 3. Define LLaMA-based augmentation
# -------------------------
llm = Llama(model_path=LLAMA_PATH, n_ctx=2048, n_threads=6, verbose=True)

def generate_esconv_style_response(original_text, examples):
    prompt = f"""
You are an emotional support assistant. Below are example supportive messages from real conversations:
{examples}

Now, rewrite the following message in a similar supportive tone:
"{original_text}"

Response:"""
    output = llm(prompt, max_tokens=150, temperature=0.8, stop=["User:", "Assistant:"])
    return output["choices"][0]["text"].strip()

# -------------------------
# 4. Define classical augmentations
# -------------------------
def synonym_replacement(text):
    words = text.split()
    new_words = words[:]
    num_replacements = max(1, len(words) // 5)
    random_words = random.sample(words, num_replacements)
    for word in random_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            if synonym != word:
                new_words = [synonym if w == word else w for w in new_words]
    return ' '.join(new_words)

def random_insertion(text, n=1):
    words = text.split()
    for _ in range(n):
        new_word = random.choice(words)
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, new_word)
    return ' '.join(words)

def random_deletion(text, p=0.3):
    words = text.split()
    if len(words) == 1:
        return text
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(new_words) if new_words else random.choice(words)

def back_translation(text, src_lang='en', mid_lang='fr', max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            translated = GoogleTranslator(source=src_lang, target=mid_lang).translate(text)
            back_translated = GoogleTranslator(source=mid_lang, target=src_lang).translate(translated)
            return back_translated
        except Exception as e:
            print(f"Back translation error on attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(1)
    return text

# -------------------------
# 5. Unified augmentation function
# -------------------------
def augment_text(text, num_augments=2, mode='mixed'):
    augmented_texts = []
    for _ in range(num_augments):
        if mode == 'llm':
            augmented_texts.append(generate_esconv_style_response(text, style_examples))
        elif mode == 'classic':
            choice = random.choice(['synonym', 'insertion', 'deletion', 'back_translation'])
            if choice == 'synonym':
                augmented_texts.append(synonym_replacement(text))
            elif choice == 'insertion':
                augmented_texts.append(random_insertion(text))
            elif choice == 'deletion':
                augmented_texts.append(random_deletion(text))
            elif choice == 'back_translation':
                augmented_texts.append(back_translation(text))
        elif mode == 'mixed':
            if random.random() < 0.5:
                augmented_texts.append(generate_esconv_style_response(text, style_examples))
            else:
                augmented_texts.append(random_deletion(text))
    return augmented_texts

# -------------------------
# 6. Augmentation function for full dataset
# -------------------------
def augment_binary_data_percent(df, label_column, augment_text_func, augment_percent=10):
    class_counts = df[label_column].value_counts()
    print(f"Liczność klas przed augmentacją: {class_counts.to_dict()}")
    augmented_data = {'Utterances': [], label_column: []}

    for label in class_counts.index:
        class_subset = df[df[label_column] == label].copy()
        num_to_add = int(class_counts[label] * (augment_percent / 100))
        augment_per_sample = max(1, num_to_add // len(class_subset))
        remaining = num_to_add

        for _, row in tqdm(class_subset.iterrows(), total=len(class_subset), desc=f"Augmenting class {label}"):
            if remaining <= 0:
                break
            new_texts = augment_text_func(row['Utterances'], min(augment_per_sample, remaining))
            for new_text in new_texts:
                if remaining <= 0:
                    break
                augmented_data['Utterances'].append(new_text)
                augmented_data[label_column].append(label)
                remaining -= 1

    augmented_df = pd.DataFrame(augmented_data)
    final_df = pd.concat([df, augmented_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    final_counts = final_df[label_column].value_counts()
    print(f"Liczność klas po augmentacji: {final_counts.to_dict()}")
    return final_df

# -------------------------
# 7. Evaluation
# -------------------------
def evaluate_model(df, label_column='label'):
    X_train, X_test, y_train, y_test = train_test_split(df['Utterances'], df[label_column], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = LogisticRegression()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    return classification_report(y_test, y_pred, output_dict=True)

# -------------------------
# 8. Run experiments
# -------------------------
print("Running classic augmentation...")
df_classic = augment_binary_data_percent(original_df, 'label', lambda t, n: augment_text(t, n, mode='classic'), augment_percent=70)

print("Running LLM-based augmentation...")
df_llm = augment_binary_data_percent(original_df, 'label', lambda t, n: augment_text(t, n, mode='llm'), augment_percent=70)

classic_metrics = evaluate_model(df_classic)
llm_metrics = evaluate_model(df_llm)

# -------------------------
# 9. Compare results
# -------------------------
def flatten_report(report):
    return {
        'accuracy': report['accuracy'],
        'f1_macro': report['macro avg']['f1-score'],
        'f1_weighted': report['weighted avg']['f1-score']
    }

results_df = pd.DataFrame([
    {'method': 'classic', **flatten_report(classic_metrics)},
    {'method': 'llm', **flatten_report(llm_metrics)}
])

print("\n=== Augmentation Comparison Results ===")
print(results_df)

results_df.to_csv("augmentation_comparison_results_70.csv", index=False)

# -------------------------
# 10. Save augmented datasets to Excel
# -------------------------
df_classic.to_excel("augmented_dataset_classic_70percent.xlsx", index=False)
df_llm.to_excel("augmented_dataset_llm_70percent.xlsx", index=False)