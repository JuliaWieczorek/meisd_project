import re
import csv

input_file = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/Adolescence - 1x03 - Episode 3.WEB.NF.en.srt'
output_file = 'C:/Users/juwieczo/DataspellProjects/meisd_project/data/adolescence_s01_e03_subtitles_eng.csv'

with open(input_file, 'r', encoding='latin-1') as file:
    content = file.read()

# Podział na bloki (indeks + czas + tekst)
blocks = re.split(r'\n\s*\n', content.strip())

rows = []
for block in blocks:
    lines = block.strip().split('\n')
    if len(lines) >= 3:
        index = lines[0]
        times = lines[1].split(' --> ')
        start_time = times[0].strip()
        end_time = times[1].strip()
        text = ' '.join(lines[2:]).replace('{\\an8}', '').strip()
        rows.append([index, start_time, end_time, text])

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'start_time', 'end_time', 'text'])
    writer.writerows(rows)

#%%
