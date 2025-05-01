import re

def analyze_sections(text):
    sections = re.findall(r'<Section \d+>', text)
    num_sections = len(sections)
    
    scores = re.findall(r'Score: (\d+)', text)
    scores = list(map(int, scores))

    if num_sections != len(scores):
        return -1, -1, -1
    
    last_score = scores[-1]
    avg = sum(scores) / len(scores) if scores else 0
    return num_sections, avg, last_score

with open('./output.txt', 'r', encoding='utf-8') as file:
    input_text = file.read()

num_sections, avg_score, last_score = analyze_sections(input_text)

print("Section Num:", num_sections)
print("Avg Score:", avg_score)
print("Last Score:", last_score)