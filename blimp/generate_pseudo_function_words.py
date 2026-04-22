from wuggy import WuggyGenerator
from tqdm import tqdm

DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because", "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has","did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as","over","than","during","about","between","among","except","through","around","after","like","off","without","under","before","throughout","unlike","across","toward","along","above","aboard","until","upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards","underneath","alongside"]

FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP)
g = WuggyGenerator()
g.load("orthographic_english")
with open('function_word_pseudowords.txt', 'w', encoding='utf-8') as f_out:
    for w in tqdm(list(FUNCTION_WORDS)):
        try:
            for match in g.generate_classic([w],ncandidates_per_sequence=10,max_search_time_per_sequence=25):
                print(match["pseudoword"])
                f_out.write(f"{w}\t{match['pseudoword']}\n")
        except:
            print(w)
            continue