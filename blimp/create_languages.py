import random
from pathlib import Path
from conllu import parse
from tqdm import tqdm
import os
from random import choice
from random import shuffle
import pandas as pd
import json
import json
import random


random.seed(42)
closed_class = {'ADP', 'AUX', 'CCONJ', 'DET', 'PRON', 'SCONJ'}
# PRON = ["i", "me",  "myself", "we", "ours", "ourselves", "you",  "yours", "yourself", "yourselves", "he", "him", "himself", "she",  "hers", "herself", "it",  "itself", "they", "them",  "theirs", "themselves", "what", "which", "who", "whom","when", "where", "why", "how"]
DET = ["the","this","a","an","no","all","another","each","that","any","those","these","both","every","either","neither"]
CCONJ = ["and","but","or","yet"]
SCONJ = ["that","if","although","after","whereas","while","before","as","though","until","because", "since","once","whether","unless","albeit","till","whilst"]
AUX = ["will","be","had","were","being","is","would","was","do","could","are","have","been","has","did","should","might","can","does","'s","may","must","ca","'s","am","shall","art","ar","re","ought","need"]
ADP = ["at","in","of","near","for","by","to","with","on","from","behind","into","within","despite","against","as","over","than","during","about","between","among","except","through","around","after","like","off","without","under","before","throughout","unlike","across","toward","along","above","aboard","until","upon","via","beneath","unto","beyond","per","below","amongst","till","beside","amid","onto","towards","underneath","alongside"]
FUNCTION_WORDS = set(DET + CCONJ + SCONJ + AUX + ADP)

compact_function_words = {}

for w in FUNCTION_WORDS:
    if w in DET:
        compact_function_words[w] ='the'
    elif w in CCONJ:
        compact_function_words[w] ='and'
    elif w in SCONJ:
        compact_function_words[w] ='that'
    elif w in AUX:
        compact_function_words[w] ='will'
    elif w in ADP:
        compact_function_words[w] ='at'

FIVE_FUNCTION_WORDS= ['the', 'and', 'that', 'will', 'at']

# Pseudowords mapping
pesudo_class = {x:[x] for x in FUNCTION_WORDS}

pesudo_words = Path('function_word_pseudowords.txt').read_text().strip().split('\n')
all_pesudo_words = []
for line in pesudo_words:
    word, pseudo = line.strip().split('\t')
    all_pesudo_words.append(pseudo)
    pesudo_class[word].append(pseudo)

for c, p in pesudo_class.items():
    if len(pesudo_class[c])==1:
        pesudo_class[c] = random.sample(all_pesudo_words, 10)


#===============================================================
def no_function_words(words):
    return [word for word in words if word.lower() not in FUNCTION_WORDS]



# def build_function_word_bigram_pair(train_split, dev_split, test_split):
#     func_word_dict = {}
#     all_data = train_split + dev_split + test_split
#
#     for sent in tqdm(all_data):
#         for word in sent.strip().split():
#             func_word_dict[word.lower()] = random.choice(list(FUNCTION_WORDS))
#
#     for fw in FUNCTION_WORDS:
#         fw_l = fw.lower()
#         if fw_l not in func_word_dict:
#             func_word_dict[fw_l] = random.choice(list(FUNCTION_WORDS))
#
#     func_word_dict["unk"] = random.choice(list(FUNCTION_WORDS))
#     with open("bigram_function.json", "w", encoding="utf-8") as f:
#         json.dump(func_word_dict, f, ensure_ascii=False, indent=4)
#     return func_word_dict


def bigram_function_words(words, func_word_dict):
    n = len(words)
    for i in range(n - 1, -1, -1):
        if words[i].lower() in FUNCTION_WORDS:
            if i < n - 1:
                next_token = words[i + 1].lower()
                marker = func_word_dict.get(next_token, func_word_dict["unk"])
                words[i] = marker
            else:
                words[i] = func_word_dict["unk"]
    return words

def add_function_words(words):
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random.choice(pesudo_class[word.lower()])
    return words

def reduce_function_words(words):
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = compact_function_words[word.lower()]
    return words

def random_function_words(words):
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random.choice(list(FUNCTION_WORDS))
    return words


def random_five_function_words(words):
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random.choice(list(FIVE_FUNCTION_WORDS))
    return words

def within_boundary(tree_lines, change, overall):
    tree = list(tree_lines)
    for i in range(len(tree) - 1, -1, -1):
        line = tree[i].strip()
        if not line or line.startswith("#"):
            continue

        cols = line.split("\t")
        if len(cols) <= 4:
            continue
        try:
            word_id = int(cols[0])
            word = cols[1].lower()
            head = int(cols[4])
        except ValueError:
            print('ValueError in line:', line)
            continue

        if word in FUNCTION_WORDS:
            overall+=1
        if word not in FUNCTION_WORDS or head <= 0:
            continue
        if abs(head - word_id) == 1:
            continue


        change+=1

        target_idx = head - 1
        line_moved = tree.pop(i)

        if i < target_idx:
            target_idx -= 1
        tree.insert(target_idx, line_moved)
    return tree, change, overall


def get_split_sent(conll):
    words = [x.split('\t')[1] for x in conll.strip().split('\n') if not x.startswith('#') and '-' not in x.split('\t')[0] and '.' not in x.split('\t')[0]]
    return words

def get_split_tree(conll):
    tree_lines = [x for x in conll.strip().split('\n') if not x.startswith('#') and '-' not in x.split('\t')[0] and '.' not in x.split('\t')[0]]
    return tree_lines


def convert_text(function_name):
    if 'bigram_function' in function_name:
        with open('bigram_function.json', 'r', encoding='utf-8') as f:
            func_word_dict = json.load(f)
    with open(f'/Users/xiulinyang/Desktop/conll/data/{function_name}/train.txt', 'w', encoding='utf-8') as f_train, \
         open(f'/Users/xiulinyang/Desktop/conll/data/{function_name}/dev.txt', 'w', encoding='utf-8') as f_dev, \
         open(f'/Users/xiulinyang/Desktop/conll/data/{function_name}/test.txt', 'w', encoding='utf-8') as f_test:
        overall=0
        change=0
        for sent in tqdm(train):
            text = get_split_sent(sent)
            if function_name =='no_function':
                filtered_text = no_function_words(text)
            elif function_name == 'bigram_function':
                filtered_text = bigram_function_words(text, func_word_dict)
            elif function_name =='more_function':
                filtered_text = add_function_words(text)
            elif function_name =='five_function':
                filtered_text = reduce_function_words(text)
            elif function_name =='random_function':
                filtered_text = random_function_words(text)
            elif function_name == 'within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change,overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
            elif function_name =='natural_function':
                filtered_text = text
            elif function_name == 'five_function_within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change, overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
                filtered_text = reduce_function_words(filtered_text)
            elif function_name =='five_function_random_function':
                filtered_text = random_five_function_words(text)
            elif function_name =='bigram_function_within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change, overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
                filtered_text = bigram_function_words(filtered_text, func_word_dict)
            f_train.write(' '.join(filtered_text).lower())
            f_train.write('\n')
        if 'within_boundary' in function_name:
            print("overall:", overall, "change:", change, 'rate:', change/overall)

        for sent in tqdm(dev):
            text = get_split_sent(sent)
            if function_name =='no_function':
                filtered_text = no_function_words(text)
            elif function_name == 'bigram_function':
                filtered_text = bigram_function_words(text, func_word_dict)
            elif function_name =='more_function':
                filtered_text = add_function_words(text)
            elif function_name =='five_function':
                filtered_text = reduce_function_words(text)
            elif function_name =='random_function':
                filtered_text = random_function_words(text)
            elif function_name == 'within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree = within_boundary(tree_lines)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
            elif function_name =='natural_function':
                filtered_text = text
            elif function_name == 'five_function_within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change, overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
                filtered_text = reduce_function_words(filtered_text)
            elif function_name =='five_function_random_function':
                filtered_text = random_five_function_words(text)
            elif function_name =='bigram_function_within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change, overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
                filtered_text = bigram_function_words(filtered_text, func_word_dict)

            f_dev.write(' '.join(filtered_text).lower())
            f_dev.write('\n')

        for sent in tqdm(test):
            text = get_split_sent(sent)
            if function_name == 'no_function':
                filtered_text = no_function_words(text)
            elif function_name == 'bigram_function':
                filtered_text = bigram_function_words(text, func_word_dict)
            elif function_name =='more_function':
                filtered_text = add_function_words(text)
            elif function_name =='five_function':
                filtered_text = reduce_function_words(text)
            elif function_name =='random_function':
                filtered_text = random_function_words(text)
            elif function_name == 'within_boundary':
                tree_lines = get_split_tree(sent)
                modified_tree = within_boundary(tree_lines)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
            elif function_name =='natural_function':
                filtered_text = text
            elif function_name == 'five_function_within_boundary':

                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change, overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
                filtered_text = reduce_function_words(filtered_text)
            elif function_name =='five_function_random_function':
                filtered_text = random_five_function_words(text)
            elif function_name =='bigram_function_within_boundary':

                tree_lines = get_split_tree(sent)
                modified_tree, change, overall = within_boundary(tree_lines, change, overall)
                filtered_text = [line.split('\t')[1] for line in modified_tree]
                filtered_text = bigram_function_words(filtered_text, func_word_dict)
            f_test.write(' '.join(filtered_text).lower())
            f_test.write('\n')

if __name__ == '__main__':
    exp_name = 'bigram_function_within_boundary'  # no_function, bigram_function, more_function, five_function, random_function, within_boundary
    os.makedirs(f'/Users/xiulinyang/Desktop/conll/data/{exp_name}', exist_ok=True)

    train = Path('/Users/xiulinyang/Desktop/conll/train.conll').read_text(encoding='utf-8').strip().split('\n\n')
    dev = Path('/Users/xiulinyang/Desktop/conll/dev.conll').read_text(encoding='utf-8').strip().split('\n\n')
    test = Path('/Users/xiulinyang/Desktop/conll/test.conll').read_text(encoding='utf-8').strip().split('\n\n')

    convert_text(exp_name)
