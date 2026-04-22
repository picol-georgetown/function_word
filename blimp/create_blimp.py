from glob import glob
from pathlib import Path
import json
import os
import stanza
from tqdm import tqdm
import random

nlp = stanza.Pipeline(
    'en',
    processors='tokenize',
    use_gpu=False
)



from create_languages import pesudo_class,reduce_function_words, no_function_words, bigram_function_words, within_boundary, get_split_sent, get_split_tree
from create_languages import FUNCTION_WORDS, FIVE_FUNCTION_WORDS
category_jsonl = glob('natural_function_blimp/*.jsonl')
EXPERIMENT = 'five_function_within_boundary'
output_path = f'{EXPERIMENT}_blimp'
os.makedirs(output_path, exist_ok=True)
FILTER_FUNCTIONS = False #only when one wants to start over the creation process, change it to True


if FILTER_FUNCTIONS:
    for cat in tqdm(category_jsonl):
        cat_name = cat.split('/')[-1]
        try:
            with open(f'{output_path}/{cat_name}', 'w', encoding='utf-8') as f_out:
                sents = Path(cat).read_text().strip().split('\n')
                for sent in sents:
                    sent_dict = json.loads(sent)
                    sent_good = [x.text for x in nlp(sent_dict['sentence_good'].lower()).sentences[0].words]
                    sent_bad = [x.text for x in nlp(sent_dict['sentence_bad'].lower()).sentences[0].words]

                    diff_word = [x for x in sent_good if x not in sent_bad] + [x for x in sent_bad if x not in sent_good]
                    func_word = [x for x in diff_word if x in FUNCTION_WORDS]
                    if len(func_word)>0:
                        continue
                    else:
                        sent_dict['sentence_good'] = ' '.join(sent_good).lower()
                        sent_dict['sentence_bad'] = ' '.join(sent_bad).lower()
                        f_out.write(json.dumps(sent_dict))
                        f_out.write('\n')
        except json.decoder.JSONDecodeError as e:
            print(f'the document {cat_name} is skipped because it does not have any sentence')

def random_function_words(words):
    random_func_dict = {}
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random.choice(list(FUNCTION_WORDS))
            random_func_dict[word.lower()] = words[i]
    return words, random_func_dict


def random_five_function_words(words):
    random_func_dict={}
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random.choice(list(FIVE_FUNCTION_WORDS))
            random_func_dict[word.lower()] = words[i]
    return words, random_func_dict



def generate_random_function_pair(words, random_func_dict):
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random_func_dict[word.lower()]
    return words


def add_function_words(words):
    function_dic = {}
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = random.choice(pesudo_class[word.lower()])
            function_dic[word.lower()] = words[i]
    return words, function_dic

def add_function_words_to_pair(words, func_word_dict):
    for i, word in enumerate(words):
        if word.lower() in FUNCTION_WORDS:
            words[i] = func_word_dict[word.lower()]
    return words


if 'bigram_function' in EXPERIMENT:
    with open('/Users/xiulinyang/Desktop/conll/bigram_function.json', 'r', encoding='utf-8') as f:
        func_word_dict = json.load(f)

def parse_tree(sent_line):
    return json.loads(sent_line)

def cross_boundary(tree):
    good_tree = get_split_tree(tree['sentence_good'])
    bad_tree = get_split_tree(tree['sentence_bad'])
    original_sent = tree['sentence_good'].split('\n')[0][len('# text = '):]
    good_sent,_,_ = within_boundary(good_tree,0,0)
    bad_sent,_,_ = within_boundary(bad_tree,0,0)
    good_words = [line.split('\t')[1] for line in good_sent][:-1]
    bad_words = [line.split('\t')[1] for line in bad_sent][:-1]
    return good_words, bad_words, original_sent

for cat in tqdm(sorted(category_jsonl)):
    cat_name = cat.split('/')[-1]

    with open(f'{output_path}/{cat_name}', 'w', encoding='utf-8') as f_out:
        sents = Path(cat).read_text().strip().split('\n')

        for i, sent in enumerate(sents):
            if not sent.strip():
                continue
            if 'within_boundary' in EXPERIMENT:
                trees = Path(f'blimp_conll/{cat_name}').read_text().strip().split('\n')
                assert len(trees)==len(sents)
                cat_tree = parse_tree(trees[i])
            new_sent_dict = {}
            sent_dict = json.loads(sent)
            punck = sent_dict['sentence_good'][-1]
            sent_good = [x.text for x in nlp(sent_dict['sentence_good'][:-1].lower()).sentences[0].words]
            sent_bad = [x.text for x in nlp(sent_dict['sentence_bad'][:-1].lower()).sentences[0].words]
            if EXPERIMENT=='random_function':
                no_func_good, fun_dict = random_function_words(sent_good)
                no_func_bad = generate_random_function_pair(sent_bad, fun_dict)
            elif EXPERIMENT=='five_function':
                no_func_good = reduce_function_words(sent_good)
                no_func_bad = reduce_function_words(sent_bad)
            elif EXPERIMENT =='more_function':
                no_func_good, fun_dict = add_function_words(sent_good)
                no_func_bad = add_function_words_to_pair(sent_bad, fun_dict)
            elif EXPERIMENT =='bigram_function':
                no_func_good = bigram_function_words(sent_good, func_word_dict)
                no_func_bad = bigram_function_words(sent_bad, func_word_dict)
            elif EXPERIMENT =='no_function':
                no_func_good = no_function_words(sent_good)
                no_func_bad = no_function_words(sent_bad)

            elif EXPERIMENT=='within_boundary':
                no_func_good, no_func_bad, original_sent = cross_boundary(cat_tree)
            elif EXPERIMENT =='five_function_random_function':
                no_func_good, fun_dict = random_five_function_words(sent_good)
                no_func_bad = generate_random_function_pair(sent_bad, fun_dict)
            elif EXPERIMENT == 'five_function_within_boundary':
                no_func_good, no_func_bad, original_sent = cross_boundary(cat_tree)
                no_func_good = reduce_function_words(no_func_good)
                no_func_bad = reduce_function_words(no_func_bad)


            elif EXPERIMENT == 'bigram_function_within_boundary':
                no_func_good, no_func_bad, original_sent = cross_boundary(cat_tree)
                no_func_good = bigram_function_words(no_func_good, func_word_dict)
                no_func_bad = bigram_function_words(no_func_bad, func_word_dict)

            if no_func_bad!=no_func_good:
                new_sent_dict['original'] = sent_dict['sentence_good']
                new_sent_dict['sentence_good'] = ' '.join(no_func_good)+' '+punck
                new_sent_dict['sentence_bad'] = ' '.join(no_func_bad)+' '+ punck
                f_out.write(json.dumps(new_sent_dict))
                f_out.write('\n')
            else:
                print(cat)
                print(sent_good, sent_bad)
