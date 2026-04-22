from conllu import parse
from glob import glob
from tqdm import tqdm
from pathlib import Path
import json
open_class = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB','NUM'}
closed_class = {'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'SCONJ','PRON'}
other = {'PUNCT', 'SYM', 'X'}
with open('morph_complexity_updated.json', 'r') as fam:
    fam_results = json.load(fam)

lang_names = []
uds = sorted(glob('/Users/xiulinyang/Downloads/ud-treebanks-v2.17/*'))
with open('close_vs_open_count.tsv' ,'w') as co:
    co.write(f'name\tlanguage\tfamily\tclosed_class\topen_class\tvocab\tclosed_ratio\topen_ratio\tclosed_freq_ratio\topen_freq_ratio\n')
    for l in tqdm(uds):
        name = l.split('/')[-1]
        lang = l.split('/')[-1].split('-')[0][3:]
        c_w = []
        o_w = []
        ps = glob(f'{l}/*.conllu')
        for con in ps:
            texts = Path(con).read_text().strip().split('\n\n')
            for sent in texts:
                sent_lines = [x for x in sent.split('\n') if not x.startswith('#')]
                closed_words = [x.split()[1] for x in sent_lines if x.split()[3] in closed_class]
                open_words = [x.split()[1] for x in sent_lines if x.split()[3] in open_class]

                c_w.extend(closed_words)
                o_w.extend(open_words)
            if lang in fam_results:
                family = fam_results[lang]['language_family']
            else:
                family='unknown'
                print(lang)
            lang_names.append(lang)
            ratio= len(set(c_w))/(len(set(o_w))+len(set(c_w)))

            frq_ratio = len(c_w)/(len(c_w)+len(o_w))
            c_ratio =len(set(o_w))/(len(set(o_w))+len(set(c_w)))
            cfrq_ratio = len(o_w)/(len(c_w)+len(o_w))
            co.write(f'{name}\t{lang}\t{family}\t{len(set(c_w))}\t{len(set(o_w))}\t{len(set(o_w))+len(set(c_w))}\t{ratio}\t{c_ratio}\t{frq_ratio}\t{cfrq_ratio}\n')
            # print(f'{lang}\t{len(set(c_w))}\t{len(set(o_w))}\t{len(set(o_w))+len(set(c_w))}\t{ratio}\n')

print(len(set(lang_names)))