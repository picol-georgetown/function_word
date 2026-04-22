import os
from glob import glob
import stanza
from itertools import islice
from tqdm import tqdm
from pathlib import Path
import json
nlp = stanza.Pipeline(
    'en',
    processors='tokenize,pos,lemma,depparse',
    use_gpu=False,
    tokenize_no_ssplit=False,
    tokenize_batch_size=8,
    pos_batch_size=8,
    depparse_batch_size=8
)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk

sources = glob('natural_function_blimp/*.jsonl')
os.makedirs('blimp_conll',exist_ok=True)
for source in tqdm(sources):
    sents = Path(source).read_text().strip().split('\n')
    name = source.split('/')[-1].split('.')[0]
    out_path = f"blimp_conll/{name}.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for sent in sents:
            if not sent.strip():
                continue
            sent_pair = json.loads(sent)
            sent_good = sent_pair['sentence_good']
            sent_bad = sent_pair['sentence_bad']
            try:
                doc_good = nlp(sent_good)
                for sent in doc_good.sentences:
                    sent_text = sent.text if getattr(sent, "text", None) else " ".join(w.text for w in sent.words)
                    rows = []
                    for i, w in enumerate(sent.words, start=1):
                        form   = (w.text or '').replace('\t', ' ')
                        lemma  = (w.lemma or '_').replace('\t', ' ')
                        upos   = (w.upos or '_')
                        head   = str(w.head if w.head is not None else 0)
                        deprel = (w.deprel or '_')
                        rows.append("\t".join([str(i), form, lemma, upos, head, deprel]))
                    text_good = f"# text = {sent_text}\n"+"\n".join(rows)
                    sent_pair['sentence_good'] = text_good

                doc_bad = nlp(sent_bad)
                for sent in doc_bad.sentences:
                    sent_text = sent.text if getattr(sent, "text", None) else " ".join(w.text for w in sent.words)
                    rows = []
                    for i, w in enumerate(sent.words, start=1):
                        form = (w.text or '').replace('\t', ' ')
                        lemma = (w.lemma or '_').replace('\t', ' ')
                        upos = (w.upos or '_')
                        head = str(w.head if w.head is not None else 0)
                        deprel = (w.deprel or '_')
                        rows.append("\t".join([str(i), form, lemma, upos, head, deprel]))
                    text_bad = f"# text = {sent_text}\n" + "\n".join(rows)
                    sent_pair['sentence_bad'] = text_bad
                f_out.write(json.dumps(sent_pair))
                f_out.write('\n')
            except Exception as e:
                print(e)
                continue