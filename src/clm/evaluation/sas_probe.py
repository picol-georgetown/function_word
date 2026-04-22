import os
from glob import glob
import torch, pathlib, argparse, sys
from tqdm import tqdm
from word_level_input_output import WordLevelIO
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
print(f"cuda available: {torch.cuda.is_available()}")
print(f"number of gpus: {torch.cuda.device_count()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
CHECKPOINTS= [10]

empty_categories = ['superlative_quantifiers_1', 'determiner_noun_agreement_irregular_2',
                                'determiner_noun_agreement_with_adj_2', 'superlative_quantifiers_2',
                                'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_2',
                                'matrix_npi']


def head_probe(wlio, scale=True):
    """
    x_i -> x_j attention and x_i <- x_j attention
    e.g.
    1       0       0       0       0
    0.3     0.7     0       0       0
    0.2     0.2     0.6     0       0
    0.1     0.1     0.5     0.4     0
    0.01    0.09    0.3     0.3     0.3
    -> probe says: maxes that go through the diagonal points are the parents
    """
    by_head_predictions = []
    for seq_id, seq in enumerate(wlio.attentions):
        seq = seq.to(wlio.device)
        L, H, W, _ = seq.shape
        if scale:
            scaling_factor = torch.tensor(list(range(1, seq.shape[-1]+1)), device=device)
            scaling_factor = scaling_factor.expand(*seq.shape)
            seq = seq * scaling_factor  # seq should have a shape (layer, head, source, target)
        both = torch.cat([seq, seq.permute(0, 1, 3, 2)], dim=-1)
        max_ids = torch.argmax(both, dim=-1)
        parents = torch.where(
            max_ids < W,
            max_ids,
            max_ids - W
        )
        parents = parents.permute(2, 0, 1)  # (W, L, H)

        by_head_predictions.append(parents.tolist())
    return by_head_predictions

def get_and_write_by_head_predictions(
        data_fp, model, tokenizer, batch_size,
        scale, device, output_dir, model_dir, ckpt=None
):
    # get predictions
    # ['pred-pred-pred...-pred', 'pred-pred-pred-...pred', ]
    wlio = WordLevelIO(data_fp=data_fp, model= model, tokenizer=tokenizer, batch_size=batch_size, device=device, scale=scale)
    predictions = head_probe(wlio, scale=scale)

    lines = []
    for i, sent in enumerate(predictions):
        sid = wlio.sent_ids[i]
        for w_idx, word in enumerate(sent):
            cols = [
                '-'.join(str(idx) for idx in layer)
                for layer in word
            ]
            line = f"{sid}\t{w_idx}\t" + '\t'.join(cols)
            lines.append(line)

    out = '\n'.join(lines)
    data_name = data_fp.split('/')[-1].split('.')[0]
    # write to a file
    model_name = data_name+'_'+model_dir.split(os.path.sep)[-1]+f'-ckpt-{ckpt}'  # model_name/checkpoint
    suffix = '@scaled' if scale else '@unscaled'
    fn = f"{model_name}-sas_preds{suffix}.tsv"
    with open(os.path.join(output_dir, fn), 'w') as f:
        f.write(out)
def main():
    ROOT_DIR = pathlib.Path(__file__).parent.resolve()
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_fp', default=os.path.join(DATA_DIR, 'ewt.txt'),
                        help=f"path to token file, default={os.path.join(DATA_DIR, 'ewt.txt')}")
    parser.add_argument('-m', '--model_dir', required=False,
                        help="path to the model, if contains multiple checkpoints, runs all checkpoints")
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help=f'batch size, default=1')
    parser.add_argument('-u', '--unscale_weights', action='store_true',
                        help=f'unscale attention weights, default is false')
    parser.add_argument('-o', '--output_dir', default=os.path.join(DATA_DIR, 'sas_preds'),
                        help=f"output directory, default={os.path.join(DATA_DIR, 'sas_preds')}")
    args = parser.parse_args()

    data_fp, model_dir, output_dir, batch_size, unscale = \
        args.data_fp, args.model_dir,args.output_dir, args.batch_size, args.unscale_weights
    scale = unscale == 0


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for ckpt in CHECKPOINTS:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_dir)
        config.output_attentions = True
        config.use_cache = False
        config.output_attentions = True
        model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, revision=f'epoch-{ckpt}').to(device)
        for p in glob(f'{data_fp}/*.jsonl'):
            p_name = p.split('/')[-1].split('.')[0]
            if p_name in empty_categories:
                continue
            get_and_write_by_head_predictions(data_fp=p, model=model, tokenizer=tokenizer, batch_size=batch_size, scale=scale, device=device,output_dir=output_dir, model_dir=model_dir, ckpt=ckpt)


if __name__ == "__main__":
    main()
"""TEST
data_fp = os.path.join(os.getcwd(), 'data', 'ewt.txt')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
model = AblationGPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-70m')
ctx_size = 1024
stride = 1024
batch_size = 2
device = 'cpu'
wlio = WordLevelIO(data_fp, tokenizer, ctx_size, stride, batch_size, device)
predictions = head_probe(wlio, model)
write = []
for word in predictions:
    write.append(
        '\t'.join(
            [
                '-'.join([str(idx) for idx in layer])
                for layer in word
            ]
        )
    )
# attentions = wlio.get_attentions(model, 'word')

# seq = attentions[0]
# both = torch.cat([seq, seq.permute(0,1,3,2)], dim=-1)
# max_ids = torch.argmax(both, dim=-1)
# max_ids.shape

# for idx in max_ids:
# for i, both in enumerate(zip(seq, seq.permute(0, 1, 3, 2))):
#     outgoing, incoming = both
#     idx = np.argmax(outgoing.tolist() + incoming.tolist())
#     if idx >= len(outgoing.tolist()):
#         row, col = idx - len(outgoing), i
#         parent = row
#     else:
#         row, col = i, idx
#         parent = col

# for i, both in enumerate(zip(seq, seq.permute(0, 1, 3, 2))):
#     outgoing, incoming = both
#     break
#     idx = np.argmax(outgoing.tolist() + incoming.tolist())
#     if idx >= len(outgoing.tolist()):
#         row, col = idx - len(outgoing), i
#         parent = row
#     else:
#         row, col = i, idx
#         parent = col
"""