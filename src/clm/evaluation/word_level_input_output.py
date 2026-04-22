
from transformers import AutoModel, AutoTokenizer
import torch
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
def visualize_heatmap(
    att_matrix: torch.Tensor,
    row_labels,
    col_labels,
    title="Attention heatmap",
    value_fmt=".2f",
    annot=True,
    annot_size=6,
    figsize=(8, 6),
    unit='token',
):
    """
    att_matrix: 2D tensor or numpy array, shape (N, M)
    row_labels / col_labels: list[str]
    """
    if isinstance(att_matrix, torch.Tensor):
        att_np = att_matrix.detach().cpu().numpy()
    else:
        att_np = att_matrix

    plt.figure(figsize=figsize)
    plt.title(title, fontsize=14)
    sns.heatmap(
        att_np,
        annot=annot,
        fmt=value_fmt,
        cmap="viridis",
        linewidths=0.5,
        xticklabels=row_labels,
        yticklabels=col_labels,
        annot_kws={"size": annot_size} if annot else None,
    )
    plt.xlabel("Key/Value (attended to)", fontsize=10)
    plt.ylabel("Query (attending from)", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{title}_{unit}.pdf')


def visualize_token_level(outputs, input_ids, tokenizer, layer_idx, head_idx, unit='token'):
     # list of L tensors (B, H, T, T)


    if unit=='token':
        attention_weights = outputs.attentions
        att_layer_head = attention_weights[layer_idx][0, head_idx, :, :]  # (T, T)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    else:
        att_layer_head = outputs[layer_idx, head_idx, :, :] # (T, T)
        tokens = input_ids
    visualize_heatmap(
        att_layer_head,
        row_labels=tokens,
        col_labels=tokens,
        title=f"Token-level attention (L{layer_idx}, H{head_idx})",
        value_fmt=".2f",
        annot=True,
        annot_size=2,
        unit=unit
    )


class WordLevelIO:
    def __init__(self, data_fp, tokenizer, model, batch_size, scale, device):
        # print('before init of WLIO')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        self.data_fp = data_fp
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.scale = scale
        self.tokenizer = tokenizer
        self.sent_ids, self.words= self._get_data(self.data_fp)
        self.attentions = self.get_attention()

    def _get_data(self, data_fp):
        sents = Path(data_fp).read_text().strip().split('\n')
        words = []
        sent_id = []
        for i, sent in enumerate(sents):
            punct = json.loads(sent)['sentence_good'][-1]
            sent = json.loads(sent)['sentence_good'][:-1].split()
            sent_p = sent+[punct]
            words.append(sent_p)
            sent_id.append(i)
        return sent_id, words

    def _encode_batch(self, sentences: List[List[str]], tokenizer):
        texts = [" ".join(words) for words in sentences]

        enc = tokenizer(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        # enc["input_ids"]: (B, T)
        # enc["offset_mapping"]: (B, T, 2)
        # enc["attention_mask"]: (B, T)
        return enc

    def _build_tok_word_maps_for_sentence(self, words, offsets):
        """
        :param tokenizer: tokenizer
        Use offset to build tok2word and word2tok mappings
        """
        word_str = " ".join(words)
        # self.words: ["I", "saw", "the", "policeman", ...]
        word_spans = []  # [(start, end), ...]
        pos = 0
        for w in words:
            start = pos
            end = start + len(w)
            word_spans.append((start, end))
            pos = end + 1

        # build tok2word and word2tok
        tok2word = {}
        word2tok = defaultdict(list)
        for tid, (tok_start, tok_end) in enumerate(offsets):
            piece = word_str[tok_start:tok_end]
            if piece.isspace():
                continue
            if tok_end <= tok_start:
                continue
            anchor_wid = []
            for wid, (w_start, w_end) in enumerate(word_spans):
                if tok_end <= w_start or tok_start >= w_end:
                    continue

                anchor_wid.append(wid)

            if not anchor_wid:
                print(words)
                raise ValueError(f"Cannot find word for token {tid} with offset ({tok_start}, {tok_end})")


            assert anchor_wid.__len__() == 1, "Token {tid} maps to multiple words {anchor_wid}"
            tok2word[tid] = anchor_wid[0]
            word2tok[anchor_wid[0]].append(tid)

        # print(tok2word, word2tok)
        if not tok2word:
            raise ValueError("tok2word is empty; check tokenization / concat / offsets.")

        return tok2word, word2tok

    # original: (num_batches, layer, batch_size, head, source, target)
    # reshaped: (num_sequences=num_batch*batch_size, layer, head, source, target)

    def _to_word_level_attentions(self, att, tok2word, device):
        """
            - bpe/unigram: tid -> wid
            - superbpe:    tid -> List[wid]
            att: (L, H, T, T)
        """
        L, H, T, _ = att.shape

        valid_tids = sorted(tok2word.keys())
        T_valid_tids = len(valid_tids)
        att_valid = att[:, :, valid_tids, :]  # (L, H, T_valid, T_full)
        att_valid = att_valid[:, :, :, valid_tids]
        word_membership = torch.tensor(
            [tok2word[tid] for tid in valid_tids],
            dtype=torch.long,
            device=device,
        )
        num_words = int(word_membership.max().item()) + 1

        wm_target = word_membership.view(1, 1, 1, T_valid_tids).expand(L, H, T_valid_tids, T_valid_tids)

        target_summed = torch.zeros(L, H, T_valid_tids, num_words, device=device)
        target_summed.scatter_add_(dim=-1, index=wm_target, src=att_valid)

        # target_summed = target_summed.permute(0, 1, 3, 2)  # (L, H, target_word, source_tok)
        wm_source = word_membership.view(1, 1, T_valid_tids, 1).expand(L, H, T_valid_tids, num_words)
        source_summed = torch.zeros(L, H, num_words, num_words, device=device)  # (L, H, W, W)
        source_summed.scatter_add_(
            dim=2,
            index=wm_source,
            src=target_summed,
        )
        counts = torch.zeros_like(source_summed)
        counts.scatter_add_(
            dim=2,
            index=wm_source,
            src=torch.ones_like(target_summed),
        )
        word_attn = source_summed / counts

        return word_attn  # (L, H, W, W)

    def get_attention(self):
        self.model.to(self.device)
        self.model.eval()

        all_word_attn = []
        sentences = []
        for word in self.words:
            sentences.append(word)
        skip = 0
        for start in range(0, len(sentences), self.batch_size):
            batch_sents = sentences[start:start + self.batch_size]

            enc = self._encode_batch(batch_sents, self.tokenizer)

            input_ids = enc["input_ids"].to(self.device)
            if len(input_ids[0])>128:
                skip+=1
                print(f'sent is tooooooo long with {len(input_ids[0])} tokens.')
                continue
            att_mask = enc["attention_mask"].to(self.device)  # (B, T)
            offsets = enc["offset_mapping"]  # (B, T, 2)
            # print("input_ids:", input_ids)
            # print("attention_mask:", att_mask)
            # print("valid lengths:", att_mask.sum(dim=-1))
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                )

                # visualize_token_level(outputs, input_ids, self.tokenizer, 6, 2)
            att = torch.stack(outputs.attentions, dim=0)  # (L, B, H, T, T)

            B = input_ids.size(0)
            for b in range(B):
                words = batch_sents[b]
                seq_len = int(att_mask[b].sum().item())

                offsets_b = offsets[b][:seq_len].tolist()
                att_b = att[:, b, :, :seq_len, :seq_len]  # (L, H, T', T')

                tok2word, word2tok = self._build_tok_word_maps_for_sentence(
                    words, offsets_b
                )
                word_attn_b = self._to_word_level_attentions(
                    att_b, tok2word, self.device
                )
                # visualize_token_level(word_attn_b,batch_sents[0], self.tokenizer, 6, 2, unit='word')
                all_word_attn.append(word_attn_b.cpu())
        print(f"Skipped {skip}/1000 examples because length > 128.")
        return all_word_attn

    # def _generate_batched_attentions(self, model):
    #     model.eval()
    #     model.to(self.device)
    #
    #     # batches, tok2word, toks = _tokenize_and_batchify(tokenizer, ctx_size, stride, batch_size, words)
    #     with torch.no_grad():
    #         for batch in tqdm(self.batches):
    #             # output = model(batch.to(device), output_attentions=True)
    #             output = model(batch.to(self.device), output_attentions=True)
    #             # original: (layer, batch_size, head, source, target)
    #             # reshaped: (batch_size, layer, head, source, target)
    #             attention = torch.stack(output.attentions, dim=1)
    #             del output
    #             yield attention

    # def _head_probe(self, attention):
    #     """
    #     x_i -> x_j attention and x_i <- x_j attention
    #     e.g.
    #     1       0       0       0       0
    #     0.3     0.7     0       0       0
    #     0.2     0.2     0.6     0       0
    #     0.1     0.1     0.5     0.4     0
    #     0.01    0.09    0.3     0.3     0.3
    #     -> probe says: maxes that go through the diagonal points are the parents
    #
    #     1. scaling based on the number of 'attendable' tokens
    #     2. crude 1024 with 1024 stride vs sentence level
    #     """
    #
    #     by_head_predictions = []
    #     for sent in attention:
    #         for layer in range(12):
    #             for head in range(12):
    #                 for w in range(sent.shape[-1]):
    #                     if self.scale:
    #                         scaling_factor = torch.tensor(list(range(1, sent.shape[-1]+1)), device=self.device)
    #                         scaled_att = sent[layer, head] * scaling_factor  # sent[layer, head]: (source, target)
    #                     else:
    #                         scaled_att = sent[layer, head]
    #                     both = torch.cat([scaled_att, scaled_att.permute(1, 0)], dim=-1)
    #                     max_id = torch.argmax(both[w]).item()
    #                     if max_id >= sent.shape[-1]:
    #                         max_id = max_id - sent.shape[-1]
    #                     by_head_predictions.append(max_id)
    #     return by_head_predictions

    # def get_sas_preds(self, model, unit='word'):
    #     batched_attentions_generator = self._generate_batched_attentions(model)
    #     batch_counter = 0
    #     num_words = 0
    #     by_head_predictions = []
    #     for batched_attentions in batched_attentions_generator:
    #         torch.cuda.empty_cache()
    #         # def sizeof_fmt(num, suffix='B'):
    #         #     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    #         #     for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
    #         #         if abs(num) < 1024.0:
    #         #             return "%3.1f %s%s" % (num, unit, suffix)
    #         #         num /= 1024.0
    #         #     return "%.1f %s%s" % (num, 'Yi', suffix)
    #         #
    #         # for tensor in [batched_attentions]:
    #         #     if type(tensor) == list:
    #         #         memory_bytes = tensor[0].element_size() * tensor[0].nelement() * len(tensor)
    #         #         memory_mb = memory_bytes / (1024 ** 2)
    #         #     elif type(tensor) == torch.Tensor:
    #         #         memory_bytes = tensor.element_size() * tensor.nelement()
    #         #         memory_mb = memory_bytes / (1024 ** 2)
    #         # print(f"Tensor memory usage: {memory_mb:.2f} MiB")
    #         #
    #         # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
    #         #         locals().items())), key=lambda x: -x[1])[:10]:
    #         #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    #         # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    #         if unit == 'word':
    #             batched_attentions = self._to_word_level_attentions_single_batch(
    #                 cml_num_seqs=batch_counter * self.batch_size,
    #                 attentions=batched_attentions
    #             )
    #         batch_by_head_predictions, first_word_overlaps = self._head_probe(
    #             batch=batched_attentions,
    #             num_words=num_words,
    #             num_seqs=batch_counter * self.batch_size
    #         )
    #         batch_counter += 1
    #         num_words += len(batch_by_head_predictions)
    #         if first_word_overlaps:
    #             num_words -= 1
    #             by_head_predictions.pop()  # pop the overlapping element
    #         by_head_predictions.extend(batch_by_head_predictions)
    #     print(len(by_head_predictions), len(self.word2tok))
    #     assert len(by_head_predictions) == len(self.word2tok), 'Numbers of words do not match.'
    #     return by_head_predictions
