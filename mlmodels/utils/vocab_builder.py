# -*- coding: utf-8 -*-
"""
Created on 2020-03-05
@author: duytinvo
"""
from collections import Counter
from random import choices
from mlmodels.utils.csvIO import CSV
from mlmodels.utils.jsonIO import JSON
from mlmodels.utils.special_tokens import PAD, SOT, EOT, UNK, COL, TAB

sys_tokens = [PAD, SOT, EOT, UNK]


# ----------------------------------------------------------------------------------------------------------------------
# ======================================== VOCAB-BUILDER FUNCTIONS =====================================================
# ----------------------------------------------------------------------------------------------------------------------
class Vocab(object):
    def __init__(self, s_paras, t_paras):
        """
        s_paras = [swl_th=None, swcutoff=1]
        t_paras = [twl_th=None, twcutoff=1]
        """
        # NL query
        self.swl, self.swcutoff = s_paras
        self.sw2i, self.i2sw = {}, {}

        # sql or semQL
        self.twl, self.twcutoff = t_paras
        self.tw2i, self.i2tw = {}, {}

    def build(self, files, limit=-1, firstline=True):
        """
        Read a list of file names, return vocabulary
        :param files: list of file names
        :param limit: read number of lines
        """
        swcnt, swl = Counter(), 0
        twcnt, twl = Counter(), 0
        count = 0

        for fname in files:
            # Read input files
            if fname.split(".")[-1] == "csv":
                raw = CSV(fname, limit=limit, firstline=firstline)
            elif fname.split(".")[-1] == "json":
                raw = JSON(fname, source2idx=None, target2idx=None, limit=-1)
            else:
                raise Exception("Not implement yet")

            for line in raw:
                count += 1
                (nl, target) = line
                nl = Vocab.process_nl(nl)
                target = Vocab.process_target(target)
                swcnt, swl = Vocab.update_sent(nl, swcnt, swl)
                twcnt, twl = Vocab.update_sent(target, twcnt, twl)

        swvocab = Vocab.update_vocab(swcnt, self.swcutoff, sys_tokens)

        twvocab = Vocab.update_vocab(twcnt, self.twcutoff, sys_tokens)

        self.sw2i = swvocab
        self.i2sw = Vocab.reversed_dict(swvocab)
        self.swl = swl if self.swl < 0 else min(swl, self.swl)

        self.tw2i = twvocab
        self.i2tw = Vocab.reversed_dict(twvocab)
        self.twl = twl if self.twl < 0 else min(twl, self.twl)

        print("\t- Extracting vocabulary: %d total samples" % count)

        print("\t\t- Natural Language Side: ")
        print("\t\t\t- %d total words" % (sum(swcnt.values())))
        print("\t\t\t- %d unique words" % (len(swcnt)))
        print("\t\t\t- %d unique words appearing at least %d times" % (len(swvocab) - 4, self.swcutoff))
        print("\t\t- Label Side: ")
        print("\t\t\t- %d total words" % (sum(twcnt.values())))
        print("\t\t\t- %d unique words" % (len(twcnt)))
        print("\t\t\t- %d unique words appearing at least %d times" % (len(twvocab) - 4, self.twcutoff))

    @staticmethod
    def flatten(lst):
        if isinstance(lst[0], list) or isinstance(lst[0], tuple):
            return [item for sublist in lst for item in sublist]
        else:
            return lst

    @staticmethod
    def idx2text(pad_ids, i2t, level=2, i2colname=None, i2tabname=None, dbids=None):

        def i2token(token_id, i):
            token = UNK
            if token_id < len(i2t):
                token = i2t.get(token_id, UNK)
            elif token_id >= len(i2t):
                if dbids is not None:
                    if token_id < len(i2t) + len(i2colname[dbids[i]]) - 4:
                        token = i2colname[dbids[i]].get(token_id - len(i2t) + 4, COL)
                    else:
                        token = i2tabname[dbids[i]].get(token_id - len(i2t) - len(i2colname[dbids[i]]) + 8, TAB)
            return token

        if level == 3:
            docs = []
            for i, sent_ids in enumerate(pad_ids):
                sents = []
                for j, token_ids in enumerate(sent_ids):
                    tokens = []
                    for k, token_id in enumerate(token_ids):
                        tokens.append(i2token(token_id, i))
                    sents.append(tokens)
                docs.append(sents)
            return docs
            # return [[[i2t[char] for char in chars] for chars in wds] for wds in pad_ids]
        elif level == 2:
            sents = []
            for i, token_ids in enumerate(pad_ids):
                tokens = []
                for j, token_id in enumerate(token_ids):
                    tokens.append(i2token(token_id, i))
                sents.append(tokens)
            return sents
            # return [[i2t[wd] for wd in wds] for wds in pad_ids]
        else:
            tokens = []
            for i, token_id in enumerate(pad_ids):
                tokens.append(i2token(token_id, i))
            return tokens
            # return [i2t[token] for token in pad_ids]

    @staticmethod
    def update_sent(sent, wcnt, wl):
        newsent = []
        for item in sent:
            if isinstance(item, list) or isinstance(item, tuple):
                newsent.extend([tk for tk in item])
            else:
                newsent.append(str(item))
        # newsent = " ".join(newsent).split()
        wcnt.update(newsent)
        wl = max(wl, len(newsent))
        return wcnt, wl

    @staticmethod
    def update_label(sent, wcnt):
        newsent = []
        if isinstance(sent, int):
            wcnt.update([sent])
        else:
            for item in sent:
                if isinstance(item, list) or isinstance(item, tuple):
                    newsent.extend([tk for tk in item])
                else:
                    newsent.append(item)
            wcnt.update(newsent)
        return wcnt

    @staticmethod
    def update_vocab(cnt, cutoff, pads):
        lst = pads + [x for x, y in cnt.most_common() if y >= cutoff]
        vocabs = dict([(y, x) for x, y in enumerate(lst)])
        return vocabs

    @staticmethod
    def reversed_dict(cur_dict):
        inv_dict = {v: k for k, v in cur_dict.items()}
        return inv_dict

    @staticmethod
    def hierlst2idx(vocab_words=None, unk_words=True, sos=False, eos=False, reverse=False):
        """
        Return a function to convert hierarchical list of tokens into indexes (a list of lists [of lists] of tokens)
        """
        def f(sent):
            level_flag = False
            word_ids = []
            if vocab_words is not None:
                for word in sent:
                    token_id = []
                    for token in word:
                        # if table
                        if isinstance(token, list) or isinstance(token, tuple):
                            level_flag = True
                            tk_ids = []
                            for tk in token:
                                # ignore words out of vocabulary
                                if tk in vocab_words:
                                    tk_ids += [vocab_words[tk]]
                                else:
                                    if unk_words:
                                        tk_ids += [vocab_words[UNK]]
                                    else:
                                        raise Exception(
                                            "Unknown key is not allowed. Check that your vocab (tags?) is correct")
                            token_id += [tk_ids]
                        # if either type or nl
                        else:
                            # ignore words out of vocabulary
                            if token in vocab_words:
                                tk_ids = vocab_words[token]
                            else:
                                if unk_words:
                                    tk_ids = vocab_words[UNK]
                                else:
                                    raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                            token_id += [tk_ids]

                    if level_flag:
                        if reverse:
                            token_id = token_id[::-1]
                        if sos:
                            # Add start-of-sentence
                            token_id = [[vocab_words[SOT]]] + token_id

                        if eos:
                            # add end-of-sentence
                            token_id = token_id + [[vocab_words[EOT]]]

                        if len(token_id) == 0:
                            token_id = [[[vocab_words[PAD], ]]]

                    word_ids += [token_id]
            if not level_flag:
                if reverse:
                    word_ids = word_ids[::-1]
                if sos:
                    # Add start-of-sentence

                    word_ids = [[vocab_words[SOT]]] + word_ids

                if eos:
                    # add end-of-sentence
                    word_ids = word_ids + [[vocab_words[EOT]]]

                if len(word_ids) == 0:
                    word_ids = [[[vocab_words[PAD], ]]]
            return word_ids
        return f

    @staticmethod
    def process_target(target):
        target_toks = target.split()
        return target_toks

    @staticmethod
    def process_nl(nl):
        nl_toks = nl.lower().split()
        return nl_toks

    @staticmethod
    def lst2idx(tokenizer=None, vocab_words=None, unk_words=True, sos=False, eos=False,
                vocab_chars=None, unk_chars=True, sow=False, eow=False,
                reverse=False):
        """
        Return a function to convert tag2idx or word/word2idx (a list of words comprising characters)
        """

        def f(sequence):
            sent = tokenizer(sequence)
            if vocab_words is not None:
                # SOw,EOw words for  SOW
                word_ids = []
                for word in sent:
                    # ignore words out of vocabulary
                    if word in vocab_words:
                        word_ids += [vocab_words[word]]
                    else:
                        if unk_words:
                            word_ids += [vocab_words[UNK]]
                        else:
                            raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if reverse:
                    word_ids = word_ids[::-1]
                if sos:
                    # Add start-of-sentence
                    word_ids = [vocab_words[SOT]] + word_ids

                if eos:
                    # add end-of-sentence
                    word_ids = word_ids + [vocab_words[EOT]]

                if len(word_ids) == 0:
                    word_ids = [[vocab_words[PAD], ]]

            if vocab_chars is not None:
                char_ids = []
                padding = []
                if sow:
                    # add start-of-word
                    padding = [vocab_chars[SOT]] + padding
                if eow:
                    # add end-of-word
                    padding = padding + [vocab_chars[EOT]]

                for word in sent:
                    if word not in [SOT, EOT, PAD, UNK]:
                        char_id = []
                        for char in word:
                            # ignore chars out of vocabulary
                            if char in vocab_chars:
                                char_id += [vocab_chars[char]]
                            else:
                                if unk_chars:
                                    char_id += [vocab_chars[UNK]]
                                else:
                                    raise Exception("Unknow key is not allowed. Check that your vocab (tags?) is correct")
                        if sow:
                            # add start-of-word
                            char_id = [vocab_chars[SOT]] + char_id
                        if eow:
                            # add end-of-word
                            char_id = char_id + [vocab_chars[EOT]]
                        char_ids += [char_id]
                    else:
                        char_ids += [[vocab_chars[word], ]]

                if reverse:
                    char_ids = char_ids[::-1]
                if sos:
                    # add padding start-of-sentence
                    char_ids = [padding] + char_ids
                if eos:
                    # add padding end-of-sentence
                    char_ids = char_ids + [padding]
                if len(char_ids) == 0:
                    char_ids = [padding]

            if vocab_words is not None:
                if vocab_chars is not None:
                    return list(zip(char_ids, word_ids))
                else:
                    return word_ids
            else:
                return char_ids

        return f

    @staticmethod
    def tgt2idx(vocab_words=None, unk_words=True, sos=False, eos=False,
                vocab_col=None, vocab_tab=None, reverse=False):
        """
        Return a function to convert tag2idx or word/word2idx (a list of words comprising characters)
        """

        def f(sent, dbid):
            # SOw,EOw words for  SOW
            word_ids = []
            for word in sent:
                out_id = ()
                if vocab_words is not None:
                    tok_id = vocab_words.get(word, vocab_words[UNK])
                    out_id += (tok_id, )
                    if not unk_words and tok_id == vocab_words[UNK]:
                        raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if vocab_col is not None:
                    col_id = vocab_col[dbid].get(word, vocab_col[dbid][UNK])
                    out_id += (col_id,)
                    if not unk_words and col_id == vocab_col[dbid][UNK]:
                        raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if vocab_tab is not None:
                    tab_id = vocab_tab[dbid].get(word, vocab_tab[dbid][UNK])
                    out_id += (tab_id,)
                    if not unk_words and tab_id == vocab_tab[dbid][UNK]:
                        raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if len(out_id) == 1:
                    word_ids += [out_id[0]]
                else:
                    word_ids += [out_id]
            if reverse:
                word_ids = word_ids[::-1]
            if sos:
                # Add start-of-sentence
                out_id = ()
                if vocab_words is not None:
                    out_id += (vocab_words[SOT],)
                if vocab_col is not None:
                    out_id += (vocab_col[dbid][UNK],)
                if vocab_tab is not None:
                    out_id += (vocab_tab[dbid][UNK],)
                if len(out_id) == 1:
                    word_ids = [out_id[0]] + word_ids
                else:
                    word_ids = [out_id] + word_ids

            if eos:
                # add end-of-sentence
                out_id = ()
                if vocab_words is not None:
                    out_id += (vocab_words[EOT],)
                if vocab_col is not None:
                    out_id += (vocab_col[dbid][UNK],)
                if vocab_tab is not None:
                    out_id += (vocab_tab[dbid][UNK],)
                if len(out_id) == 1:
                    word_ids = word_ids + [out_id[0]]
                else:
                    word_ids = word_ids + [out_id]

            if len(word_ids) == 0:
                out_id = ()
                if vocab_words is not None:
                    out_id += (vocab_words[PAD],)
                if vocab_col is not None:
                    out_id += (vocab_col[dbid][PAD],)
                if vocab_tab is not None:
                    out_id += (vocab_tab[dbid][PAD],)
                if len(out_id) == 1:
                    word_ids = [[out_id[0], ]]
                else:
                    word_ids = [[out_id, ]]

            return word_ids
        return f

    @staticmethod
    def minibatches(data, batch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)

        Yields:
            list of tuples
        """
        d_batch = []
        for datum in data:
            if len(d_batch) == batch_size:
                yield d_batch
                d_batch = []
            d_batch += [datum]

        if len(d_batch) != 0:
            if len(d_batch) != batch_size:
                d_batch = choices(d_batch, k=batch_size)
            yield d_batch

    @staticmethod
    def absa_extractor(tokens, labels, prob=None):
        cur = []
        tok = []
        p = []
        span = []
        por = []
        for i in range(len(labels) - 1):
            if labels[i] != 'O' and labels[i] not in sys_tokens:
                cur += [labels[i]]  # idx_tag in [S, B, I, E]
                por += [labels[i][2:]]
                tok += [tokens[i]]
                p += [prob[i] if prob is not None else 0]
                if labels[i].startswith("S"):
                    span.extend([[" ".join(tok), Counter(por).most_common(1)[0][0], " ".join(cur), sum(p)/len(p)]])
                    cur = []
                    por = []
                    tok = []
                    p = []
                else:
                    if labels[i + 1] == 'O' or labels[i + 1].startswith('S') or \
                            labels[i + 1].startswith('B') or labels[i + 1] in sys_tokens:
                        span.extend([[" ".join(tok), Counter(por).most_common(1)[0][0], " ".join(cur), sum(p)/len(p)]])
                        cur = []
                        por = []
                        tok = []
                        p = []
        # we don't care the'O' label
        if labels[-1] != 'O' and labels[-1] not in sys_tokens:
            cur += [labels[-1]]  # idx_tag in [S, B, I, E]
            por += [labels[i][2:]]
            tok += [tokens[-1]]
            p += [prob[-1] if prob is not None else 0]
            span.extend([[" ".join(tok), Counter(por).most_common(1)[0][0], " ".join(cur), sum(p)/len(p)]])
        return span


if __name__ == '__main__':
    import torch
    from mlmodels.utils.idx2tensor import Data2tensor, seqPAD

    Data2tensor.set_randseed(12345)
    device = torch.device("cpu")
    dtype = torch.long
    use_cuda = False
    filename = "/media/data/review_response/Dev.json"

    s_paras = [-1,  1]
    t_paras = [-1, 1]

    vocab = Vocab(s_paras, t_paras)
    vocab.build([filename])

    nl2ids = vocab.lst2idx(vocab_words=vocab.sw2i, unk_words=True, eos=True)

    tg2ids = vocab.lst2idx(vocab_words=vocab.tw2i, unk_words=False, sos=True, eos=True)

    train_data = JSON(filename, source2idx=nl2ids, target2idx=tg2ids)
    # train_data = Csvfile(filename)

    data_idx = []
    batch = 8
    for d in Vocab.minibatches(train_data, batch):
        data_idx.append(d)
        nl, target = list(zip(*d))

        nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=vocab.sw2i[PAD], nlevels=1)
        nl_tensor = Data2tensor.idx2tensor(nl_pad_ids, dtype=torch.long, device=device)
        nl_len_tensor = Data2tensor.idx2tensor(nl_lens, dtype=torch.long, device=device)

        lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=vocab.tw2i[PAD], nlevels=1)
        lb_tensor = Data2tensor.idx2tensor(lb_pad_ids, dtype=torch.long, device=device)

        break