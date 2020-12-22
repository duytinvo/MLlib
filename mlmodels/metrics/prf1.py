# -*- coding: utf-8 -*-
"""
Created on 2020-02-24
@author: duytinvo
"""
from collections import Counter
from sklearn import metrics
from mlmodels.utils.special_tokens import PAD, SOT, EOT, UNK, NULL

sys_tokens = [PAD, SOT, EOT, UNK]


class APRF1:
    @staticmethod
    def sklearn(y_true, y_pred, labels=['0', '1']):
        acc = metrics.accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        return precision, recall, f1, acc

    @staticmethod
    def sklearn_bin(y_true, y_pred, labels=['0', '1']):
        acc = metrics.accuracy_score(y_true, y_pred)
        metric_i = metrics.precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
        results = {"precision_i": metric_i[0],
                   "recall_i": metric_i[1],
                   "f1_i": metric_i[2]}
        metric_a = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        results.update({"precision": metric_a[0],
                        "recall": metric_a[1],
                        "f1": metric_a[2]})
        return results

    @staticmethod
    def accuracies(reference, candidate):
        flatten = lambda l: [item for sublist in l for item in sublist]
        sep_acc = metrics.accuracy_score(flatten(reference), flatten(candidate))
        compose = lambda l: ["_".join(sublist) for sublist in l]
        full_acc = metrics.accuracy_score(compose(reference), compose(candidate))
        return sep_acc, full_acc


class NER_metrics:
    @staticmethod
    def sklearn_metrics(reference, candidate):
        # acc = metrics.accuracy_score(y_true, y_pred)
        # f1_ma = metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
        y_true, y_pred = NER_metrics.span_batch_pair(reference, candidate)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # f1_no = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
        # measures = {"acc": acc, "prf_macro": f1_ma, "prf_weighted": f1_we, "prf_individual": f1_no}
        return precision, recall, f1

    @staticmethod
    def span_metrics(reference, candidate):
        y_true, y_pred = NER_metrics.span_batch(reference, candidate)
        right_ner = len(set(y_true).intersection(set(y_pred)))
        if right_ner != 0:
            precision = right_ner / len(y_pred)
            recall = right_ner / len(y_true)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f1 = 0., 0., 0.
        return precision, recall, f1

    @staticmethod
    def span_batch(reference, candidate):
        pred_labels = []
        gold_labels = []
        for i in range(len(reference)):
            assert len(reference[i]) == len(candidate[i]), print(len(reference[i]), reference[i], len(candidate[i]), candidate[i])
            pred_span = NER_metrics.span_ner(candidate[i])
            pred_span = [str(i) + "_" + l for l in pred_span]
            gold_span = NER_metrics.span_ner(reference[i])
            gold_span = [str(i) + "_" + l for l in gold_span]
            pred_labels.extend(pred_span)
            gold_labels.extend(gold_span)
        return gold_labels, pred_labels

    @staticmethod
    def span_ner(tags):
        cur = []
        span = []
        for i in range(len(tags) - 1):
            if tags[i].upper() != 'O' and tags[i] not in sys_tokens:
                cur += ["_".join([str(i), tags[i]])]  # idx_tag in [S, B, I, E]
                if tags[i].upper().startswith("S"):
                    span.extend(cur)
                    cur = []
                else:
                    if tags[i+1].upper() == 'O' or tags[i+1].upper().startswith('S') or \
                            tags[i+1].upper().startswith('B'):
                        span.extend(["-".join(cur)])
                        cur = []
        # we don't care the'O' label
        if tags[-1].upper() != 'O' and tags[-1] not in sys_tokens:
            cur += ["_".join([str(len(tags) - 1), tags[-1]])]
            span.extend(["-".join(cur)])
        return span

    @staticmethod
    def absa_extractor(tokens, labels, prob=None):
        cur = []
        tok = []
        p = []
        span = []
        por = []
        for i in range(len(labels) - 1):
            if labels[i].upper() != 'O' and labels[i] not in sys_tokens:
                cur += [labels[i]]  # idx_tag in [S, B, I, E]
                por += [labels[i][2:]]
                tok += [tokens[i]]
                p += [prob[i] if prob is not None else 0]
                if labels[i].upper().startswith("S"):
                    span.extend([[" ".join(tok), Counter(por).most_common(1)[0][0], " ".join(cur), sum(p) / len(p)]])
                    cur = []
                    por = []
                    tok = []
                    p = []
                else:
                    if labels[i + 1].upper() == 'O' or labels[i + 1].upper().startswith('S') or \
                            labels[i + 1].upper().startswith('B'):
                        span.extend(
                            [[" ".join(tok), Counter(por).most_common(1)[0][0], " ".join(cur), sum(p) / len(p)]])
                        cur = []
                        por = []
                        tok = []
                        p = []
        # we don't care the'O' label
        if labels[-1].upper() != 'O' and labels[-1] not in sys_tokens:
            cur += [labels[-1]]  # idx_tag in [S, B, I, E]
            por += [labels[-1][2:]]
            tok += [tokens[-1]]
            p += [prob[-1] if prob is not None else 0]
            span.extend([[" ".join(tok), Counter(por).most_common(1)[0][0], " ".join(cur), sum(p) / len(p)]])
        return span

    @staticmethod
    def span_batch_pair(reference, candidate):
        pred_labels = []
        gold_labels = []
        for i in range(len(reference)):
            assert len(reference[i]) == len(candidate[i])
            gold_span, pred_span = NER_metrics.span_ner_pair(reference[i], candidate[i])
            gold_span = [str(i) + "_" + l for l in gold_span]
            pred_span = [str(i) + "_" + l for l in pred_span]
            pred_labels.extend(pred_span)
            gold_labels.extend(gold_span)
        return gold_labels, pred_labels

    @staticmethod
    def span_ner_pair(gold_tags, pred_tags):
        pred_cur = []
        pred_span = []
        cur = []
        span = []
        for i in range(len(gold_tags) - 1):
            if gold_tags[i].upper() != 'O' and gold_tags[i] not in sys_tokens:
                cur += ["_".join([str(i), gold_tags[i]])]  # idx_tag in [S, B, I, E]
                pred_cur += ["_".join([str(i), pred_tags[i]])]
                if gold_tags[i].upper().startswith("S"):
                    span.extend(cur)
                    cur = []
                    pred_span.extend(pred_cur)
                    pred_cur = []

                else:
                    if gold_tags[i+1].upper() == 'O' or gold_tags[i+1].upper().startswith('S') or \
                            gold_tags[i+1].upper().startswith('B'):
                        span.extend(["-".join(cur)])
                        cur = []
                        pred_span.extend(["-".join(pred_cur)])
                        pred_cur = []
        # we don't care the'O' label
        if gold_tags[-1].upper() != 'O' and gold_tags[-1] not in sys_tokens:
            cur += ["_".join([str(len(gold_tags) - 1), gold_tags[-1]])]
            span.extend(["-".join(cur)])
            pred_cur += ["_".join([str(len(pred_tags) - 1), pred_tags[-1]])]
            pred_span.extend(["-".join(pred_cur)])
        return span, pred_span


if __name__ == '__main__':
    reference = \
        [['O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'O'],
         ['O', 'S', 'O', 'B', 'E', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O', 'O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O'], ['S', 'O', 'O', 'O', 'S', 'O', 'O', 'O'],
         ['O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'S', 'O', 'O', 'O', 'O', 'O',
          'O'], ['O', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S', 'O'],
         ['O', 'B', 'E', 'O', 'O', 'O', 'B', 'E', 'O', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'E', 'O',
          'O', 'O', 'O', 'O', 'O', 'S', 'O'],
         ['O', 'S', 'O', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
          'O', 'O'], ['B', 'E', 'O', 'O', 'S', 'O', 'O', 'O', 'S', 'O', 'S', 'O', 'O', 'O', 'O', 'O']]

    candidate = \
        [['I', '<s>', '<s>', '<s>', 'I', '<s>', '<s>', 'I', '<s>', 'I', '<s>', 'S', 'I', 'I', '<PAD>', '<s>', 'I', 'S'],
         ['I', 'I', 'E', 'I', '<PAD>', '<s>', 'I', 'I', 'I', 'I', 'I', '<PAD>', 'S', 'I', 'E', 'I', '<s>', '<s>', '<s>',
          'I', '<PAD>', '<s>', 'I', 'E', 'E', 'E', 'I', 'I', '<s>', '<PAD>', 'I', '<s>', 'I', '<s>'],
         ['<s>', 'I', '<PAD>', '<s>', '<s>', '<PAD>', '<PAD>', '<s>'],
         ['S', 'E', '<s>', '<PAD>', 'I', '<PAD>', 'I', 'I', '<PAD>', 'I', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<s>',
          '<PAD>', 'I', 'S', 'S', '<s>', '<s>', 'I', 'I'],
         ['<s>', 'I', '<PAD>', '<s>', 'I', '<s>', 'I', 'I', '<s>', '<s>', '<s>', '<s>', '<s>', '<PAD>'],
         ['I', '<s>', 'I', '<s>', '<s>', 'I', 'E', 'I', '<s>', '<s>', '<s>', '<PAD>', '<s>', 'I', '<s>', '<s>', '<s>',
          'I', 'I', 'E', 'I', '<s>', '<s>', 'I', 'E', '<s>', '<PAD>', '<PAD>', '<s>'],
         ['<PAD>', '<s>', 'I', 'I', 'I', 'I', '<PAD>', 'I', 'I', 'I', 'I', 'I', '<PAD>', '<PAD>', '<PAD>', '<s>', 'E',
          'E', 'I', '<s>', 'E', '<PAD>', 'E', '<s>'],
         ['<PAD>', 'I', '<PAD>', '<s>', 'I', '<s>', '<PAD>', '<PAD>', '<PAD>', 'E', 'I', '<PAD>', 'I', 'I', 'I', '<s>']]

    gold_labels, pred_labels = NER_metrics.span_batch_pair(reference, candidate)