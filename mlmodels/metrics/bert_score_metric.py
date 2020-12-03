import bert_score
from mlmodels.metrics.metric import Metric

class BertScoreMetric(Metric):
    def __init__(self, lang='en', model_type='bert-base-uncased', num_layers=8, verbose=False, idf=False,\
                 nthreads=4, batch_size=64, rescale_with_baseline=False):

        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose
        self.idf = idf
        self.nthreads = nthreads
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline

    def evaluate_example(self, summary, reference):
        assert not self.idf, "idf mode not supported for evaluating a single example"
        if isinstance(reference, str):
            reference = [reference]
        all_preds, hash_code = bert_score.score([summary], reference, model_type=self.model_type, \
                                                num_layers=self.num_layers,
                                                verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                                nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                                rescale_with_baseline=self.rescale_with_baseline)
        print(f"hash_code: {hash_code}")
        score = [{"bert_score_precision": p.cpu().item(), "bert_score_recall": r.cpu().item(), "bert_score_f1": \
                 f.cpu().item()} for (p, r, f) in [all_preds]]
        return score

    def evaluate_batch(self, summaries, references, aggregate=True):
        all_preds, hash_code = bert_score.score(summaries, references, model_type=self.model_type, \
                                                num_layers=self.num_layers,
                                                verbose=self.verbose, idf=self.idf, batch_size=self.batch_size,
                                                nthreads=self.nthreads, lang=self.lang, return_hash=True,
                                                rescale_with_baseline=self.rescale_with_baseline)
        print(f"hash_code: {hash_code}")
        if aggregate:
            avg_scores = [s.mean(dim=0) for s in all_preds]
            p_val = avg_scores[0].cpu().item()
            r_val = avg_scores[1].cpu().item()
            f1_val = avg_scores[2].cpu().item()
            scores = {"bert_score_precision": p_val, "bert_score_recall": r_val, "bert_score_f1": f1_val}
            return scores
        else:
            cur_items = [{"bert_score_precision": p.cpu().item(), "bert_score_recall": r.cpu().item(), \
                         "bert_score_f1": f.cpu().item()} for (p, r, f) in list(zip(*all_preds))]
            return cur_items

if __name__ == '__main__':
    bert_score_value = BertScoreMetric()
    score = bert_score_value.evaluate_example('Hi how are you?', 'How are you doing?')
    print(score)