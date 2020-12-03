from multiprocessing import Pool
from collections import Counter
from mlmodels.metrics.utils import rouge_n_we, load_embeddings
from mlmodels.metrics.metric import Metric

class RougeWeMetric(Metric):
    def __init__(self, emb_path='embeddings/deps.words', n_gram=3, \
                 n_workers=24, tokenize=True):

        self.word_embeddings = load_embeddings(emb_path)
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.tokenize = tokenize

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(summary, list):
            summary = [summary]
        score = rouge_n_we(summary, reference, self.word_embeddings, self.n_gram, \
                 return_all=True, tokenize=self.tokenize)
        score_dict = {f"rouge_we_{self.n_gram}_p": score[0], f"rouge_we_{self.n_gram}_r": score[1], \
                      f"rouge_we_{self.n_gram}_f": score[2]}
        return score_dict

    def evaluate_batch(self, summaries, references, aggregate=True):
        p = Pool(processes=self.n_workers)
        results = p.starmap(self.evaluate_example, zip(summaries, references))
        p.close()
        if aggregate:
            corpus_score_dict = Counter()
            for x in results:
                corpus_score_dict.update(x)
            for key in corpus_score_dict.keys():
                corpus_score_dict[key] /= float(len(summaries))
            return corpus_score_dict
        else:
            return results

if __name__ == '__main__':

    rouge = RougeWeMetric()
    score = rouge.evaluate_example('How are you?', 'How you doing?')
    print(score)