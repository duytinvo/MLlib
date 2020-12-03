from mlmodels.metrics.rouge import RougeMetric
from mlmodels.metrics.rouge_we_metric import RougeWeMetric
from mlmodels.metrics.bert_score_metric import BertScoreMetric

class MetricsFactory:
    def create_metrics(self, metric, rouge_dir=''):
        if metric == 'rouge':
            metric = RougeMetric(rouge_dir)
        elif metric == 'rouge_we':
            metric = RougeWeMetric()
        elif metric == 'bert_score':
            metric = BertScoreMetric()
        return metric
