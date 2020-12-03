# pylint: disable=C0103
import unittest
from mlmodels.metrics.summeval.cider_metric import CiderMetric
from mlmodels.metrics.summeval.test_util import EPS, CANDS, REFS


class TestScore(unittest.TestCase):
    def test_score(self):
        metric = CiderMetric(tokenize=False)
        score = metric.evaluate_batch(CANDS, REFS)
        ref = 2.5759014911864084
        self.assertTrue((score['cider'] - ref) < EPS)

if __name__ == '__main__':
    unittest.main()
