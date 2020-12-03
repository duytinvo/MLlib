import os
import tempfile
import shutil
from pyrouge import Rouge155
from mlmodels.metrics.metric import Metric

class RougeMetric(Metric):
    def __init__(self, rouge_dir='mlmodels/metrics/ROUGE-1.5.5/', rouge_args=None, verbose=False):
        self.r = Rouge155(rouge_dir=rouge_dir, rouge_args=rouge_args)
        self.rouge_args = rouge_args

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if len(summary) == 0:
            return {'rouge_1_recall': 0.0, 'rouge_1_recall_cb': 0.0, 'rouge_1_recall_ce': 0.0, 'rouge_1_precision': 0.0,
                    'rouge_1_precision_cb': 0.0, 'rouge_1_precision_ce': 0.0, 'rouge_1_f_score': 0.0,
                    'rouge_1_f_score_cb': 0.0, 'rouge_1_f_score_ce': 0.0, 'rouge_2_recall': 0.0, 'rouge_2_recall_cb':
                        0.0, 'rouge_2_recall_ce': 0.0, 'rouge_2_precision': 0.0, 'rouge_2_precision_cb': 0.0,
                    'rouge_2_precision_ce': 0.0, 'rouge_2_f_score': 0.0, 'rouge_2_f_score_cb': 0.0, 'rouge_2_f_score_ce'
                    : 0.0, 'rouge_3_recall': 0.0, 'rouge_3_recall_cb': 0.0, 'rouge_3_recall_ce': 0.0, 'rouge_3_precision'
                    : 0.0, 'rouge_3_precision_cb': 0.0, 'rouge_3_precision_ce': 0.0, 'rouge_3_f_score': 0.0,
                    'rouge_3_f_score_cb': 0.0, 'rouge_3_f_score_ce': 0.0, 'rouge_4_recall': 0.0, 'rouge_4_recall_cb':
                        0.0, 'rouge_4_recall_ce': 0.0, 'rouge_4_precision': 0.0, 'rouge_4_precision_cb': 0.0,
                    'rouge_4_precision_ce': 0.0, 'rouge_4_f_score': 0.0, 'rouge_4_f_score_cb': 0.0, 'rouge_4_f_score_ce'
                    : 0.0, 'rouge_l_recall': 0.0, 'rouge_l_recall_cb': 0.0, 'rouge_l_recall_ce': 0.0, 'rouge_l_precision'
                    : 0.0, 'rouge_l_precision_cb': 0.0, 'rouge_l_precision_ce': 0.0, 'rouge_l_f_score': 0.0,
                    'rouge_l_f_score_cb': 0.0, 'rouge_l_f_score_ce': 0.0, 'rouge_w_1.2_recall': 0.0,
                    'rouge_w_1.2_recall_cb': 0.0, 'rouge_w_1.2_recall_ce': 0.0, 'rouge_w_1.2_precision': 0.0,
                    'rouge_w_1.2_precision_cb': 0.0, 'rouge_w_1.2_precision_ce': 0.0, 'rouge_w_1.2_f_score': 0.0,
                    'rouge_w_1.2_f_score_cb': 0.0, 'rouge_w_1.2_f_score_ce': 0.0, 'rouge_s*_recall': 0.0,
                    'rouge_s*_recall_cb': 0.0, 'rouge_s*_recall_ce': 0.0, 'rouge_s*_precision': 0.0,
                    'rouge_s*_precision_cb': 0.0, 'rouge_s*_precision_ce': 0.0, 'rouge_s*_f_score': 0.0,
                    'rouge_s*_f_score_cb': 0.0, 'rouge_s*_f_score_ce': 0.0, 'rouge_su*_recall': 0.0,
                    'rouge_su*_recall_cb': 0.0, 'rouge_su*_recall_ce': 0.0, 'rouge_su*_precision': 0.0,
                    'rouge_su*_precision_cb': 0.0, 'rouge_su*_precision_ce': 0.0, 'rouge_su*_f_score': 0.0,
                    'rouge_su*_f_score_cb': 0.0, 'rouge_su*_f_score_ce': 0.0}
        self.r.system_dir = tempfile.mkdtemp()
        self.r.model_dir = tempfile.mkdtemp()
        self.r.system_filename_pattern = 'system.(\d+).txt'
        self.r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
        with open(os.path.join(self.r.system_dir, "system.0.txt"), "w") as outputf:
            outputf.write(summary)
        for ref_idx, ref in enumerate(reference):
            with open(os.path.join(self.r.model_dir, f"model.{chr(ord('A') + ref_idx)}.0.txt"), "w") as outputf:
                outputf.write(ref)
        if self.rouge_args is not None:
            output = self.r.convert_and_evaluate(rouge_args=f"-e {self.r.data_dir} " + self.r.args)
        else:
            output = self.r.convert_and_evaluate()
        output_dict = self.r.output_to_dict(output)
        shutil.rmtree(self.r.system_dir)
        shutil.rmtree(self.r.model_dir)
        return {"rouge": output_dict}

    def evaluate_batch(self, summaries, references, aggregate=True):
        if not aggregate:
            results = [self.evaluate_example(summ, ref) for ref, summ in zip(references, summaries)]
            return results
        self.r.system_dir = tempfile.mkdtemp()
        self.r.model_dir = tempfile.mkdtemp()
        self.r.system_filename_pattern = 'system.(\d+).txt'
        self.r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
        for idx, (refs, summ) in enumerate(zip(references, summaries)):
            with open(os.path.join(self.r.system_dir, f"system.{idx}.txt"), "w") as outputf:
                outputf.write(summ)
            if not isinstance(refs, list):
                refs = [refs]
            for ref_idx, ref in enumerate(refs):
                with open(os.path.join(self.r.model_dir, f"model.{chr(ord('A') + ref_idx)}.{idx}.txt"), "w") as outputf:
                    outputf.write(ref)
        if self.rouge_args is not None:
            output = self.r.convert_and_evaluate(rouge_args=f"-e {self.r.data_dir} " + self.r.args)
        else:
            output = self.r.convert_and_evaluate()
        output_dict = self.r.output_to_dict(output)
        shutil.rmtree(self.r.system_dir)
        shutil.rmtree(self.r.model_dir)
        return {"rouge": output_dict}


if __name__ == '__main__':

    rouge = RougeMetric()
    score = rouge.evaluate_example('How are you?', 'How you doing?')
    print(score)