# -*- coding: utf-8 -*-
"""
Created on 2020-03-04
@author: duytinvo
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, \
    AutoModelWithLMHead, AutoModelForTokenClassification
import torch
from transformers import pipeline


class Transformer_pipeline(object):
    def __init__(self, pl_name="sentiment-analysis"):
        self.nlp = pipeline(pl_name)

    def SA(self, sent):
        return self.nlp(sent)

    def QA(self, question, context):
        return self.nlp(question=question, context=context)

    def MLM(self, leftpart, rightpart):
        return self.nlp(f"{leftpart} {self.nlp.tokenizer.mask_token} {rightpart}")

    def NER(self, sent):
        return self.nlp(sent)



class AutoTransformer_api(object):
    def __init__(self, model_name="bert-base-cased-finetuned-mrpc", task="SequenceClassification"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if task == "SC":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "QA":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        elif task == "LM":
            self.model = AutoModelWithLMHead.from_pretrained(model_name)
        elif task == "TC":
            self.model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

    def SC(self, sent1, sent2):
        classes = ["not paraphrase", "is paraphrase"]
        paraphrase = self.tokenizer.encode_plus(sent1, sent2, return_tensors="pt")
        paraphrase_classification_logits = self.model(**paraphrase)[0]
        paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]

        print("Should be paraphrase")
        for i in range(len(classes)):
            print(f"{classes[i]}: {round(paraphrase_results[i] * 100)}%")
        return paraphrase_results

    def QA(self, question, context):
        inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        # text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = self.model(**inputs)

        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1
        # Get the most likely end of answer with the argmax of the score

        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        print(f"Question: {question}")
        print(f"Answer: {answer}\n")
        return answer

    def MLM(self, leftpart, rightpart):
        sequence = f"{leftpart} {self.tokenizer.mask_token} {rightpart}"
        input = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == self.tokenizer.mask_token_id)[1]

        token_logits = self.model(input)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]

        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

        for token in top_5_tokens:
            print(sequence.replace(self.tokenizer.mask_token, self.tokenizer.decode([token])))

    def CLM(self, sequence, max_length=50):
        input = self.tokenizer.encode(sequence, return_tensors="pt")
        generated = self.model.generate(input, max_length=max_length, do_sample=True)

        resulting_string = self.tokenizer.decode(generated.tolist()[0])
        # print(resulting_string)
        return resulting_string

    def NER(self, sequence):
        label_list = [
            "O",  # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",  # Beginning of a person's name right after another person's name
            "I-PER",  # Person's name
            "B-ORG",  # Beginning of an organisation right after another organisation
            "I-ORG",  # Organisation
            "B-LOC",  # Beginning of a location right after another location
            "I-LOC"  # Location
        ]

        # Bit of a hack to get the tokens with the special tokens
        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(sequence)))
        inputs = self.tokenizer.encode(sequence, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)
        outcomes = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]
        print(outcomes)
        return outcomes


if __name__ == '__main__':
    # # test paraphrase API
    # sequence_0 = "The company HuggingFace is based in New York City"
    # sequence_1 = "Apples are especially bad for your health"
    # sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
    # model_api = AutoTransformer_api(model_name="bert-base-cased-finetuned-mrpc", task="SC")
    # model_api.SC(sequence_0, sequence_2)
    # model_api.SC(sequence_0, sequence_1)
    #
    # # test QA API
    # context = r"""
    # 🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    # architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    # Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    # TensorFlow 2.0 and PyTorch.
    # """
    #
    # questions = [
    #     "How many pretrained models are available in Transformers?",
    #     "What does Transformers provide?",
    #     "Transformers provides interoperability between which frameworks?",
    # ]
    # model_api = AutoTransformer_api(model_name="bert-large-uncased-whole-word-masking-finetuned-squad", task="QA")
    # model_api.QA(questions[0], context)
    #
    # # test MLM API
    # left_seq = "Distilled models are smaller than the models they mimic. Using them instead of the large versions would help"
    # right_seq = "our carbon footprint."
    # model_api = AutoTransformer_api(model_name="distilbert-base-cased", task="LM")
    # model_api.MLM(left_seq, right_seq)

    # test CLM API
    sequence = "We love Pho"
    NLG = AutoTransformer_api(model_name="gpt2", task="LM")
    NLG.CLM(sequence, 100)

    # # test NER API
    # sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
    #            "close to the Manhattan Bridge."
    # model_api = AutoTransformer_api(model_name="bert-base-cased", task="TC")
    # model_api.NER(sequence)
