# -*- coding: utf-8 -*-
"""
Created on 2020-04-03
@author duytinvo
"""
import argparse
import logging
import numpy as np
import torch
import os
import re
import textwrap
from mlmodels.training.trans_language_model import TransLanguageModel
from mlmodels.utils.idx2tensor import Data2tensor
from mlmodels.utils.special_tokens import SENSP, SENGE, SOT, EOT
from mlmodels.utils.jsonIO import JSON


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """<s> chata.ai is a startup company that is doing AI used in e-commerce </s>"""
# Functions to prepare models' input


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class LM(object):
    def __init__(self, args):
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        self.args = args
        margs = torch.load(os.path.join(args.model_name_or_path, "training_args.bin"))
        margs.no_cuda = args.no_cuda
        margs.model_name_or_path = args.model_name_or_path
        margs.overwrite_output_dir = True
        self.lm = TransLanguageModel(margs)
        self.lm.model_init()
        # lm.load_model(args.model_name_or_path)
        Data2tensor.set_randseed(args.seed)

    @staticmethod
    def prepare_entry(task, rv_text, rv_rate, rv_name, rv_title, rv_hotel, rouge_score=None):
        if task == "sentiment":
            prompt_text = rv_text
        elif task == "paraphrase":
            prompt_text = rv_text
        else:
            if rouge_score is not None:
                # prompt_text = " ".join([rouge_score, SENSP, rv_rate.lower(), SENSP, rv_hotel.lower(), SENSP,
                #                         rv_name.lower(), SENSP, rv_title.lower(), SENSP, rv_text.lower(), SENGE])
                prompt_text = " ".join([rouge_score, rv_hotel.lower(), rv_name.lower(), rv_text.lower(), SENGE])
            else:
                # prompt_text = " ".join([rv_rate.lower(), SENSP, rv_hotel.lower(), SENSP, rv_name.lower(), SENSP,
                #                         rv_title.lower(), SENSP, rv_text.lower(), SENGE])
                prompt_text = " ".join([rv_hotel.lower(), rv_name.lower(), rv_text.lower(), SENGE])
        prompt_text = JSON.process_nl(prompt_text)
        return prompt_text

    def input2tensor(self, prompt_text):
        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = self.args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(self.args.model_type)
            preprocessed_prompt_text = prepare_input(self.args, self.lm.model, self.lm.tokenizer, prompt_text)
            # encoded_prompt = lm.tokenizer.encode(preprocessed_prompt_text, add_special_tokens=False,
            #                                      return_tensors="pt", add_space_before_punct_symbol=True)
            encoded_prompt = self.lm.collate([[self.lm.source2idx(preprocessed_prompt_text), []]])[0]
        else:
            # encoded_prompt = lm.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = self.lm.collate([[self.lm.source2idx(prompt_text), []]])[0]
        encoded_prompt = encoded_prompt.to(self.args.device)
        return encoded_prompt

    def inference(self, task, rv_text, rv_rate="", rv_name="", rv_title="", rv_hotel="", rouge_score=None):
        prompt_text = LM.prepare_entry(task, rv_text, rv_rate, rv_name, rv_title, rv_hotel, rouge_score=rouge_score)
        encoded_prompt = self.input2tensor(prompt_text)
        if task == "sentiment":
            self.lm.model.eval()
            with torch.no_grad():
                output_sequences = self.lm.model(input_ids=encoded_prompt,
                                                 decoder_input_ids=self.lm.model._shift_right(encoded_prompt[:, 0].reshape(-1, 1)))
                logits = output_sequences[0]
                norm_logits = torch.softmax(logits, dim=-1)
                pred_p, pred_id = norm_logits.max(dim=-1)
                pred_l = self.lm.tokenizer.decode(pred_id.item())
                return pred_l, pred_p.item()
        else:
            length = self.args.length if self.lm.args.model_type == "t5" else self.args.length + len(encoded_prompt[0])
            output_sequences = self.lm.model.generate(
                input_ids=encoded_prompt,
                max_length=length,
                temperature=self.args.temperature,
                top_k=self.args.k,
                top_p=self.args.p,
                repetition_penalty=self.args.repetition_penalty,
                num_beams=self.args.num_beams,
                do_sample=self.args.do_sample,
                num_return_sequences=self.args.num_return_sequences,
            )

            # Remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = self.lm.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True,
                                                skip_special_tokens=True)

                # Remove all text after the stop token
                # gen_text = text[: text.find(self.args.stop_token) if self.args.stop_token else None]
                gen_text = text[: text.find(self.lm.tokenizer.eos_token) if text.find(self.lm.tokenizer.eos_token) != -1 else None]

                if self.lm.args.model_type != "t5":
                    gen_text = gen_text[len(self.lm.tokenizer.decode(encoded_prompt[0],
                                                                     clean_up_tokenization_spaces=True,
                                                                     skip_special_tokens=True)):]
                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (prompt_text, gen_text)
                generated_sequences.append(total_sequence)
                # print("".join(total_sequence))
            return generated_sequences

    @staticmethod
    def post_process_string(s):
        # s = re.sub(r'\<extra\_id\_\d+\>', '', s)
        s = re.sub(r'\<\w+\>', '', s)
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'\\n ', '\n', s)
        return s

    @staticmethod
    def pretty_print(s, width=80):
        textLines = s.split('\n')
        wrapedLines = []
        # Preserve any indent (after the general indent)
        _indentRe = re.compile('^(\W+)')
        for line in textLines:
            preservedIndent = ''
            existIndent = re.search(_indentRe, line)
            # Change the existing wrap indent to the original one
            if existIndent:
                preservedIndent = existIndent.groups()[0]
            wrapedLines.append(textwrap.fill(line, width=width, subsequent_indent=preservedIndent))
        s = '\n'.join(wrapedLines)
        return s

    def predict_batch(self):
        pass

    def batch_inference(self):
        pass

    def regression_test(self, pfile, test_file, limit=None, batch_size=8):
        pass


if __name__ == "__main__":
    from mlmodels.utils.csvIO import CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="paraphrase", type=str, choices=["response", "sentiment", "paraphrase"],
                        help="Select a task to solve", )
    parser.add_argument("--model_type", default=None, type=str, help="Model type selected in the list ")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/media/data/paraphrase/trained_model/checkpoint_by-epoch_1/",
                        help="Path to pre-trained model or shortcut name selected in the list")

    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="temperature of 1.0 has no effect, lower tend toward greedy sampling", )
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_false", help="Avoid using CUDA when available")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling multinomial technique")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="The number of samples to generate.")
    parser.add_argument("--num_beams", type=int, default=10, help="The number of beams to generate.")
    parser.add_argument('--test_file', help='Test file', type=str,
                        default="/media/data/review_response/customer_data/NH_Collection_Guadalajara_Centro_Historico.csv")

    args = parser.parse_args()
    lm_api = LM(args)

    if args.task == "paraphrase":
        question = "what is the price of GOOG"
        generated_sequences = lm_api.inference(task=args.task,
                                               rv_text=question)
        responses = []
        for generated_sequence in generated_sequences:
            responses += [lm_api.post_process_string(generated_sequence[-1])]
            # print(lm_api.pretty_print(lm_api.post_process_string(generated_sequence[-1]), width=180))
        # print(lm_api.pretty_print(lm_api.post_process_string(generated_sequences[-1]), width=180))
    elif args.task == "sentiment":
        review = "it is so delicious"
        generated_sequences = lm_api.inference(task=args.task,
                                               rv_text=review)
    else:
        data = CSV.read(args.test_file, firstline=True, slices=[0, 1, 2, 3, 4, 5])
        reviews = [d[0] for d in data]

        # rv_hotel = "Woodbridge Ford"
        # rv_name = "guest"
        # rv_rate = ""
        # rv_title = ""
        # rv_text = "Great staff awesome inventory. See Brock Dennison and team Atlas for the best deals and over the top customer service. " \
        #           "Add in the fact that they care and support the community in so many ways. " \
        #           "You definitely need to go see the great team at Woodbridge Ford."
        rouge_score = "0.3500"
        responses = []
        for d in data:
            rv_text = d[0]
            rv_rate = d[1]
            rv_title = d[2]
            rv_name = d[3]
            rv_hotel = d[4]

            generated_sequences = lm_api.inference(rv_hotel=rv_hotel, rv_name=rv_name, rv_rate=rv_rate, rv_title=rv_title,
                                                   rv_text=rv_text, rouge_score=rouge_score)
            # responses.append(generated_sequences[0][-1])
            rp = []
            for generated_sequence in generated_sequences:
                rp += [lm_api.post_process_string(generated_sequence[-1])]
                # print(lm_api.pretty_print(lm_api.post_process_string(generated_sequence[-1]), width=180))
            responses.append(rp)

        write_data = []
        for i in range(len(data)):
            write_data.append(data[i] + responses[i])
        CSV.write(write_data, "/media/data/review_response/customer_data/response_NH_Collection_Guadalajara_Centro_Historico.csv")



        # generated_sequences = lm_api.inference(rv_hotel="Le Germain Hotel Calgary", rv_name="tinvo", rv_rate="1",
        #                                        rv_title="disappointing",
        #                                        rv_text="Incredibly disappointing staff, there was no service to be found at the bar. After hunting down the concierge, he was of no help either."
        #                                                " The lack of service was unexpected and jarring. Won’t be coming back anytime soon. Thoroughly disappointed.",
        #                                        rouge_score="0.4000")
        # print(lm_api.pretty_print(lm_api.post_process_string(generated_sequence[-1]), width=180))


