import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

def score(sentence, model, tokenizer):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss, *_ = model(tensor_input, labels=tensor_input)
    return math.exp(loss)


def main(sentences, model_name='gpt2-large'):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, do_lower_case=True)

    avg_prep = 0
    for sentence in tqdm(sentences):
        avg_prep += score(sentence, model, tokenizer)
    return avg_prep/len(sentences)

if __name__ == '__main__':
    sentences = ["Hi, How are you?"]
    print(main(sentences))
