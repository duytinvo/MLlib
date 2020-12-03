"""
Created on 2019-02-20
@author: duytinvo
"""
import argparse
from mlmodels.inference.trans_language_serve import LM


def load_model(args):
    model_api = LM(args)
    return model_api


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, help="Model type selected in the list ")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/media/data/review_response/trained_model/checkpoint_by-step_175000/",
                        help="Path to pre-trained model or shortcut name selected in the list")

    parser.add_argument("--prompt", type=str,
                        default="<SSP> 1 "
                                "<SSP> the gates hotel south beach - a doubletree by hilton "
                                "<SSP> mrs.m "
                                "<SSP> nasty "
                                "<SSP> this has to be the nastiest hotel ever. "
                                "<SSP>")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_false", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    args = parser.parse_args()
    lm_api = LM(args)

    generated_sequences = lm_api.inference(
                     rv_hotel="hyatt regency calgary",
                     rv_name="tin",
                     rv_rate="1",
                     rv_title="unworthy",
                     rv_text="Service was really good. Staff members were friendly and helpful. "
                             "Was not the cleanest hotel. For this price I can find a much cleaner and better hotel. "
                             "Paid parking for $30 sucks, paid breakfast for $40 sucks, rooms were too small. "
                             "View was good from the rooms. Uneven beds, got neck pain next day when I realized it. "
                             "Overall, will never go to them again.")
