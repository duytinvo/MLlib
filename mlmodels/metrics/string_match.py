import copy
import csv
import os
"""
This module computes the exact string-match score for a set of translated segmdnts against references
"""


def compute_string_match_score(reference_corpus, translation_corpus, nl_tokens, pred_file):
    """Computes string match score of translated segments against references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
    Returns:
        Proportion of exact string matches
    """
    string_match_score = 0
    total_samples = len(translation_corpus)
    if not os.path.exists(os.path.dirname(pred_file)):
        os.mkdir(os.path.dirname(pred_file))
    with open(pred_file, "w") as f:
        csvwriter = csv.writer(f)
        hearder = ["question", "query", "predicted_query", "matching"]
        csvwriter.writerow(hearder)
        for (references, translation, nl) in zip(reference_corpus, translation_corpus, nl_tokens):
            if references[0] == translation:  # Assuming only one reference
                string_match_score += 1
                # row = [" ".join(nl), " ".join(references[0]), " ".join(translation), 'TRUE']
            else:
                row = [" ".join(nl), " ".join(references[0]), " ".join(translation), 'FALSE']
                csvwriter.writerow(row)
    string_match_score = string_match_score / total_samples
    return string_match_score


def compute_string_match(reference_corpus, translation_corpus):
    """Computes string match score of translated segments against references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
    Returns:
        Proportion of exact string matches
    """
    string_match_score = 0
    total_samples = len(translation_corpus)
    for (references, translation) in zip(reference_corpus, translation_corpus):
        if " ".join(references).strip() == " ".join(translation).strip():  # Assuming only one reference
            string_match_score += 1
            # row = [" ".join(nl), " ".join(references[0]), " ".join(translation), 'TRUE']
    string_match_score = string_match_score / total_samples
    return string_match_score
