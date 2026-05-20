import argparse
import json
from collections import Counter

import numpy as np
from utils.em_eval_utils import compute_exact_match, normalize_text, record_doi_punctuation
from utils.rouge_eval_utils import compute_rouge

extract_entrys = [
    "title",
    "publication year",
    "doi",
    "journal",
    "authors",
    "contact author",
    "affiliations",
    "abstract",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-file",
        type=str,
        required=True,
        help="Prediction file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`",
    )
    parser.add_argument(
        "--gold-file",
        type=str,
        required=True,
        help="Gold file. Should have field `question`, `answer`, (accuracy) `qa_pairs`, (AIS) `docs`",
    )
    parser.add_argument("--rouge", action="store_true", help="Evaluate ROUGE score")
    parser.add_argument("--match", action="store_true", help="Evaluate exact match score")
    parser.add_argument(
        "--bertScore-model",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model name (default: microsoft/deberta-xlarge-mnli)",
    )

    args = parser.parse_args()

    if args.pred_file.endswith(".json"):
        with open(args.pred_file) as f:
            pred_data = json.load(f)
    if args.gold_file.endswith(".json"):
        with open(args.gold_file) as f:
            gold_data = json.load(f)

    if args.match:
        exact_match_scores = []
        for idx, pred in enumerate(pred_data):
            exact_match = []
            rouge_scores = []
            gold = gold_data[idx]
            for entry, pred_answer in pred:
                gold_answer = gold[entry]
                if normalize_text(entry) in extract_entrys:
                    if isinstance(pred_answer, str) and isinstance(gold_answer, str):
                        match_result = compute_exact_match(pred_answer, gold_answer)
                        if match_result:
                            if entry.lower() == "doi":
                                match_flag = True
                                pred_doi_punct = record_doi_punctuation(pred_answer)
                                gold_doi_punct = record_doi_punctuation(gold_answer)
                                for pred_punct, pred_indices in pred_doi_punct.items():
                                    gold_indices = gold_doi_punct.get(pred_punct, [])
                                    if Counter(pred_indices) != Counter(gold_indices):
                                        exact_match.append(0.0)
                                        match_flag = False
                                        break
                                if match_flag:
                                    exact_match.append(1.0)
                            else:
                                exact_match.append(1.0)
                        else:
                            exact_match.append(0.0)
                    elif isinstance(pred_answer, list) and isinstance(gold_answer, list):
                        pred_nomalized_answer = [normalize_text(p) for p in pred_answer]
                        gold_nomalized_answer = [normalize_text(g) for g in gold_answer]
                        if Counter(pred_nomalized_answer) == Counter(gold_nomalized_answer):
                            exact_match.append(1.0)
                        else:
                            exact_match.append(0.0)
                else:
                    temp_rouge_scores = []
                    for p in pred_answer:
                        rouge_score, reference_idx = compute_rouge(p, gold_answer)
                        temp_rouge_scores.append(rouge_score)
                        del gold_answer[reference_idx]
                    rouge_scores.append(np.mean(temp_rouge_scores))
            exact_match_scores.append(np.mean(exact_match))
            rouge_scores.append(np.mean(rouge_scores))


if __name__ == "__main__":
    main()
