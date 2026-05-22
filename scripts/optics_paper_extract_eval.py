import argparse
import json
from collections import Counter

import numpy as np
from utils.em_eval_utils import compute_exact_match, normalize_text, record_doi_punctuation
from utils.rouge_eval_utils import compute_rouge
from utils.sentence_similarity_utils import SentenceEmbedder
from utils.hungarian_algorithm_utils import hungarian_match
from utils.bertScore_eval_utils import compute_bert_score
ENTRY_NAMES = frozenset({
    "title",
    "publication year",
    "doi",
    "journal",
    "authors",
    "contact author",
    "affiliations",
    "abstract",
})


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
    parser.add_argument("--bertScore", action="store_true", help="Evaluate BERTScore score")
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

    if args.bertScore or args.rouge:
        embedder = SentenceEmbedder()

    all_scores = {}
    if args.match:
        exact_match_scores = []
        rouge_score_all = []
        bert_score_all = []
        for idx, pred in enumerate(pred_data):
            exact_match = []
            rouge_scores = []
            bert_scores = []
            gold = gold_data[idx]
            for entry, pred_answer in pred:
                gold_answer = gold[entry]
                norm_entry = normalize_text(entry)
                if norm_entry in ENTRY_NAMES:
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
                        pred_normalized_answer = [normalize_text(p) for p in pred_answer]
                        gold_normalized_answer = [normalize_text(g) for g in gold_answer]
                        if Counter(pred_normalized_answer) == Counter(gold_normalized_answer):
                            exact_match.append(1.0)
                        else:
                            exact_match.append(0.0)
                else:
                    if args.rouge or args.bertScore:
                        continue
                    pred_embs = embedder.encode(pred_answer)
                    gold_embs = embedder.encode(gold_answer)
                    sim_matrix = np.dot(pred_embs, gold_embs.T).astype(np.float32)
                    assignments, total_score = hungarian_match(sim_matrix)
                    temp_rouge_scores = []
                    temp_bert_scores = []
                    for pred_idx, gold_idx in assignments:
                        if args.rouge:
                            rouge_score = compute_rouge(pred_answer[pred_idx], gold_answer[gold_idx])
                            temp_rouge_scores.append(rouge_score)
                        if args.bertScore:
                            bert_score = compute_bert_score(
                                pred_answer[pred_idx],
                                gold_answer[gold_idx],
                                model_name=args.bertScore_model,
                            )
                            temp_bert_scores.append(bert_score['f1'])
                    if temp_rouge_scores:
                        rouge_scores.append(np.mean(temp_rouge_scores))
                    if temp_bert_scores:
                        bert_scores.append(np.mean(temp_bert_scores))

            exact_match_scores.append(np.mean(exact_match))
            if rouge_scores:
                rouge_score_all.append(np.mean(rouge_scores))
            if bert_scores:
                bert_score_all.append(np.mean(bert_scores))

        all_scores["exact_match"] = np.mean(exact_match_scores) if exact_match_scores else 0.0
        all_scores["rouge"] = np.mean(rouge_score_all) if rouge_score_all else 0.0
        all_scores["bertScore"] = np.mean(bert_score_all) if bert_score_all else 0.0
    


if __name__ == "__main__":
    main()
