"""OptiS Benchmark - Paper Extraction Evaluation Pipeline."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithm.bertScore_eval_utils import compute_bert_score
from src.algorithm.bleu_eval_utils import compute_bleu
from src.algorithm.cider_eval_utils import compute_cider
from src.algorithm.edit_distance_utils import normalized_edit_similarity
from src.algorithm.em_eval_utils import compute_exact_match, normalize_text, record_doi_punctuation
from src.algorithm.hungarian_algorithm_utils import hungarian_match
from src.algorithm.jaccard_similarity_utils import jaccard_similarity
from src.algorithm.meteor_eval_utils import compute_meteor
from src.algorithm.perplexity_eval_utils import compute_perplexity
from src.algorithm.rouge_eval_utils import compute_rouge
from src.algorithm.sentence_similarity_utils import SentenceEmbedder

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


def _score_str_entry(
    pred_answer: str,
    gold_answer: str,
    entry: str,
    args: argparse.Namespace,
) -> dict:
    """Compute all enabled lexical/string-level scores for a (pred, gold) pair.

    This applies to entries whose value is a plain string (e.g. title, doi).
    """
    scores: dict[str, float] = {}

    if args.match:
        scores["exact_match"] = compute_exact_match(pred_answer, gold_answer)

        if entry.lower() == "doi":
            pred_doi_punct = record_doi_punctuation(pred_answer)
            gold_doi_punct = record_doi_punctuation(gold_answer)
            for punct, pred_indices in pred_doi_punct.items():
                gold_indices = gold_doi_punct.get(punct, [])
                if Counter(pred_indices) != Counter(gold_indices):
                    scores["exact_match"] = 0.0
                    break

    if args.bleu:
        result = compute_bleu(pred_answer, [gold_answer])
        scores["bleu"] = result["bleu"]

    if args.edit_distance:
        scores["edit_similarity"] = normalized_edit_similarity(pred_answer, gold_answer)

    if args.jaccard:
        scores["jaccard"] = jaccard_similarity(pred_answer, gold_answer)

    if args.rouge:
        scores["rouge"] = compute_rouge(pred_answer, [gold_answer])

    if args.bertScore:
        bs = compute_bert_score(pred_answer, [gold_answer], model_name=args.bertScore_model)
        scores["bertScore"] = bs["f1"]

    if args.perplexity:
        scores["perplexity"] = compute_perplexity(pred_answer)["perplexity"]

    if args.meteor:
        scores["meteor"] = compute_meteor(pred_answer, [gold_answer])["meteor"]

    if args.cider:
        scores["cider"] = compute_cider(pred_answer, [gold_answer])["cider"]

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate paper extraction predictions against gold references."
    )
    parser.add_argument("--pred-file", type=str, required=True, help="Prediction JSON file")
    parser.add_argument("--gold-file", type=str, required=True, help="Gold reference JSON file")

    # Metric flags
    parser.add_argument("--match", action="store_true", help="Evaluate exact match score")
    parser.add_argument("--rouge", action="store_true", help="Evaluate ROUGE-L score")
    parser.add_argument("--bleu", action="store_true", help="Evaluate BLEU score")
    parser.add_argument("--edit-distance", action="store_true", help="Evaluate edit distance similarity")
    parser.add_argument("--jaccard", action="store_true", help="Evaluate Jaccard similarity")
    parser.add_argument("--bertScore", action="store_true", help="Evaluate BERTScore")
    parser.add_argument("--perplexity", action="store_true", help="Evaluate perplexity (requires GPT-2 / causal LM)")
    parser.add_argument("--meteor", action="store_true", help="Evaluate METEOR score (requires WordNet)")
    parser.add_argument("--cider", action="store_true", help="Evaluate CIDEr score")
    parser.add_argument(
        "--bertScore-model",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model name (default: microsoft/deberta-xlarge-mnli)",
    )

    args = parser.parse_args()

    with open(args.pred_file) as f:
        pred_data = json.load(f)
    with open(args.gold_file) as f:
        gold_data = json.load(f)

    # Embedder for non-string entries (multi-sentence / list fields)
    embedder = SentenceEmbedder() if (args.bertScore or args.rouge or args.cider) else None

    # Accumulators
    metric_accums: dict[str, list[float]] = {}

    for idx, pred in enumerate(pred_data):
        gold = gold_data[idx]

        for entry, pred_answer in pred:
            gold_answer = gold[entry]
            norm_entry = normalize_text(entry)

            if norm_entry in ENTRY_NAMES:
                if isinstance(pred_answer, str) and isinstance(gold_answer, str):
                    scores = _score_str_entry(pred_answer, gold_answer, entry, args)
                    for key, val in scores.items():
                        metric_accums.setdefault(key, []).append(val)
                elif isinstance(pred_answer, list) and isinstance(gold_answer, list):
                    if args.match:
                        pred_norm = [normalize_text(p) for p in pred_answer]
                        gold_norm = [normalize_text(g) for g in gold_answer]
                        metric_accums.setdefault("exact_match", []).append(
                            1.0 if Counter(pred_norm) == Counter(gold_norm) else 0.0
                        )
            else:
                # Non-ENTRY_NAMES: multi-sentence fields → use Hungarian matching
                if args.rouge or args.bertScore or args.cider:
                    continue  # skip Hungarian for these; handled elsewhere

                if embedder is None:
                    continue

                pred_embs = embedder.encode(pred_answer)
                gold_embs = embedder.encode(gold_answer)
                sim_matrix = np.dot(pred_embs, gold_embs.T).astype(np.float32)
                assignments, _ = hungarian_match(sim_matrix)

                for pred_idx, gold_idx in assignments:
                    matched_pred = pred_answer[pred_idx]
                    matched_gold = gold_answer[gold_idx]

                    if args.rouge:
                        metric_accums.setdefault("rouge", []).append(
                            compute_rouge(matched_pred, [matched_gold])
                        )
                    if args.bertScore:
                        bs = compute_bert_score(
                            matched_pred, [matched_gold], model_name=args.bertScore_model
                        )
                        metric_accums.setdefault("bertScore", []).append(bs["f1"])

    # Aggregate and print results
    if not metric_accums:
        print("{}")
        return

    results = {key: round(float(np.mean(vals)), 4) for key, vals in metric_accums.items()}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
