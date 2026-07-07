import torch
import re
import copy
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize
from src.utils import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

OSU_AUTOAIS_MODEL = "osunlp/attrscore-flan-t5-xl"
claim_autoais_model = None
claim_autoais_tokenizer = None
input_prompt = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {claim}\n Reference: {output}"

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB - 6}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = dict.fromkeys(range(n_gpus), max_memory)
    return max_memory

def remove_citations(text):
    """
    Remove citation markers from text.

    Uses regex to detect and delete citation patterns such as [1] or [1, 2, 3].
    Also cleans up extra whitespace left after removal and fixes spacing before
    punctuation marks.

    Args:
        text: Input string containing citations.

    Returns:
        Cleaned text with all citations removed.

    Example:
        Input:  "AI is evolving rapidly [1, 2, 3], which will impact society."
        Output: "AI is evolving rapidly, which will impact society."
    """
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r"\[\d+(?:,\s*\d+)*\]"
    # Remove all citations from the text
    cleaned_text = re.sub(citation_pattern, "", text)
    # Optionally, remove extra spaces that might result from removing citations
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    cleaned_text = cleaned_text.replace(" .", ".")
    cleaned_text = cleaned_text.replace(" ,", ",")
    return cleaned_text

def extract_citations(text):
    """
    Extract citation numbers from text.

    Uses regex to find citation patterns like [1] or [1, 2, 3] and normalizes
    them into individual citation markers.

    Args:
        text: Input string containing citations.

    Returns:
        List of citation markers, each formatted as '[number]'.

    Example:
        Input:  "A citation [1, 2, 3] and another [4] in text."
        Output: ['[1]', '[2]', '[3]', '[4]']
    """
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
    # Find all matches in the text
    matches = re.findall(citation_pattern, text)
    # Extract individual numbers and convert them to integers
    citations = []
    for match in matches:
        # Split by commas, strip any extra whitespace, and convert to integers
        citations.extend([int(num.strip()) for num in match.split(",")])
    citations = [f"[{i}]" for i in citations]
    return citations

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    # global autoais_model, autoais_tokenizer
    # input_text = "premise: {} hypothesis: {}".format(passage, claim)
    # input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    # with torch.inference_mode():
    #     outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    # result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # inference = 1 if result == "1" else 0
    global claim_autoais_model, claim_autoais_tokenizer
    input_text = input_prompt.format_map({"output": passage, "claim": claim})
    input_ids = claim_autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(
        claim_autoais_model.device
    )
    with torch.inference_mode():
        outputs = claim_autoais_model.generate(input_ids, max_new_tokens=10)
    result = claim_autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if result == "Attributable":
        inference = 1.0
    else:
        inference = 0.0

    return inference

def compute_citation_f1(pred_answer, citations, at_most_citations=None):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
    """

    global claim_autoais_model, claim_autoais_tokenizer

    if claim_autoais_model is None:
        logger.info("Loading Claims AutoAIS model...")
        claim_autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            OSU_AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        claim_autoais_tokenizer = AutoTokenizer.from_pretrained(OSU_AUTOAIS_MODEL, use_fast=False)

    logger.info("Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: {}\n{}".format(doc["title"], doc["sent"])
        else:
            if "title" in doc:
                return "Title: {}\n{}".format(doc["title"], doc["text"])
            else:
                return doc["text"]

    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    cited_paper_total = []

    sents = sent_tokenize(pred_answer)
    if len(sents) == 0:
        return {
            "citation_rec": 0,
            "citation_prec": 0,
            "citation_f1": 0,
            "cited_paper_numbers": 0,
        }

    target_sents = [remove_citations(sent).strip() for sent in sents]

    cited_papers = set(extract_citations(pred_answer))
    cited_paper_total.append(len(cited_papers))

    entail = 0
    entail_prec = 0
    total_citations = 0
    total_sents = len(sents)
    previous_citations = None
    for sent_id, sent in enumerate(sents):
      target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
      joint_entail = -1  # Undecided

      # Find references
      ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
      logger.info(f"For `{sent}`, find citations {ref}")
      if len(ref) == 0 and previous_citations is not None:
          ref = previous_citations

      if len(ref) == 0:
          # No citations
          joint_entail = 0
      elif any(ref_id >= len(citations) for ref_id in ref):
          # Citations out of range
          joint_entail = 0
      else:
          previous_citations = ref
          if at_most_citations is not None:
              ref = ref[:at_most_citations]
          total_citations += len(ref)
          # print(item['docs'].keys())
          joint_passage = "\n".join(
              [_format_document(citations[psgs_id]["text"]) for psgs_id in ref if psgs_id >= 0]
          )

      # If not directly rejected by citation format error, calculate the recall score
      if joint_entail == -1:
          joint_entail = _run_nli_autoais(joint_passage, target_sent)
          autoais_log.append(
                {
                    "output": pred_answer,
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                }
            )

      entail += joint_entail
      if len(ref) > 1:
          sent_mcite += 1

      # calculate the precision score if applicable
      if joint_entail and len(ref) > 1:
          sent_mcite_support += 1
          # Precision check: did the model cite any unnecessary documents?
          for psgs_id in ref:
              # condition A
              passage = _format_document(citations[psgs_id]["text"])
              nli_result = _run_nli_autoais(passage, target_sent)

              # condition B
              if not nli_result:
                  subset_exclude = copy.deepcopy(ref)
                  subset_exclude.remove(psgs_id)
                  passage = "\n".join(
                      [_format_document(citations[pid]["text"]) for pid in subset_exclude]
                  )
                  nli_result = _run_nli_autoais(passage, target_sent)
                  if nli_result:  # psgs_id is not necessary
                      sent_mcite_overcite += 1
                  else:
                      entail_prec += 1
              else:
                  entail_prec += 1
      else:
          entail_prec += joint_entail

    if total_sents > 0:
        citation_rec = entail / total_sents
    else:
        citation_rec = 0

    citation_prec = entail_prec / total_citations if total_citations > 0 else 0

    return {
        "citation_rec": citation_rec,
        "citation_prec": citation_prec,
        "citation_f1": 2 * citation_rec * citation_prec / (citation_rec + citation_prec) if citation_rec + citation_prec > 0 else 0,
        "cited_paper_numbers": total_citations,
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Citation F1 Evaluation (AutoAIS)")
    parser.add_argument("--pred", type=str, required=True, help="Predicted answer with citations")
    parser.add_argument(
        "--citations",
        type=str,
        required=True,
        help="Path to JSON file with citations array [{\"title\": ..., \"text\": ...}]",
    )
    args = parser.parse_args()

    with open(args.citations, encoding="utf-8") as f:
        citations = json.load(f)

    try:
        result = compute_citation_f1(args.pred, citations)
    except Exception as e:
        result = {"error": str(e)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

