#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
import time
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import spacy
    from spacy.tokens import Doc
    from spacy.tokens import Span
    import pandas as pd
except Exception as e:
    spacy = None
    pd = None


def read_texts_from_csv(csv_path: str, text_column: str, num_texts: int, seed: int = 42) -> List[str]:
    if pd is None:
        raise RuntimeError("pandas is required. Please install with `pip install pandas`.")
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")
    # Filter to non-empty strings and reasonable length
    texts = [str(t) for t in df[text_column].dropna().tolist() if isinstance(t, str) and len(str(t).strip()) > 20]
    random.Random(seed).shuffle(texts)
    return texts[:num_texts]


def read_texts_from_file(txt_path: str, num_texts: int, seed: int = 42) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
    random.Random(seed).shuffle(blocks)
    # If the file has one text per line, fall back to lines
    if len(blocks) < num_texts:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        random.Random(seed).shuffle(lines)
        return lines[:num_texts]
    return blocks[:num_texts]


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_model(name: str):
    if spacy is None:
        raise RuntimeError("spaCy not available. Install with `pip install spacy`.")
    try:
        return spacy.load(name)
    except Exception as e:
        print(f"[WARN] Could not load model '{name}': {e}", file=sys.stderr)
        return None


def timed_process(nlp, text: str) -> Tuple[float, Any]:
    t0 = time.perf_counter()
    doc = nlp(text)
    dt = time.perf_counter() - t0
    return dt, doc


def run_timing(models: Dict[str, Any], texts: List[str], repeat: int = 1) -> pd.DataFrame:
    rows = []
    for model_name, nlp in models.items():
        if nlp is None:
            continue
        for idx, text in enumerate(texts):
            # Optionally repeat to smooth noise
            dts = []
            doc_last = None
            for _ in range(repeat):
                dt, doc_last = timed_process(nlp, text)
                dts.append(dt)
            row = {
                "model": model_name,
                "text_id": idx,
                "len_chars": len(text),
                "len_tokens": len(doc_last) if doc_last is not None else None,
                "elapsed_sec_avg": sum(dts) / len(dts)
            }
            rows.append(row)
    return pd.DataFrame(rows)


def export_displacy(models, texts, outdir, max_per_model=-1, dep_all_sentences=False):
    """
    Write displaCy HTML for entities (ent) for each text, and dependency (dep)
    for either the first sentence or all sentences (toggle with dep_all_sentences).

    max_per_model:
        -1  => export for ALL texts
         N  => export for the first N texts
    """
    from pathlib import Path
    from spacy import displacy

    disp_dir = Path(outdir) / "displacy"
    disp_dir.mkdir(parents=True, exist_ok=True)

    # how many texts per model
    n_texts = len(texts) if max_per_model == -1 else min(max_per_model, len(texts))

    for model_name, nlp in models.items():
        if nlp is None:
            continue
        for i, text in enumerate(texts[:n_texts]):
            doc = nlp(text)

            # ENTITIES HTML (full doc)
            ent_html = displacy.render(doc, style="ent", page=True)
            (disp_dir / f"{model_name}_text{i}_ents.html").write_text(ent_html, encoding="utf-8")

            # DEPENDENCIES HTML (first sentence or all)
            if dep_all_sentences:
                dep_docs = list(doc.sents) if list(doc.sents) else [doc[:]]
                dep_html = displacy.render(dep_docs, style="dep", page=True)
            else:
                first_sent = next(doc.sents, doc[:])
                dep_html = displacy.render(first_sent, style="dep", page=True)

            (disp_dir / f"{model_name}_text{i}_dep.html").write_text(dep_html, encoding="utf-8")


def iob_spans_from_doc(doc: Doc) -> List[Tuple[int, int, str]]:
    """Return entity spans as (start_char, end_char, label) from a Doc."""
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


def ner_prf(gold: List[Tuple[int, int, str]], pred: List[Tuple[int, int, str]]) -> Tuple[float, float, float]:
    """Exact-span match P/R/F1. Simple and strict. Label must match, spans must match exactly."""
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def load_gold(gold_json_path: str) -> List[Dict[str, Any]]:
    with open(gold_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    cleaned = []
    for s in samples:
        text = s["text"]
        ents = [(int(a), int(b), str(lbl)) for a, b, lbl in s.get("entities", [])]
        cleaned.append({"text": text, "entities": ents})
    return cleaned


def evaluate_ner(models: Dict[str, Any], gold_samples: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, nlp in models.items():
        if nlp is None:
            continue
        for i, sample in enumerate(gold_samples):
            text = sample["text"]
            gold_ents = sample["entities"]
            doc = nlp(text)
            pred_ents = iob_spans_from_doc(doc)
            p, r, f1 = ner_prf(gold_ents, pred_ents)
            rows.append({
                "model": model_name,
                "sample_id": i,
                "gold_count": len(gold_ents),
                "pred_count": len(pred_ents),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4)
            })
    return pd.DataFrame(rows)

def _label_filter_set(label_str: str):
    s = set(l.strip() for l in label_str.split(",") if l.strip())
    return s if s else None

def _span_text(text: str, start: int, end: int) -> str:
    if start < 0 or end > len(text) or start >= end:
        return f"<SPAN_OUT_OF_RANGE start={start} end={end} len={len(text)}>"
    return text[start:end]

def _norm(s: str) -> str:
    # normalize spaces to detect “same text, different offsets” bugs
    return re.sub(r"\s+", " ", s.strip())

def write_ner_debug_files(outdir: str, gold_samples: list, preds_by_model: dict, label_set=None):
    """
    gold_samples: [{"text": str, "entities": [[start,end,label], ...]}, ...]
    preds_by_model: { model_name: [ (doc, sample_id), ... ] }
    Writes:
      - ner_debug_gold.csv
      - ner_debug_predictions_<model>.csv
      - ner_debug_pairs_<model>.csv (gold vs pred with match flags)
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # GOLD
    gold_csv = out / "ner_debug_gold.csv"
    with gold_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "gold_start", "gold_end", "gold_label", "gold_text", "gold_text_repr", "full_text"])
        for sid, s in enumerate(gold_samples):
            text = s["text"]
            for (gs, ge, glabel) in s.get("entities", []):
                if label_set and glabel not in label_set:
                    continue
                gtxt = _span_text(text, gs, ge)
                w.writerow([sid, gs, ge, glabel, gtxt, repr(gtxt), text])

    # PREDICTIONS and PAIRS
    for model_name, items in preds_by_model.items():
        # flat predictions
        pred_csv = out / f"ner_debug_predictions_{model_name}.csv"
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "pred_start", "pred_end", "pred_label", "pred_text", "pred_text_repr", "full_text"])
            for (doc, sid) in items:
                text = doc.text
                for ent in doc.ents:
                    if label_set and ent.label_ not in label_set:
                        continue
                    ptxt = _span_text(text, ent.start_char, ent.end_char)
                    w.writerow([sid, ent.start_char, ent.end_char, ent.label_, ptxt, repr(ptxt), text])

        # pairwise: gold x pred with flags
        pairs_csv = out / f"ner_debug_pairs_{model_name}.csv"
        with pairs_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "sample_id",
                "gold_start","gold_end","gold_label","gold_text",
                "pred_start","pred_end","pred_label","pred_text",
                "exact_match","text_label_match","text_match_any_label","offset_differs"
            ])

            # index per sample
            gold_index = {}
            for sid, s in enumerate(gold_samples):
                g = []
                text = s["text"]
                for (gs, ge, glabel) in s.get("entities", []):
                    if label_set and glabel not in label_set:
                        continue
                    gtxt = _span_text(text, gs, ge)
                    g.append((gs, ge, glabel, gtxt))
                gold_index[sid] = (text, g)

            pred_index = {}
            for (doc, sid) in items:
                pred_index.setdefault(sid, [])
                for ent in doc.ents:
                    if label_set and ent.label_ not in label_set:
                        continue
                    ptxt = _span_text(doc.text, ent.start_char, ent.end_char)
                    pred_index[sid].append((ent.start_char, ent.end_char, ent.label_, ptxt))

            # emit paired rows
            for sid, (text, gspans) in gold_index.items():
                preds = pred_index.get(sid, [])
                if not preds and not gspans:
                    continue
                if not preds:  # no predictions, print gold with blanks
                    for (gs, ge, glab, gtxt) in gspans:
                        w.writerow([sid, gs, ge, glab, gtxt, "", "", "", "", 0, 0, 0, ""])
                    continue

                for (gs, ge, glab, gtxt) in gspans:
                    gnorm = _norm(gtxt)
                    any_row_written = False
                    for (ps, pe, plab, ptxt) in preds:
                        pnorm = _norm(ptxt)
                        exact = (gs == ps and ge == pe and glab == plab)
                        text_label = (gnorm == pnorm and glab == plab)
                        text_any = (gnorm == pnorm)
                        offset_differs = (text_label and not exact)
                        w.writerow([
                            sid, gs, ge, glab, gtxt,
                            ps, pe, plab, ptxt,
                            int(exact), int(text_label), int(text_any), ("1" if offset_differs else "")
                        ])
                        any_row_written = True
                    if not any_row_written:
                        w.writerow([sid, gs, ge, glab, gtxt, "", "", "", "", 0, 0, 0, ""])

def main():
    parser = argparse.ArgumentParser(description="SpaCy Tech Review: Demo + Evaluation Template")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--dataset_csv", type=str, help="Path to CSV (e.g., Kaggle Simple English Wikipedia)")
    parser.add_argument("--text_column", type=str, default="text", help="Text column in CSV")
    src.add_argument("--texts_file", type=str, help="Path to a .txt file (paragraphs separated by blank lines)")
    parser.add_argument("--num_texts", type=int, default=12, help="How many texts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat timings per text")
    parser.add_argument("--gold_json", type=str, help="Optional gold annotations for NER evaluation")
    parser.add_argument("--skip_displacy", action="store_true", help="Skip saving displaCy HTML files")
    parser.add_argument("--displacy_max", type=int, default=-1,
                    help="How many texts per model to render (-1 = all).")
    parser.add_argument("--dep_all_sentences", action="store_true",
                    help="If set, render dependency trees for all sentences, not just the first.")
    parser.add_argument("--eval_labels", type=str, default="",
                    help="Comma-separated labels to include in NER eval (e.g., 'GPE,DATE'). Empty = all.")
    parser.add_argument("--ner_debug", action="store_true",
                    help="If set, write gold and predicted spans (start,end,label,text) for each sample.")

    args = parser.parse_args()

    ensure_outdir(args.outdir)
    
    # Load texts
    if args.dataset_csv:
        texts = read_texts_from_csv(args.dataset_csv, args.text_column, args.num_texts, seed=args.seed)
    elif args.texts_file:
        texts = read_texts_from_file(args.texts_file, args.num_texts, seed=args.seed)
    else:
        # Fallback texts (short, varied)
        texts = [
            "Barack Obama was the 44th President of the United States and was born in Hawaii in 1961.",
            "Apple Inc. announced new iPhone models in Cupertino on September 12, 2018, attracting global media attention.",
            "The city of Paris, located on the River Seine, is the capital of France and a major European center of art and fashion.",
            "The FIFA World Cup is an international soccer tournament organized by FIFA and contested by national teams every four years.",
            "In physics, quantum entanglement describes a physical phenomenon in which the quantum states of two or more particles become correlated.",
            "Tesla opened a new Gigafactory in Shanghai, expanding its manufacturing capacity in Asia and reducing shipping times.",
            "Marie Curie won the Nobel Prize in Chemistry in 1911 for her work on radioactivity.",
            "Mount Everest, part of the Himalayas, lies on the border of Nepal and China and is the Earth's highest mountain above sea level."
        ][:args.num_texts]

    # Load models
    model_names = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    models = {m: load_model(m) for m in model_names}
    available = [k for k, v in models.items() if v is not None]
    if not available:
        print("[ERROR] No SpaCy models could be loaded. Install models with:", file=sys.stderr)
        print("  python -m spacy download en_core_web_sm", file=sys.stderr)
        print("  python -m spacy download en_core_web_md", file=sys.stderr)
        print("  python -m spacy download en_core_web_lg", file=sys.stderr)
        sys.exit(1)

    # Timing
    timing_df = run_timing(models, texts, repeat=args.repeat)
    timing_csv = os.path.join(args.outdir, "timing_results.csv")
    if pd is None:
        raise RuntimeError("pandas is required to export results. Install with `pip install pandas`.")
    timing_df.to_csv(timing_csv, index=False)
    print(f"[OK] Wrote timing results to {timing_csv}")

    # displaCy HTML
    if not args.skip_displacy:
        displacy_dir = os.path.join(args.outdir, "displacy")
        export_displacy(models, texts, outdir=args.outdir,
                    max_per_model=args.displacy_max,
                    dep_all_sentences=args.dep_all_sentences)
        print(f"[OK] Wrote displaCy HTML to {displacy_dir}")

    # Optional NER evaluation
    if args.gold_json:
        gold_samples = load_gold(args.gold_json)
        ner_df = evaluate_ner(models, gold_samples)
        ner_csv = os.path.join(args.outdir, "ner_eval.csv")
        ner_df.to_csv(ner_csv, index=False)
        print(f"[OK] Wrote NER evaluation to {ner_csv}")
        
    label_set = _label_filter_set(args.eval_labels)
    preds_by_model = {m: [] for m in models if models[m] is not None}
    for model_name, nlp in models.items():
        if nlp is None:
            continue
        # run in the same order as gold_samples to keep sample_id aligned
        for sid, s in enumerate(gold_samples):
            doc = nlp(s["text"])
            preds_by_model[model_name].append((doc, sid))
            # your existing scoring logic (precision/recall/F1) stays the same here
    # After scoring is done:
    if args.ner_debug:
        write_ner_debug_files(args.outdir, gold_samples, preds_by_model, label_set=label_set)
        print(f"[NER DEBUG] Wrote gold/pred debug CSVs to {args.outdir}")

    # Also export a quick preview of the sampled texts for citation
    sampled_path = os.path.join(args.outdir, "sampled_texts.txt")
    with open(sampled_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(f"--- SAMPLE {i} ---\n")
            f.write(t.strip() + "\n\n")
    print(f"[OK] Wrote sampled texts to {sampled_path}")


if __name__ == "__main__":
    main()
