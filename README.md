# SpaCy Tech Review: Demo + Evaluation Template

This repo contains a ready-to-run script to reproduce the demo and lightweight evaluation for the SpaCy Tech Review.

## Contents
- `examples` contain the examples used in Tokenization, POS Tagging and NER as mentioned in the report.
- `texts` folder contains the texts from the [Plain text Wikipedia (SimpleEnglish) dataset available on kaggle](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish). `AllCombined_short` contains texts 600 - 1200 chars long and `AllCombined_medium.txt` contains texts 1500-2400 chars long.
- `results_short` contains the chosen 4 sampled short texts, along with their generated results.
- `results_medium` contains the chosen 4 sampled medium texts, along with their generated results.

## What it does
- Loads a small set of texts (from a CSV column, a plain text file, or fallback samples).
- Runs SpaCy pipelines (`en_core_web_sm`, `en_core_web_md`, `en_core_web_lg`) to demonstrate tokenization, POS, and NER.
- Measures runtime per model on the same texts and exports a CSV.
- (Optional) Evaluates NER against a tiny hand-labeled gold set and exports precision/recall/F1.
- Saves displaCy visualizations (entities + dependencies) as HTML files.

## Requirements
```bash
pip install spacy pandas
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

## Usage
### Runtime Evaluation
```bash
python spacy_tech_review_template.py \
  --texts_file data/simple_wiki_short.txt \
  --num_texts 4 \
  --repeat 3 \
  --outdir results_runtime/
```

### NER Accuracy Evaluation (with Gold Set)
```bash
python spacy_tech_review_template.py \
  --texts_file data/simple_wiki_short.txt \
  --gold_json data/tiny_gold.json \
  --outdir results_accuracy/
```

### Debugging Entity Spans (Gold vs Predicted)
Create a JSON like:
```bash
python spacy_tech_review_template.py \
  --texts_file data/simple_wiki_short.txt \
  --gold_json data/tiny_gold.json \
  --ner_debug \
  --outdir results_debug/
```

### Filter Evaluation by Specific Labels
```bash
python spacy_tech_review_template.py \
  --eval_labels GPE,DATE \
  --gold_json data/tiny_gold.json \
  --outdir results_filtered/
```

### Export displaCy Visualizations
```bash
python spacy_tech_review_template.py \
  --texts_file data/simple_wiki_medium.txt \
  --displacy_max -1 \
  --dep_all_sentences \
  --outdir results_displacy/
```

## Outputs
- `runtime_per_model.csv` — runtime benchmarks
- `ner_eval.csv` — macro precision/recall/F1 results
- `ner_debug_*.csv` — gold vs predicted span diagnostics
- `displacy/*.html` — entity & dependency visualizations
