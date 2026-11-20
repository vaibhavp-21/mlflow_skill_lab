# Champion-Challenger Model Comparison for Emotion Classification

## Student Details

- **Name:** Vaibhav Patil
- **USN:** 1MS25CI134
- **Institution:** Ramaiah Institute of Technology, Bengaluru
- **Department:** CSE (AI & ML)
- **Course:** MLOps with MLflow

## Abstract and Objective

This individual mini-project evaluates two pretrained Hugging Face models on exactly the same reproducible sample from the official test split of `dair-ai/emotion`. It tracks evaluation and packaged models in MLflow, chooses a Champion using a rule declared before evaluation, registers both versions, assigns `champion` and `challenger` aliases, and uses the Champion alias for predictions on new text.

This is six-class emotion classification, not binary sentiment analysis. The classes are Sadness, Joy, Love, Anger, Fear, and Surprise.

## Dataset and Label Compatibility

The project loads `dair-ai/emotion`, configuration `split`, split `test` with Hugging Face Datasets. The test split has `text` and `label` fields; `label` is a six-class `ClassLabel`. The script selects 200 stratified examples with seed 42, then writes the shared inputs and reference labels to `data/eval_inputs.csv` before model evaluation.

Both requested model configurations are inspected at runtime. Their six label names are compared with the dataset label names. Predicted IDs are translated using each model's own `id2label` configuration and then aligned by label name; the implementation does not rely on equal numeric ID ordering.

## Models

- **Model A, baseline:** `bhadresh-savani/distilbert-base-uncased-emotion` (DistilBERT).
- **Model B, challenger:** `bhadresh-savani/bert-base-uncased-emotion` (BERT).

DistilBERT is a compressed, smaller architecture that typically needs less memory and has lower inference latency. BERT has more parameters and may provide stronger predictive performance at the cost of model size and speed. Those are expected trade-offs, not results: the actual Accuracy, Macro-F1, file sizes, and measured latency are reported below only after the evaluation script runs. Macro-F1 gives each emotion equal weight, which is useful when class frequencies differ; Weighted-F1 weights classes by their support.

## Predeclared Champion Selection Rule

This rule is fixed before model evaluation and is implemented unchanged in `evaluate.py`:

- Primary metric: Macro-F1 (higher is better).
- Minimum improvement margin: 0.01.
- Latency guardrail: Model B's average inference latency must be no more than five times Model A's.
- Model B becomes Champion only if its Macro-F1 improves over Model A by at least 0.01 **and** it satisfies the latency guardrail. Otherwise, Model A remains Champion.

The latency ratio is Model B average latency divided by Model A average latency. Average latency is total timed inference duration divided by the number of examples; model download and initialization are not included.

## Installation and Execution (Windows PowerShell)

Use Python 3.10, 3.11, or 3.12. Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the MLflow tracking server in a separate PowerShell window from this folder and leave it running:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Run the evaluation, register both models and aliases, then predict with the Champion:

```powershell
.\.venv\Scripts\Activate.ps1
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
```

The first evaluation downloads the dataset and model files to the standard Hugging Face cache. Inference uses CPU when CUDA is unavailable, with batch size 16 and maximum sequence length 128. MLflow UI: <http://127.0.0.1:5000>.

## Scripts and Outputs

- `evaluate.py` selects and saves the shared stratified sample, inspects model label mappings, evaluates both models, logs two model runs and one comparison run, and produces metrics, prediction CSVs, plots, and disagreement examples.
- `emotion_model.py` packages a local copy of each fine-tuned model and tokenizer as an MLflow PyFunc model. This lets MLflow load the model without a Hugging Face download in the final prediction script.
- `register_champion.py` registers both logged models under `emotion-classifier`, then assigns `champion` and `challenger` aliases according to the comparison result.
- `predict_from_mlflow.py` loads `models:/emotion-classifier@champion` through MLflow PyFunc and writes the new-data predictions.
- `MLproject` provides an optional MLflow Projects entry point for evaluation.

`data/new_data.csv` contains eight examples intended to be distinct from the evaluation sample. The prediction script checks for exact text overlap before inference.

## Evaluation Results

Results are not filled in until `evaluate.py` has run successfully; no metrics or model outcome are assumed in advance.

| Metric | Model A: DistilBERT | Model B: BERT |
|---|---:|---:|
| Accuracy | 0.9400 | 0.9150 |
| Macro Precision | 0.9126 | 0.8870 |
| Macro Recall | 0.9012 | 0.8595 |
| Macro-F1 | 0.9058 | 0.8703 |
| Weighted-F1 | 0.9394 | 0.9135 |
| Average latency (ms/example) | 89.45 | 116.69 |
| Total inference time (seconds) | 17.889 | 23.337 |
| Packaged model size (MB, weights + tokenizer) | 256.34 | 418.58 |

**Final Champion:** Model A, DistilBERT, registered as version 1 with the `champion` alias. Model B, BERT, is version 2 with the `challenger` alias.

Model B's Macro-F1 difference (B minus A) was -0.0355, so it did not achieve the required +0.01 improvement. Its latency ratio was 1.305, within the 5.0 guardrail; however, both conditions are required, so Model A remains Champion. The full-precision values and run IDs are in `outputs/metrics_summary.json`.

## Limitations

- This project uses only 200 examples from the provided test split, so measured results can vary with sample choice and do not replace full test-set evaluation.
- The models are used as published; their label mapping and performance may be affected by upstream revisions.
- Inference latency is hardware-, software-, and batch-size-dependent. The timing excludes model loading and network downloads.
- Emotion labels are subjective and the model outputs should not be treated as clinical or high-stakes judgments.
- Registering both packaged checkpoints stores their model artifacts in the local MLflow artifact store, which requires additional disk space.

## MLflow and Submission Notes

The experiment is `hf-emotion-champion-challenger` at `http://127.0.0.1:5000`. The runs are `A_emotion_distilbert`, `B_emotion_bert`, and `compare_emotion`. Each candidate run logs parameters, metrics, tags, and a loadable PyFunc model; the comparison run logs the rule and winner.

The `screenshots/` directory contains capture instructions but no image files yet. Genuine captures of the running MLflow experiment, registry aliases, and successful prediction terminal still need to be added; do not substitute generated mockups. The current dev container could not launch Chromium because required system libraries are unavailable.

## AI Tools Declaration

AI assistance was used to help draft and review the project structure and code. The student should run the workflow, inspect its outputs, and be prepared to explain and verify each implementation choice.

## Viva Explanation

The workflow loads the official six-class test split and creates one reproducible stratified sample shared by both models. It checks each Hugging Face model's label names and maps predicted class IDs through that model's own configuration. Scikit-learn calculates classification metrics, while MLflow records each model, its parameters, metrics, tags, and packaged inference artifact. A third run applies the predeclared Macro-F1 and latency rule. Both models are registered, the winner and other candidate receive aliases, and the final script loads the winner through the Champion alias to classify eight new examples without directly invoking Hugging Face.