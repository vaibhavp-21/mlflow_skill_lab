import argparse
import json
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from emotion_model import EmotionClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_A_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
MODEL_B_ID = "bhadresh-savani/bert-base-uncased-emotion"
DATASET_NAME = "dair-ai/emotion"
EXPERIMENT_NAME = "hf-emotion-champion-challenger"
EVAL_SAMPLES = 200
RANDOM_SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
MIN_F1_IMPROVEMENT = 0.01
LATENCY_GUARDRAIL = 5.0


def normalize_label(label):
    return str(label).strip().lower()


def model_label_mapping(config, dataset_labels, model_id):
    """Validate the model's label names and preserve its own ID mapping."""
    raw_mapping = getattr(config, "id2label", None) or {}
    mapping = {str(int(class_id)): normalize_label(label) for class_id, label in raw_mapping.items()}
    expected_labels = {normalize_label(label) for label in dataset_labels}
    if len(mapping) != len(dataset_labels) or set(mapping.values()) != expected_labels:
        raise ValueError(
            f"{model_id} labels {mapping} do not match dataset labels {sorted(expected_labels)}"
        )
    return mapping


def choose_evaluation_rows(dataset, label_names):
    """Select a reproducible, stratified sample and retain source row indices."""
    if len(dataset) < EVAL_SAMPLES:
        raise ValueError(f"Test split has {len(dataset)} rows; need {EVAL_SAMPLES}.")
    labels = np.asarray(dataset["label"])
    selected, _ = train_test_split(
        np.arange(len(dataset)),
        train_size=EVAL_SAMPLES,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    selected = np.sort(selected)
    sampled = dataset.select(selected.tolist())
    frame = pd.DataFrame(
        {
            "source_index": selected,
            "text": sampled["text"],
            "label_id": sampled["label"],
            "label": [label_names[int(label_id)] for label_id in sampled["label"]],
        }
    )
    return frame


def predict_model(model, tokenizer, id_to_label, texts, device):
    """Run batched inference and return name-aligned labels and confidences."""
    predicted_labels = []
    confidences = []
    start_time = time.perf_counter()
    model.eval()
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
        batch_confidence, batch_ids = probabilities.max(dim=-1)
        predicted_labels.extend(id_to_label[str(int(class_id))] for class_id in batch_ids.tolist())
        confidences.extend(float(value) for value in batch_confidence.tolist())
    elapsed_seconds = time.perf_counter() - start_time
    return predicted_labels, confidences, elapsed_seconds


def metric_values(expected, predicted, total_seconds):
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_precision": float(precision_score(expected, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(expected, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(expected, predicted, average="weighted", zero_division=0)),
        "average_latency_ms": float(total_seconds * 1000 / len(expected)),
        "total_inference_time_s": float(total_seconds),
    }


def save_confusion_matrix(expected, predicted, label_names, title, output_path):
    matrix = confusion_matrix(expected, predicted, labels=label_names)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=[name.title() for name in label_names],
        yticklabels=[name.title() for name in label_names],
        xlabel="Predicted emotion",
        ylabel="True emotion",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Examples")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_comparison_plot(results):
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    display_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "Weighted-F1"]
    positions = np.arange(len(metric_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - width / 2, [results["Model A"][key] for key in metric_names], width, label="Model A: DistilBERT")
    axis.bar(positions + width / 2, [results["Model B"][key] for key in metric_names], width, label="Model B: BERT")
    axis.set_xticks(positions, display_names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Emotion classifier evaluation metrics")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)


def save_disagreements(frame, predictions_a, predictions_b):
    disagreements = []
    for index, (prediction_a, prediction_b) in enumerate(zip(predictions_a, predictions_b)):
        if prediction_a == prediction_b:
            continue
        row = frame.iloc[index]
        disagreements.append(
            {
                "text": row["text"],
                "reference_emotion": row["label"],
                "distilbert_prediction": prediction_a,
                "bert_prediction": prediction_b,
                "explanation": (
                    f"The models assigned different labels ({prediction_a} vs {prediction_b}); "
                    f"the dataset reference label is {row['label']}."
                ),
            }
        )
        if len(disagreements) == 5:
            break
    if len(disagreements) < 5:
        raise RuntimeError(
            f"Only {len(disagreements)} disagreement examples were observed; five cannot be reported truthfully."
        )
    pd.DataFrame(disagreements).to_csv(OUTPUT_DIR / "disagreements.csv", index=False)


def log_candidate_run(run_name, model_id, role, model, tokenizer, id_to_label, frame, metrics, predictions):
    device = next(model.parameters()).device
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "hf_model_id": model_id,
                "task": "six-class emotion classification",
                "model_role": role,
                "dataset_name": DATASET_NAME,
                "eval_samples": EVAL_SAMPLES,
                "random_seed": RANDOM_SEED,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "device": str(device),
            }
        )
        mlflow.set_tags(
            {
                "student_name": "Vaibhav Patil",
                "student_usn": "1MS25CI134",
                "dataset_name": DATASET_NAME,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "mlflow_version": mlflow.__version__,
                "hf_model_id": model_id,
            }
        )
        mlflow.log_metrics(metrics)
        predictions_path = OUTPUT_DIR / ("predictions_distilbert.csv" if role == "Model A baseline" else "predictions_bert.csv")
        prediction_frame = pd.DataFrame(
            {
                "source_index": frame["source_index"],
                "text": frame["text"],
                "reference_emotion": frame["label"],
                "predicted_emotion": predictions[0],
                "confidence": predictions[1],
            }
        )
        prediction_frame.to_csv(predictions_path, index=False)
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
        confusion_path = OUTPUT_DIR / ("confusion_distilbert.png" if role == "Model A baseline" else "confusion_bert.png")
        save_confusion_matrix(frame["label"].tolist(), predictions[0], list(id_to_label.values()), run_name, confusion_path)
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")

        with tempfile.TemporaryDirectory(prefix="emotion_model_") as temporary_dir:
            model.save_pretrained(temporary_dir)
            tokenizer.save_pretrained(temporary_dir)
            model_size_mb = sum(
                path.stat().st_size
                for path in Path(temporary_dir).rglob("*")
                if path.is_file()
            ) / (1024 * 1024)
            mlflow.log_param("packaged_model_size_mb", round(model_size_mb, 3))
            with (Path(temporary_dir) / "emotion_metadata.json").open("w", encoding="utf-8") as metadata_file:
                json.dump(
                    {
                        "hf_model_id": model_id,
                        "id2label": id_to_label,
                        "max_length": MAX_LENGTH,
                        "batch_size": BATCH_SIZE,
                    },
                    metadata_file,
                    indent=2,
                )
            signature = ModelSignature(
                inputs=Schema([ColSpec("string", "text")]),
                outputs=Schema([ColSpec("string", "predicted_emotion"), ColSpec("double", "confidence")]),
            )
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=EmotionClassifier(),
                artifacts={"model_dir": temporary_dir},
                code_paths=[str(PROJECT_DIR / "emotion_model.py")],
                signature=signature,
                pip_requirements=[
                    "mlflow==2.22.0",
                    "pandas==2.2.3",
                    "SQLAlchemy==2.0.40",
                    "--extra-index-url https://download.pytorch.org/whl/cpu",
                    "torch==2.6.0+cpu",
                    "transformers==4.51.3",
                ],
            )
        return run.info.run_id, model_size_mb


def main():
    parser = argparse.ArgumentParser(description="Evaluate two Hugging Face emotion classifiers with MLflow.")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset = load_dataset(DATASET_NAME, "split", split="test")
    dataset_labels = dataset.features["label"].names
    print(f"Dataset test split: {len(dataset)} rows; labels: {dataset_labels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = [
        ("A_emotion_distilbert", "Model A baseline", MODEL_A_ID),
        ("B_emotion_bert", "Model B challenger", MODEL_B_ID),
    ]
    loaded_models = []
    for run_name, role, model_id in model_specs:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        id_to_label = model_label_mapping(model.config, dataset_labels, model_id)
        model.to(device)
        model.eval()
        print(f"Verified {model_id}: {id_to_label}")
        loaded_models.append((run_name, role, model_id, model, tokenizer, id_to_label))

    frame = choose_evaluation_rows(dataset, dataset_labels)
    frame.to_csv(DATA_DIR / "eval_inputs.csv", index=False)
    print(f"Saved {len(frame)} shared stratified rows to data/eval_inputs.csv")
    expected = [normalize_label(label) for label in frame["label"].tolist()]
    texts = frame["text"].tolist()
    results = {}
    run_ids = {}
    predictions_by_model = {}

    for run_name, role, model_id, model, tokenizer, id_to_label in loaded_models:
        mapped = {class_id: normalize_label(label) for class_id, label in id_to_label.items()}
        predictions, confidences, total_seconds = predict_model(model, tokenizer, mapped, texts, device)
        metrics = metric_values(expected, predictions, total_seconds)
        model_key = "Model A" if role == "Model A baseline" else "Model B"
        results[model_key] = metrics
        predictions_by_model[model_key] = (predictions, confidences)
        run_ids[model_key], model_size_mb = log_candidate_run(
            run_name,
            model_id,
            role,
            model,
            tokenizer,
            mapped,
            frame,
            metrics,
            (predictions, confidences),
        )
        results[model_key]["model_size_mb"] = float(model_size_mb)
        print(f"{run_name}: {json.dumps(metrics, sort_keys=True)}")

    macro_f1_difference = results["Model B"]["macro_f1"] - results["Model A"]["macro_f1"]
    latency_ratio = results["Model B"]["average_latency_ms"] / max(results["Model A"]["average_latency_ms"], 1e-12)
    winner = "Model B" if macro_f1_difference >= MIN_F1_IMPROVEMENT and latency_ratio <= LATENCY_GUARDRAIL else "Model A"
    decision_explanation = (
        f"{winner} is Champion: Model B Macro-F1 improvement is {macro_f1_difference:.4f} "
        f"(required >= {MIN_F1_IMPROVEMENT:.2f}); latency ratio B/A is {latency_ratio:.3f} "
        f"(guardrail <= {LATENCY_GUARDRAIL:.1f})."
    )

    save_comparison_plot(results)
    save_disagreements(frame, predictions_by_model["Model A"][0], predictions_by_model["Model B"][0])
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": args.tracking_uri,
        "dataset_name": DATASET_NAME,
        "eval_samples": EVAL_SAMPLES,
        "random_seed": RANDOM_SEED,
        "label_names": [normalize_label(label) for label in dataset_labels],
        "models": {
            "Model A": {
                "name": "A_emotion_distilbert",
                "hf_model_id": MODEL_A_ID,
                "run_id": run_ids["Model A"],
                "model_uri": f"runs:/{run_ids['Model A']}/model",
                "metrics": results["Model A"],
            },
            "Model B": {
                "name": "B_emotion_bert",
                "hf_model_id": MODEL_B_ID,
                "run_id": run_ids["Model B"],
                "model_uri": f"runs:/{run_ids['Model B']}/model",
                "metrics": results["Model B"],
            },
        },
        "decision_rule": {
            "primary_metric": "macro_f1",
            "minimum_improvement_margin": MIN_F1_IMPROVEMENT,
            "latency_guardrail_ratio": LATENCY_GUARDRAIL,
            "latency_ratio_definition": "Model B average latency / Model A average latency",
        },
        "comparison": {
            "macro_f1_difference_b_minus_a": macro_f1_difference,
            "latency_ratio_b_over_a": latency_ratio,
            "winner": winner,
            "decision_explanation": decision_explanation,
        },
    }

    with mlflow.start_run(run_name="compare_emotion") as comparison_run:
        mlflow.log_params(
            {
                "model_a_name": "A_emotion_distilbert",
                "model_b_name": "B_emotion_bert",
                "primary_metric": "Macro-F1",
                "minimum_improvement_margin": MIN_F1_IMPROVEMENT,
                "latency_guardrail_ratio": LATENCY_GUARDRAIL,
                "macro_f1_difference_b_minus_a": macro_f1_difference,
                "latency_ratio_b_over_a": latency_ratio,
                "final_winner": winner,
                "decision_explanation": decision_explanation,
            }
        )
        mlflow.log_metrics(
            {
                "macro_f1_difference_b_minus_a": macro_f1_difference,
                "latency_ratio_b_over_a": latency_ratio,
                "model_a_macro_f1": results["Model A"]["macro_f1"],
                "model_b_macro_f1": results["Model B"]["macro_f1"],
            }
        )
        mlflow.set_tags(
            {
                "student_name": "Vaibhav Patil",
                "student_usn": "1MS25CI134",
                "final_winner": winner,
                "decision_explanation": decision_explanation,
            }
        )
        for artifact in (
            OUTPUT_DIR / "metrics_summary.json",
            OUTPUT_DIR / "model_comparison.png",
            OUTPUT_DIR / "disagreements.csv",
        ):
            if artifact.name == "metrics_summary.json":
                artifact.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact), artifact_path="comparison")
        summary["comparison"]["run_id"] = comparison_run.info.run_id

    (OUTPUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, filename in (("Model A", "confusion_distilbert.png"), ("Model B", "confusion_bert.png")):
        print(f"{key} classification report:\n{classification_report(expected, predictions_by_model[key][0], labels=summary['label_names'], zero_division=0)}")
    print(decision_explanation)
    print(f"Metrics and run references saved to {OUTPUT_DIR / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()