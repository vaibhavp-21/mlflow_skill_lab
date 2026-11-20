import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "data" / "new_data.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "new_data_predictions.csv"
MODEL_NAME = "emotion-classifier"
MODEL_ALIAS_URI = f"models:/{MODEL_NAME}@champion"


def main():
    summary_path = PROJECT_DIR / "outputs" / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run evaluate.py and register_champion.py before prediction.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(summary["tracking_uri"])
    client = MlflowClient()
    champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model_run = client.get_run(champion_version.run_id)
    hf_model_id = model_run.data.params["hf_model_id"]

    new_data = pd.read_csv(INPUT_PATH)
    evaluation_data = pd.read_csv(PROJECT_DIR / "data" / "eval_inputs.csv")
    evaluation_texts = set(evaluation_data["text"].astype(str))
    overlap = evaluation_texts.intersection(new_data["text"].astype(str))
    if overlap:
        raise ValueError(f"New prediction inputs overlap evaluation examples: {sorted(overlap)}")

    champion = mlflow.pyfunc.load_model(MODEL_ALIAS_URI)
    prediction = champion.predict(new_data[["text"]])
    result = new_data.copy()
    result["predicted_emotion"] = prediction["predicted_emotion"].astype(str)
    result["confidence"] = prediction["confidence"].astype(float)
    result["model_version"] = str(champion_version.version)
    result["hf_model_id"] = hf_model_id

    print(f"Loaded {MODEL_ALIAS_URI}, version {champion_version.version} ({hf_model_id})")
    for row in result.itertuples(index=False):
        print(f"Text: {row.text}\nEmotion: {row.predicted_emotion} | Confidence: {row.confidence:.4f}\n")
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()