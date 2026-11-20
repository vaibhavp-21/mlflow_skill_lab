import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = PROJECT_DIR / "outputs" / "metrics_summary.json"
REGISTERED_MODEL_NAME = "emotion-classifier"


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("Run evaluate.py successfully before registering models.")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(summary["tracking_uri"])
    client = MlflowClient()

    registered_versions = {}
    for model_key in ("Model A", "Model B"):
        model_info = summary["models"][model_key]
        version = mlflow.register_model(
            model_uri=model_info["model_uri"],
            name=REGISTERED_MODEL_NAME,
            await_registration_for=300,
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version.version,
            "hf_model_id",
            model_info["hf_model_id"],
        )
        registered_versions[model_key] = str(version.version)
        print(f"Registered {model_info['name']} as {REGISTERED_MODEL_NAME} version {version.version}")

    winner_key = summary["comparison"]["winner"]
    other_key = "Model B" if winner_key == "Model A" else "Model A"
    alias_versions = {"champion": registered_versions[winner_key], "challenger": registered_versions[other_key]}
    for alias, version_number in alias_versions.items():
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version_number)
        print(f"Alias '{alias}' -> {REGISTERED_MODEL_NAME} version {version_number}")

    summary["registry"] = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "versions": registered_versions,
        "aliases": alias_versions,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()