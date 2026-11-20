import json
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmotionClassifier(mlflow.pyfunc.PythonModel):
    """MLflow wrapper around a locally packaged Hugging Face classifier."""

    def load_context(self, context):
        model_dir = Path(context.artifacts["model_dir"])
        with (model_dir / "emotion_metadata.json").open(encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(value) for value in model_input]

        predictions = []
        batch_size = int(self.metadata["batch_size"])
        max_length = int(self.metadata["max_length"])
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)

            confidences, predicted_ids = probabilities.max(dim=-1)
            for class_id, confidence in zip(predicted_ids.tolist(), confidences.tolist()):
                predictions.append(
                    {
                        "predicted_emotion": self.metadata["id2label"][str(class_id)],
                        "confidence": float(confidence),
                    }
                )

        return pd.DataFrame(predictions)