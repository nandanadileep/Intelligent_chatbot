from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class IntentClassifier:
    """
    Lightweight wrapper around a BERT intent classifier.

    - Uses the CLINC150 (clinc_oos, small) label space by default to get label names
    - Priority for loading model:
        1) HF hub model id via intent_model_id or INTENT_MODEL_ID
        2) Local checkpoint dir via model_path or INTENT_MODEL_PATH
        3) Fallback: base model name with randomly initialized classification head
    """

    def __init__(
        self,
        model_path: str | None = None,
        intent_model_id: str | None = None,
        base_model_name: str = "bert-base-uncased",
        dataset_name: str = "clinc_oos",
        dataset_config: str = "small",
        device: str | None = None,
    ) -> None:
        # Dataset label feature as a fallback for label names
        self.dataset = load_dataset(dataset_name, dataset_config)
        self.label_feature = self.dataset["train"].features["intent"]
        self.num_labels = self.label_feature.num_classes

        # Resolve inputs and env vars
        resolved_model_path = model_path or os.getenv("INTENT_MODEL_PATH")
        resolved_model_id = intent_model_id or os.getenv("INTENT_MODEL_ID")

        # Load model/tokenizer with priority: model_id > local_path > base
        if resolved_model_id:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                resolved_model_id
            )
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
        elif resolved_model_path and os.path.isdir(resolved_model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                resolved_model_path
            )
            # Prefer tokenizer from local checkpoint when available; fallback to base
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=self.num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict_intent(self, text: str) -> Tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(probs).item())
            confidence = float(probs[pred_id].item())
        # Prefer model's embedded id2label if present; else fall back to dataset mapping
        id2label: Dict[int, str] | None = None
        try:
            # config.id2label can have str keys; normalize
            raw = getattr(self.model.config, "id2label", None)
            if isinstance(raw, dict) and raw:
                id2label = {int(k): v for k, v in raw.items()}
        except Exception:
            id2label = None

        def is_default_hf_labels(mapping: Dict[int, str] | None) -> bool:
            if not mapping:
                return True
            values = list(mapping.values())
            return all(isinstance(v, str) and v.startswith("LABEL_") for v in values)

        if id2label and (not is_default_hf_labels(id2label)) and pred_id in id2label:
            label = id2label[pred_id]
        else:
            label = self.label_feature.int2str(pred_id)
        return label, confidence

    def label_names(self) -> List[str]:
        return list(self.label_feature.names)


__all__ = ["IntentClassifier"]


