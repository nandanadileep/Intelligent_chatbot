from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ResponseGenerator:
    """
    Simple FLAN-T5 based responder that conditions on retrieved memory and intent.
    """

    def __init__(self, model_name: str = "google/flan-t5-large") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _format_context(self, retrieved_context: List[dict]) -> str:
        lines = []
        for item in retrieved_context:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def generate(self, user_message: str, retrieved_context: List[dict], intent: str) -> str:
        context_str = self._format_context(retrieved_context)
        few_shots = (
            "Examples:\n"
            "User: which continent is India in?\n"
            "Assistant: Asia.\n\n"
            "User: what clothes should I wear in summer?\n"
            "Assistant: Choose lightweight, breathable fabrics like cotton or linen; light colors such as white or beige; short sleeves or sleeveless tops; and open shoes or sandals.\n\n"
            "User: what's the best burger in America?\n"
            "Assistant: I don’t have definitive rankings. Popular options include regional specialties; if you share your city, I can suggest nearby places.\n\n"
        )
        prompt = (
            "You are a concise, helpful assistant. Use the conversation context and the intent label. "
            "If the intent is 'general', just answer the user's question directly. If you are unsure, say so briefly.\n\n"
            f"Intent: {intent}\n"
            f"Context (recent messages):\n{context_str}\n\n"
            f"{few_shots}"
            f"User: {user_message}\n"
            "Assistant:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
            )
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return response.strip()


__all__ = ["ResponseGenerator"]


