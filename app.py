from __future__ import annotations

import os
from typing import List

from context_manager import ContextManager
from intent_classifier import IntentClassifier
from response_generator import ResponseGenerator


def main() -> None:
    print("Booting chatbot... (first run may download models)")

    ctx = ContextManager(max_tokens=256)
    ic = IntentClassifier(
        model_path=os.getenv("INTENT_MODEL_PATH"),
        intent_model_id=os.getenv("INTENT_MODEL_ID"),
    )
    rg = ResponseGenerator()

    print("Chat ready. Type 'exit' to quit.")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Assistant: Bye!")
            break

        # Add user message to memory
        ctx.add_message("user", user)

        # Intent classification
        intent, confidence = ic.predict_intent(user)

        # Retrieve relevant memory snippets
        retrieved = ctx.get_relevant_memory(user, k=3)

        # Generate response
        reply = rg.generate(user_message=user, retrieved_context=retrieved, intent=intent)

        # Add assistant reply to memory and print
        ctx.add_message("assistant", reply)
        print(f"Assistant: {reply} \n(intent={intent}, conf={confidence:.2f})")


if __name__ == "__main__":
    main()


