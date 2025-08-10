from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import gradio as gr

from context_manager import ContextManager
from intent_classifier import IntentClassifier
from response_generator import ResponseGenerator


@dataclass
class ChatbotState:
    ctx: ContextManager
    ic: IntentClassifier
    rg: ResponseGenerator


def create_state() -> ChatbotState:
    ctx = ContextManager(max_tokens=256)
    ic = IntentClassifier(
        model_path=os.getenv("INTENT_MODEL_PATH"),
        intent_model_id=os.getenv("INTENT_MODEL_ID"),
    )
    rg = ResponseGenerator()
    return ChatbotState(ctx=ctx, ic=ic, rg=rg)


def respond(user_message: str, history: List[Tuple[str, str]], state: ChatbotState):
    if not user_message:
        return "", history, state

    # Update memory with user message
    state.ctx.add_message("user", user_message)

    # Predict intent
    intent, confidence = state.ic.predict_intent(user_message)

    # Retrieve relevant memory
    retrieved = state.ctx.get_relevant_memory(user_message, k=3)

    # Generate reply
    reply = state.rg.generate(user_message=user_message, retrieved_context=retrieved, intent=intent)

    # Update memory with assistant reply
    state.ctx.add_message("assistant", reply)

    # Append to UI history (show intent lightly)
    annotated_reply = f"{reply}\n(intent={intent}, conf={confidence:.2f})"
    history = history + [(user_message, annotated_reply)]

    # Clear textbox (return "") and update components
    return "", history, state


with gr.Blocks(title="Chatbot (Intent + Memory + Generator)") as demo:
    gr.Markdown("""
    **End-to-end Chatbot**
    - Intent: BERT classifier on CLINC labels
    - Memory: Short-term window + FAISS long-term retrieval
    - Generator: FLAN-T5 conditioned on intent and retrieved memory
    """)

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your message and press Enter", scale=1)
    state = gr.State(create_state())
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])


if __name__ == "__main__":
    demo.launch()


