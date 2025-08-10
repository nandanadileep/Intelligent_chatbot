from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

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


state = create_state()
app = FastAPI(title="Chatbot (Intent + Memory + Generator)")


INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Chatbot</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }
      .container { max-width: 800px; margin: 0 auto; padding: 16px; }
      .chat { border: 1px solid #ddd; border-radius: 8px; padding: 12px; height: 60vh; overflow-y: auto; }
      .msg { margin: 8px 0; }
      .user { color: #111; }
      .assistant { color: #0a7; }
      .row { display: flex; gap: 8px; margin-top: 12px; }
      input[type=text] { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #bbb; }
      button { padding: 10px 14px; border: 0; border-radius: 6px; background: #0a7; color: #fff; cursor: pointer; }
      button:disabled { background: #999; cursor: not-allowed; }
      .meta { color: #666; font-size: 12px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h2>Chatbot</h2>
      <div id="chat" class="chat"></div>
      <div class="row">
        <input id="input" type="text" placeholder="Type your message and press Send" />
        <button id="send">Send</button>
      </div>
    </div>
    <script>
      const chat = document.getElementById('chat');
      const input = document.getElementById('input');
      const sendBtn = document.getElementById('send');

      function addMsg(role, content) {
        const div = document.createElement('div');
        div.className = 'msg ' + role;
        div.innerHTML = '<b>' + (role === 'user' ? 'You' : 'Assistant') + ':</b> ' + content;
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
      }

      sendBtn.onclick = async () => {
        const text = input.value.trim();
        if (!text) return;
        input.value = '';
        addMsg('user', text);
        sendBtn.disabled = true;
        try {
          const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
          });
          const data = await res.json();
          addMsg('assistant', data.reply + ' <span class="meta">(intent=' + data.intent + ', conf=' + data.confidence.toFixed(2) + ')</span>');
        } catch (e) {
          addMsg('assistant', 'Error: ' + e);
        } finally {
          sendBtn.disabled = false;
        }
      }

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendBtn.click();
      });
    </script>
  </body>
  </html>
"""


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return HTMLResponse(INDEX_HTML)


@app.post("/chat")
def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        return JSONResponse({"reply": "", "intent": "", "confidence": 0.0})

    state.ctx.add_message("user", msg)
    intent, confidence = state.ic.predict_intent(msg)
    used_intent = intent if confidence >= 0.3 else "general"
    retrieved = state.ctx.get_relevant_memory(msg, k=3)
    reply = state.rg.generate(user_message=msg, retrieved_context=retrieved, intent=used_intent)
    state.ctx.add_message("assistant", reply)
    return JSONResponse({"reply": reply, "intent": used_intent, "confidence": confidence})


@app.post("/reset")
def reset():
    state.ctx.reset()
    return JSONResponse({"ok": True})


