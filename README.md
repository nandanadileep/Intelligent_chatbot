## Intelligent Chatbot

This repository contains an end-to-end intent-aware chatbot with short- and long-term memory and a sequence-to-sequence response generator. It is built from three components: intent classification, context memory, and response generation. A command-line interface and a minimal FastAPI web UI are included.

### Features
- Intent classification using BERT with CLINC label space. Supports either a local fine-tuned checkpoint or a remote Hugging Face model id.
- Context management with short-term token window and long-term retrieval via SentenceTransformer embeddings and NumPy cosine similarity.
- Response generation using FLAN-T5, conditioned on predicted intent and retrieved context, with a few-shot prompt to stabilize answers.
- CLI runner and FastAPI-based local web UI.

### Repository Structure
- `intent_classifier.py`: Loads tokenizer and sequence classification model. Supports `INTENT_MODEL_PATH` (local) and `INTENT_MODEL_ID` (Hugging Face hub id). Falls back to base BERT if none provided.
- `context_manager.py`: Keeps recent messages within a token budget and stores long-term memory using MiniLM embeddings and cosine similarity.
- `response_generator.py`: FLAN-T5 based generator with a concise few-shot prompt.
- `app.py`: CLI orchestrator for local chat.
- `server.py`: FastAPI app serving a minimal HTML page and `/chat` endpoint.
- `ui.py`: Optional Gradio interface (may require resolving local NumPy/Matplotlib compatibility on some macOS setups).
- Notebooks: `Intent_Classification.ipynb`, `context-management-after-bert.ipynb`, `response-generator.ipynb`.

### Setup
Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you hit macOS user site-packages conflicts, prefer running with:
```bash
PYTHONNOUSERSITE=1 python -I -c "import sys; print('venv only')"
```

### Running the CLI
```bash
source .venv/bin/activate
python app.py
```

### Running the FastAPI UI
```bash
source .venv/bin/activate
PYTHONNOUSERSITE=1 uvicorn server:app --host 127.0.0.1 --port 7862
```
Then open `http://127.0.0.1:7862`.

### Using a Fine-Tuned Intent Model
- Local checkpoint:
```bash
export INTENT_MODEL_PATH="/absolute/path/to/intent_ckpt"
```
- Remote model (Hugging Face id):
```bash
export INTENT_MODEL_ID="org/model-id"
```

### Fine-Tuning Intent (Notebook)
In `Intent_Classification.ipynb`, after training completes, add:
```python
trainer.save_model("intent_ckpt")
tokenizer.save_pretrained("intent_ckpt")
```
Then set `INTENT_MODEL_PATH` as above.

### Notes
- The default intent classifier without fine-tuning has low confidence. For good routing, provide a fine-tuned intent model.
- The generator prompt is few-shot and generic; tailor examples for your domain for better quality.
- Long-term retrieval uses cosine similarity over MiniLM embeddings. For large memories, consider a vector database.

### Notebooks Overview
| Notebook | Description |
| --- | --- |
| `Intent_Classification.ipynb` | Trains a BERT-based intent classifier on CLINC small. |
| `context-management-after-bert.ipynb` | Short- and long-term memory with embeddings and optional summarization. |
| `response-generator.ipynb` | Prompts FLAN-T5 to answer using intent and retrieved context. |

### License
MIT
