# prefect_docs_rag_app

Uses the [Prefect](https://github.com/PrefectHQ/prefect) workflow orchestration framework to generate and deploy a *very rudimentary* chatbot trained with RAG on the Prefect documentation. This a project to experiment with Prefect.

* Based on the [Google BERT model](https://huggingface.co/google-bert/bert-base-uncased), which the script will download
* Leverages the [ONNX Runtime](https://onnx.ai/) for platform/architecture portability
* Clones or pulls the Prefect repository then extracts and converts .mdx files to create embeddings, poorly
* Uses FastAPI to serve an API to the model
* Uses Flask to create a simple chatbot interface

## Installation

```
git clone https://www.github.com/sirredbeard/prefect_docs_rag_app
cd prefect_docs_rag_app
pip install -r requirements.txt
python3 main.py
```

## Use

Browse to [127.0.0.1:5000/](http://127.0.0.1:5000/) after deploying.