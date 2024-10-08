# prefect_docs_rag_app

Uses the [Prefect](https://github.com/PrefectHQ/prefect) workflow orchestration framework to generate and deploy a *very rudimentary* chatbot trained with RAG on the Prefect documentation. This a project to experiment with Prefect.

* Based on the [Google BERT model](https://huggingface.co/google-bert/bert-base-uncased), which the script will download
* Leverages the [ONNX Runtime](https://onnx.ai/) for platform/architecture portability
* Clones or pulls the Prefect repository then extracts and converts .mdx files to create embeddings, *poorly*
* Uses FastAPI to serve an API to the model
* Uses Flask to create a simple chatbot interface

## Screenshots

![image](https://github.com/user-attachments/assets/72b2cd29-40b5-4b8a-aa6f-0c744dde49a3)

![image](https://github.com/user-attachments/assets/8f1a5a68-8d11-45d7-b36a-a8a2783f5e8d)


## Installation

```
git clone https://www.github.com/sirredbeard/prefect_docs_rag_app
cd prefect_docs_rag_app
pip install -r requirements.txt
python3 main.py
```

## Use

Browse to [127.0.0.1:5000](http://127.0.0.1:5000/) after deploying.

## TODO / Known Issues

- Improve handling of the global embeddings variable, currently hacky
- The CUDA provider works fine with the ONNX runtime for creating embeddings but not for inferencing, it reports as unavailable and then reverts to CPU, has the runtime not freed up the CUDA provider?
- Instead of using `process` from `multiprocessing` use Prefect-native [task runners](https://docs.prefect.io/3.0/develop/task-runners) to serve the API and web frontend simulteanously
- Switch to a better model than BERT, preference would be another SLM, like Phi-3, but because ONNX has built-in tokenizer support for BERT it was easier to start with BERT, did some testing with Phi-3 but without CUDA on inferencing it was very slow to respond, so reverted to BERT to complete the initial goals
- Improve conversion of `.mdx` files to plain text, currently using `html2txt`, tried `BeautifulSoup` but wasn't much better, still a fair bit of HTML and CSS gets through to responses *or* implement markdown/html rendering in the responses to parse HTML/CSS/.mdx
- Tune the embeddings logic to provide better, more accurate responses, currently using simple cosine similarity, sometimes answers are completely random or non-responsive
