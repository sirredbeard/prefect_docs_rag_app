import git
import html2text
import markdown
import numpy as np
import onnxruntime as ort
import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from flask import Flask, jsonify, request, render_template_string
from multiprocessing import Process
from pydantic import BaseModel
from prefect import flow, task
from transformers import BertTokenizer

# FastAPI setup
app = FastAPI()

# Function to check and download the ONNX model if not present
def check_and_download_model(model_name, model_url, model_dir="onnx_model"):
    
    # Define the path to the model
    model_path = os.path.join(model_dir, model_name)

    # If the model doesn't exist, download it
    if not os.path.exists(model_path):

        # Create the model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download the model from the model URL
        response = requests.get(model_url)
        
        # Save the model
        with open(model_path, 'wb') as f:
            f.write(response.content)

    return model_path

# Task to clone or pull the Prefect repository
@task
def clone_or_pull_repo():

    # Define the Prefect repository URL and path to clone to
    repo_url = "https://github.com/PrefectHQ/prefect"
    repo_path = "./prefect_docs"

    # Set the path to the docs directory
    docs_path = os.path.join(repo_path, "docs")

    # Shallow clone the repository if it doesn't exist, otherwise pull latest
    if not os.path.exists(repo_path):
        repo = git.Repo.clone_from(repo_url, repo_path, depth=1)
    else:
        repo = git.Repo(repo_path)
        repo.remotes.origin.pull()

    return docs_path

# Task to convert and chunk the MDX files from the Prefect repository
@task
def chunk_mdx_files(docs_path):
    chunks = []

    # Iterate over the MDX files in the docs directory
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".mdx"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    
                    # Read the MDX file
                    data = f.read()
                    
                    # Convert MDX to HTML
                    html = markdown.markdown(data)
                    
                    # Extract plain text from HTML
                    plain_text = html2text.html2text(html)
                    
                    # Chunk the plain text
                    chunk_size = 100
                    lines = plain_text.split('\n')

                    # Split the text into chunks of 100 lines
                    for i in range(0, len(lines), chunk_size):
                        chunk = "\n".join(lines[i:i + chunk_size])
                        chunks.append(chunk)
    return chunks

# Task to create embeddings from the chunks of text
@task
def create_embeddings(chunks):

    # Define the ONNX model name and URL
    model_name = "bert-base-uncased.onnx"
    model_url = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.onnx"

    # Check and download the ONNX model if not present
    model_path = check_and_download_model(model_name, model_url)

    # Load the ONNX model and tokenizer

    # Use CUDA for embedding creation if available
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create embeddings for each chunk
    embeddings = []
    for chunk in chunks:

        # Tokenize the chunk
        inputs = tokenizer(chunk, return_tensors='np', padding=True, truncation=True)

        # Prepare inputs for the ONNX model
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        ort_inputs = {
            session.get_inputs()[0].name: input_ids,
            session.get_inputs()[1].name: attention_mask,
            session.get_inputs()[2].name: token_type_ids
        }

        # Get the token embeddings
        output = session.run(None, ort_inputs)[0]

        # Use mean pooling to get the sentence embedding
        mean_embedding = np.mean(output, axis=1) 

        # Store embedding and chunk
        embeddings.append((mean_embedding.flatten(), chunk)) 

    return embeddings

# Global variable to store embeddings
global_embeddings = []

# Flow to create embeddings from the Prefect documentation
@flow
def prefect_docs_embedding_flow():

    # Get the Prefect docs and set the path
    docs_path = clone_or_pull_repo()
    
    # Chunk the MDX files in the docs_path
    chunks = chunk_mdx_files(docs_path)
    
    # Store the embeddings in the global variable accessible to other tasks
    # TODO: Find a better way to share the embeddings between tasks
    global global_embeddings
    
    # Create embeddings from the chunks
    global_embeddings = create_embeddings(chunks)
    
    return global_embeddings

# FastAPI setup
class Query(BaseModel):
    text: str

# FastAPI endpoint to query the RAG model
@app.post("/query")
def query_rag(query: Query):
    
    # Access the global embeddings
    global global_embeddings

    # Set embeddings to the global embeddings variable
    embeddings = global_embeddings

    # Define the ONNX model name and URL
    model_name = "bert-base-uncased.onnx"
    model_url = "https://path-to-your-model/bert-base-uncased.onnx"

    # Set the path to the model
    model_path = check_and_download_model(model_name, model_url)

    # Load the ONNX model and tokenizer
    session = ort.InferenceSession(model_path)
    # TODO: Debug why CUDA is not working when serving the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the query
    inputs = tokenizer(query.text, return_tensors='np', padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # Prepare inputs for the ONNX model
    ort_inputs = {
        session.get_inputs()[0].name: input_ids,
        session.get_inputs()[1].name: attention_mask,
        session.get_inputs()[2].name: token_type_ids
    }

    # Get the token embeddings
    output = session.run(None, ort_inputs)[0]

    # Use mean pooling and flatten the embeddings
    query_embedding = np.mean(output, axis=1).flatten()

    # Simple retrieval mechanism (e.g., cosine similarity)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Find the best matching document
    best_match = None
    best_score = -1
    best_chunk = None

    # Iterate over the embeddings and find the best match
    for doc_embedding, chunk in embeddings:
        score = cosine_similarity(query_embedding, doc_embedding)
        if score > best_score:
            best_score = score
            best_match = doc_embedding
            best_chunk = chunk

    # Raise an exception if no relevant document is found
    if best_match is None:
        raise HTTPException(status_code=404, detail="No relevant document found.")

    return {"best_match": best_chunk, "score": float(best_score)}

# Task to run the FastAPI application
@task
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Flask app setup
flask_app = Flask(__name__)

# HTML template for the chat interface
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat with Prefect Docs Chat Bot</title>
    <style>
        body { font-family: Arial, sans-serif; }
        #chat-box { width: 100%; height: 300px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; }
        #user-input { width: 80%; padding: 10px; }
        #send-button { padding: 10px; }
    </style>
</head>
<body>
    <h1>Chat with Prefect Docs Chat Bot</h1>
    <div id="chat-box"></div>
    <input type="text" id="user-input" placeholder="Type your message here...">
    <button id="send-button">Send</button>

    <script>
        document.getElementById('send-button').addEventListener('click', function() {
            var userInput = document.getElementById('user-input').value;
            if (userInput) {
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: userInput })
                })
                .then(response => response.json())
                .then(data => {
                    var chatBox = document.getElementById('chat-box');
                    chatBox.innerHTML += '<p><strong>You:</strong> ' + userInput + '</p>';
                    chatBox.innerHTML += '<p><strong>RAG API:</strong> ' + data.best_match + '</p>';
                    document.getElementById('user-input').value = '';
                    chatBox.scrollTop = chatBox.scrollHeight;
                })
                .catch(error => console.error('Error:', error));
            }
        });
    </script>
</body>
</html>
"""

# Define Flask app routes
@flask_app.route('/')
def index():
    return render_template_string(html_template)

# Flask app route to chat with the RAG model
@flask_app.route('/chat', methods=['POST'])
def chat():
    # Get the user input
    user_input = request.json.get('text')
    
    # Return an error if no input text is provided
    if not user_input:
        return jsonify({"error": "No input text provided"}), 400

    # Query the RAG model
    response = requests.post("http://127.0.0.1:8000/query", json={"text": user_input})

    # Return an error if the RAG API fails
    if response.status_code != 200:
        return jsonify({"error": "Failed to get response from RAG API"}), 500

    # Return the response from the RAG API
    return jsonify(response.json())

# Task to run the Flask app
@task
def run_flask():
    flask_app.run(host="127.0.0.1", port=5000)

# Flow to serve the RAG application
@flow
def serve_rag_application():

    # Create embeddings from the Prefect documentation
    embeddings = prefect_docs_embedding_flow()
    
    # Run the FastAPI and Flask applications
    fastapi_process = Process(target=run_fastapi)
    flask_process = Process(target=run_flask)
    
    # Start the FastAPI and Flask processes
    fastapi_process.start()
    flask_process.start()
    
    # Join the FastAPI and Flask processes
    fastapi_process.join()
    flask_process.join()

# Run the flow
if __name__ == "__main__":
    serve_rag_application()