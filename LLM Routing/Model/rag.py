from transformers import BertTokenizer, BertModel
import faiss
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def retrieve_documents(query, top_k=3):
    query_vector = encode_text(query)
    distances, indices = index.search(query_vector, top_k)
    return [documents[i] for i in indices[0]]

documents = ["doc1 text", "doc2 text", ...]
doc_vectors = np.array([encode_text(doc) for doc in documents])

index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(doc_vectors)

# def deployment():
#     from flask import Flask, request, jsonify

#     app = Flask(__name__)

#     @app.route('/answer', methods=['POST'])

#     def answer():
#         query = request.json['query']
#         retrieved_docs = retrieve_documents(query)
#         return jsonify({'retrieved_docs': retrieved_docs})

#     app.run(host='0.0.0.0', port=5000)

