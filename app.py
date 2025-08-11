from flask import Flask, request, jsonify
from chatbot import search_similar, generate_answer

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    contexts = search_similar(user_query)
    answer = generate_answer(contexts, user_query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(port=5000)
