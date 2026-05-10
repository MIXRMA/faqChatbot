from flask import Flask, request, jsonify, render_template
from src.chatbot import FaqChatbot

app = Flask(__name__)

bot = FaqChatbot(faqPath='data/faqs.json')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    Expects JSON body: { "message": "<user query>" }
    Returns JSON:      {
                         "answer":  "<response text>",
                         "score":   <float>,
                         "matched": "<faq question or empty string>",
                         "status":  "ok" | "low_confidence" | "fallback" | "empty"
                       }
    """
    data = request.get_json(force=True, silent=True)
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request. Send { "message": "..." }'}), 400

    userMessage = data['message'].strip()
    result = bot.respond(userMessage)
    return jsonify(result)


@app.route('/health')
def health():
    """Simple liveness check — useful when demoing."""
    return jsonify({'status': 'running', 'faq_count': len(bot.faqs)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)