from flask import Flask, request, render_template, jsonify
from model import predict_emotion_intensity
from llama_cpp import Llama

app = Flask(__name__)

llm = Llama(model_path="llama-2-7b-chat.Q4_K_M.gguf", n_ctx=512)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    user_intensity = predict_emotion_intensity(user_message)
    bot_response = generate_response(user_message, user_intensity)
    bot_intensity = predict_emotion_intensity(bot_response)

    # 4. Return everything to the frontend
    return jsonify({
        'response': bot_response,
        'user_intensity': user_intensity,
        'bot_intensity': bot_intensity
    })

def generate_response(user_message, intensity):
    prompt = f"""### System:
    You are a supportive and empathetic chatbot. The user writes a message, and you respond in a way that matches the emotional intensity.

    ### Context:
    User's emotional intensity: {intensity:.2f}

    ### User:
    {user_message}

    ### Assistant:
    """
    output = llm(prompt, max_tokens=150, temperature=0.7, stop=["###"])
    return output['choices'][0]['text'].strip()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
