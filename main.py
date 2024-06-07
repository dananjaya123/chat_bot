from flask import Flask, render_template, request, jsonify
from train import train_model
from chatbot import get_chatbot_response

app = Flask(__name__, static_folder='assets')


@app.route('/')
def index():
    return render_template('index.html', title='Welcome to')


@app.route('/login')
def login():
    return render_template('login.html', title='Login')


@app.route('/admin')
def admin():
    return render_template('adminPanel.html', title='Admin')


@app.route('/train', methods=['POST'])
def train():
    result = train_model()
    return jsonify({"status": result})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'message' not in data:
            raise ValueError("No 'message' field in request JSON")
        message = data['message']
        # print(f"Received message: {message}")
        predicted_intent = get_chatbot_response(message)
        return jsonify({'predicted_intent': predicted_intent})
    except Exception as e:
        # print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run()
