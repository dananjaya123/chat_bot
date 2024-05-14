from flask import Flask, render_template, request, jsonify
from train import train_model
from chatbot import predict_intent

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', title='Welcome to')


@app.route('/train')
def def_train_model():
    return train_model()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.json['message']
        predicted_intent = predict_intent(message)
        print(predicted_intent)
        return jsonify({'predicted_intent': predicted_intent})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run()
