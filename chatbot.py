import random
import json
import pickle
import numpy as np
import nltk
from keras.src.saving import load_model

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

nltk.download('punkt')


words_pkl = pickle.load(open('words.pkl', 'rb'))
tags_pkl = pickle.load(open('tags.pkl', 'rb'))
model = load_model('chatbot_model.keras')
# ============================
# inport the data set
with open('content.json') as content:
    intents = json.load(content)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_pkl)
    for w in sentence_words:
        for i, word in enumerate(words_pkl):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': tags_pkl[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['response'])
            break
    return result
# ==============================

# Function to predict intent
def predict_intent(message):
    user_input = message
    intents_list = predict_class(user_input)
    result = get_response(intents_list)
    return result