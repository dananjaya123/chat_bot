import json
import pickle
import random

import nltk
import numpy as np
from keras.src.saving import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# 'punkt' tokenizer from NLTK, which is used for splitting a text into a list of sentences.
nltk.download('punkt')

# Load Pre-trained Data and Model
words_pkl = pickle.load(open('words.pkl', 'rb'))
tags_pkl = pickle.load(open('tags.pkl', 'rb'))
model = load_model('chatbot_model.keras')

# inport the json data
with open('content.json') as content:
    intents = json.load(content)

def tokenize_and_lemmatize(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    # print("sentence_words:", sentence_words)
    return sentence_words


def bag_of_words(sentence):
    sentence_words = tokenize_and_lemmatize(sentence)
    # print("word_pkl length ", len(words_pkl))
    # words_plk file data leng size 0 arry create [0,0,..]
    bag = [0] * len(words_pkl)
    for wd in sentence_words:
        # print("words_pkl", words_pkl)
        # enumerate usig index the words_pkl data (eg:- 1 hi , 2 helo ..)
        for i, word in enumerate(words_pkl):
            if word == wd:
                # correctly place 1 in the bag list at the corresponding position
                bag[i] = 1
    return np.array(bag)


def classify_intent(sentence):
    bow = bag_of_words(sentence)
    #  (bag of words) array is converted into a NumPy array and passed to the model.predict function
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    # results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results = []
    # enumarate the model result (eg:[[1, 0.2848842], [2, 0.7150764]])
    for i, r in enumerate(res):
        if r > ERROR_THRESHOLD:
            results.append([i, r])

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': tags_pkl[r[0]], 'probability': str(r[1])})
    return return_list


def generate_response(intents_list):
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['response'])
            break
    return result

def get_chatbot_response(message):
    user_input = message
    intents_list = classify_intent(user_input)
    result = generate_response(intents_list)
    return result
