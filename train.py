#import the lib
import json
import pickle
import random
import string

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# tokenizer = Tokenizer()
lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')

def train_model():
    # inport the data set
    with open('content.json') as content:
        data1 = json.load(content)
        # print(data1)

    # genarate all data to list
    tags = []  # Initializes an empty list
    input_word = []  # Initializes an empty list
    response = {}  # Initializes an empty dictionary
    input_and_tag = []
    for intent in data1['intents']:
        response[intent['tag']] = intent['response']
        # print(response)

        # tokenize = splits up sentences into words
        for inputsval in intent['input']:
            # nltk library used to tokenize each input value into individual words.
            word_list = nltk.word_tokenize(inputsval)
            input_word.extend(word_list)

            input_and_tag.append((word_list, intent['tag']))
            # print("inputAndtag :",inputAndtag)

            if intent['tag'] not in tags:
                tags.append(intent['tag'])

    # Lowercase the word and remove punctuation(ex '',?$)
    input_word = [lemmatizer.lemmatize(word) for word in input_word if word not in string.punctuation]
    input_word = sorted(set(input_word))
    tags = sorted(set(tags))

    pickle.dump(input_word, open('words.pkl', 'wb'))
    pickle.dump(tags, open('tags.pkl', 'wb'))

    training = []
    output_empty = [0] * len(tags)

    for inputAndTags in input_and_tag:
        bag = []
        word_patterns = inputAndTags[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in input_word:
            bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[tags.index(inputAndTags[1])] = 1
            training.append([bag, output_row])
    max_length = max(len(x) for x, y in training)
    # print(max_length)

    # Pad sequences in training
    for i, (x, y) in enumerate(training):
        padding_length_x = max_length - len(x)
        padding_length_y = max_length - len(y)
        if padding_length_x > 0:
            x += [0] * padding_length_x
        if padding_length_y > 0:
            y += [0] * padding_length_y
        training[i] = (x, y)

    # Shuffle training data
    random.shuffle(training)

    # Convert training data to numpy array
    training = np.array(training)
    # Extract train_x and train_y
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential()  # sequential model
    # 128 = Neurons  input_shape dependant on  the size of the training data of train_x
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    # model.save('chatbot_model.model')

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    import matplotlib.pyplot as plt
    # pointing model accuracy
    plt.plot(hist.history['accuracy'], label='training accuracy')
    plt.plot(hist.history['loss'], label='training loss')
    plt.legend()

    model.save("chatbot_model.keras", hist)

    return "Model trained successfully"
