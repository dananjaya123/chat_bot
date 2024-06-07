# import the lib
import json  # load the dataset from a JSON file.
import pickle  # For saving and loading preprocessed data
import random  # For shuffling the training data
import string  # For handling punctuation

import matplotlib.pyplot as plt  # For plotting training metrics
import nltk  # Natural Language Toolkit for NLP tasks
import numpy as np  # For numerical operations and data manipulation
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# tensorflow.keras: For building and training the neural network model.

nltk.download('punkt')
nltk.download('wordnet')
# Downloads the necessary NLTK data for tokenization and lemmatization

lemmatizer = WordNetLemmatizer()


# Initializes the WordNet lemmatizer for converting words to their base form

def train_model():
    # load the json data set
    with open('content.json') as content:
        data1 = json.load(content)
        # print(data1)

    # generate all data to list
    tags = []  # Initializes an empty list
    input_word = []  # Initializes an empty list
    response = {}  # Initializes an empty dictionary
    input_and_tag = []
    for intent in data1['intents']:
        response[intent['tag']] = intent['response']
        # print(response)

        # tokenize = splits up sentences into words
        for input_value in intent['input']:
            # nltk library used to tokenize each input value into individual words.
            word_list = nltk.word_tokenize(input_value)
            # combining all the word
            input_word.extend(word_list)

            input_and_tag.append((word_list, intent['tag']))
            # print("tags :",input_and_tag)

            if intent['tag'] not in tags:
                tags.append(intent['tag'])

    lemmatized_words = []
    # Iterate over each word in the input_word list
    for word in input_word:
        # Check if the word is not a punctuation mark(ex '',?$)
        if word not in string.punctuation:
            # nltk library WordNetLemmatizer  tool used in NLP to reduce words to their base or root form.
            lemmatized_words.append(lemmatizer.lemmatize(word))

    input_word = lemmatized_words
    # print(input_word)

    # Removing Duplicates and Sorting the alphabetical order Words
    input_word = sorted(set(input_word))
    tags = sorted(set(tags))

    # Write-binary and saved to files
    pickle.dump(input_word, open('words.pkl', 'wb'))
    pickle.dump(tags, open('tags.pkl', 'wb'))  # 'wb': Write-binary mode

    training = []
    # tag length wise create arry [0,0..]
    output_empty = [0] * len(tags)

    for inputAndTags in input_and_tag:
        bag = []
        inputs_tokeniz_words = inputAndTags[0]
        word_patterns = []
        for word in inputs_tokeniz_words:
            # word to lowercase
            lowercased_word = word.lower()
            # Lemmatize the lowercase word
            lemmatized_word = lemmatizer.lemmatize(lowercased_word)
            # Append  word to the word_patterns list
            word_patterns.append(lemmatized_word)

        for word in input_word:
            # appent bag arry
            bag.append(1) if word in word_patterns else bag.append(0)

            # This creates a new list with the same length as output_empty[0,0..]
            output_row = list(output_empty)
            # inputAndTags[1]=tag
            # searches for the index of the value inputAndTags[1] within the tags list
            # When it finds a match, it returns the 1
            output_row[tags.index(inputAndTags[1])] = 1
            training.append([bag, output_row])

    # x= training[0] bag values [0,0,1..]
    # y= training[1] output_row
    # get the max lenth of x
    max_length = max(len(x) for x, y in training)
    # print(max_length)

    # Pad sequences in training

    # enumerate in training arry  eg[1 [0,0,1..][0,1,0..],2..]
    for i, (x, y) in enumerate(training):
        padding_length_x = max_length - len(x)
        padding_length_y = max_length - len(y)
        if padding_length_x > 0:
            x += [0] * padding_length_x
        if padding_length_y > 0:
            y += [0] * padding_length_y
        # train aryy index wise append bag arrys and output_row arrys [[[0, 0], [0, 0, 1, 0, 0, 0]],...]
        training[i] = (x, y)  # xpadded to the same length (max_length)

    # Shuffle training data
    random.shuffle(training)

    # Convert training data to numpy array [[[0 0 0 ... 1 0 0][0 1 0 ... 0 0 0]]
    training = np.array(training)

    # Extract train_x and train_y
    train_x = list(training[:, 0])  # Selects all rows in the training first column
    train_y = list(training[:, 1])  # Selects all rows in the training second column

    # ---model (Neural Network Architecture)--

    #Initializes a new Sequential model, which is a linear stack of  layers-by-layer.(dense and dropout layers)
    model = Sequential()
    #neurons in the dense layer will learn 256 different transformations of these (bag of length vlue)  features
    model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
    #  Dropout randomly drops out 50% of the neurons during training to prevent overfitting
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    #line adds the output layer of the model.
    model.add(Dense(len(train_y[0]), activation='softmax'))

    #  optimizer adjusts the weights of the connections in the model to minimize the loss
    optimizer = Adam(learning_rate=0.001)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define model checkpoint to save the best model
    model_checkpoint = ModelCheckpoint('chatbot_model.keras', save_best_only=True)

    # Define early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    # Train the model with early stopping and model checkpoint
    hist = model.fit(
        np.array(train_x), np.array(train_y),
        epochs=200, batch_size=5, verbose=1,
        validation_split=0.2,
        callbacks=[model_checkpoint, early_stopping]
    )


    # Plots the training and validation accuracy and loss curves to visualize model performance
    plt.plot(hist.history['accuracy'], label='training accuracy')
    plt.plot(hist.history['val_accuracy'], label='validation accuracy')
    plt.plot(hist.history['loss'], label='training loss')
    plt.plot(hist.history['val_loss'], label='validation loss')
    plt.legend()
    plt.savefig('assets/accuracy/training_metrics.png')
    plt.close()
    return "success"
