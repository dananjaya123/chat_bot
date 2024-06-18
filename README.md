NLP Chatbot Project
================================
This project is an NLP-based chatbot built with Flask and TensorFlow's Keras. 
The chatbot is designed to train on a dataset of intents and respond to user inputs based on the trained model.

============== Installation ===================
1. Clone the repository:
   ```
   git clone https://github.com/dananjaya123/chat_bot.git
   ```
3. Create a virtual environment:
   ```
   python -m venv chat_bot
   source venv/bin/activate  # On Windows, use `chat_bot\Scripts\activate`
   ```
3.Libraries
 - Flask:
   ```
   pip install Flask
   ```
 - NLTK :
   ```
   pip install nltk
   ```
 - NumPy :
   ```
   pip install numpy
   ```
 - TensorFlow Keras :
   ```
   pip install tensorflow
   ```
 - Matplotlib :
   ```
   pip install matplotlib
   ```

4. Running the Application:
- Start the Flask Server:
  ```
  python main.py
  ```
- Open your browser and navigate to:
  ```
  http://localhost:5000/
  ```


============== Endpoints ===================
- **GET /**: Render the homepage.
- **GET /login**: Render the login page.
- **GET /admin**: Render the admin panel.
- **POST /train**: Train the model and save it.
- **POST /predict**: Predict the intent of a given message and respond accordingly.

============== Project Structure ===================

- **app.py**: Main Flask application file.
- **train.py**: Script for training the model.
- **chatbot.py**: Script for handling the chatbot responses.
- **assets/**: Directory for storing static files such as images.
- **templates/**: Directory for HTML templates.
- **content.json**: Dataset file for training the model.
- **words.pkl** and **tags.pkl**: Pickle files for preprocessed data.
- **chatbot_model.keras**: Saved trained model.

![Chat_bot](assets/img/chat.png)

