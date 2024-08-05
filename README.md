# Corona_bot

This project involves developing a chatbot designed to answer questions about COVID-19. It provides users with information related to COVID-19, such as symptoms, prevention measures, and general knowledge about the virus.It uses various technologies and libraries including TensorFlow, Keras, NLTK, and Python.

## Key Components

### Data Preprocessing

- **Natural Language Toolkit (NLTK):** Used for tokenizing sentences and stemming words. The `LancasterStemmer` is employed to reduce words to their root forms.
- **JSON Data:** The dataset containing various intents (e.g., greetings, questions about COVID-19) and corresponding patterns and responses is stored in a JSON file.

### Model Training

- **Word and Class Extraction:** The script processes the input data to create a list of unique words and associated tags.
- **Bag of Words:** Each sentence is converted into a bag of words (a list representing the presence or absence of words) for training purposes.
- **Neural Network:** A Sequential model from Keras with three dense layers is used. The first two layers use the ReLU activation function, and the final layer uses the softmax activation function to classify the input into one of the defined intents.

### Training and Saving the Model

- The model is trained using stochastic gradient descent (SGD) with a learning rate of 0.01.
- The training process involves fitting the model to the prepared training data and saving the trained model to a file (`model.h5`).

### Chat Functionality

- **User Interaction:** The chatbot continuously interacts with the user, accepting input queries.
- **Prediction:** The input query is converted into a bag of words, and the model predicts the intent. If the confidence of the prediction is above a certain threshold (e.g., 70%), the chatbot responds with a corresponding answer from the dataset.
- **Response Generation:** The chatbot provides appropriate responses based on the predicted intent.

## Technologies and Libraries

- **TensorFlow and Keras:** For building and training the neural network model.
- **NLTK:** For natural language processing tasks like tokenization and stemming.
- **NumPy:** For handling numerical data and converting lists to arrays.
- **Pickle:** For saving and loading preprocessed data to avoid redundant processing.
- **JSON:** For storing and loading the intents and responses.

## Example Usage

When a user interacts with the chatbot, they can ask questions like "What are the symptoms of COVID-19?" or "How does COVID-19 spread?" The chatbot processes the input, predicts the relevant intent, and provides an accurate response based on the trained model.


![Screenshot (1041)](https://github.com/user-attachments/assets/e650ef3b-4e0e-480f-88a0-33726cd4382e)
