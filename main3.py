import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import time
import difflib
import numpy
import tflearn
import tensorflow as tf
import json
import pickle
import random
import os
import streamlit as st
import streamlit_theme as stt
import pyaudio
import glob
import shutil
import speech_recognition as sr
from helpers import is_computer_science_related
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Model

from collections import Counter
import os
import nltk

# Load intents data from JSON file
with open("intents.json") as file:
    data = json.load(file)

try:
    # Try to load preprocessed data from the pickle file
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # If the pickle file doesn't exist or there's an error, preprocess the data
    words = []
    labels =[]
    docs_patt = []
    docs_tag = []
    
    # Iterate through each intent and its patterns in the data
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize the pattern into words
            wrds = nltk.word_tokenize(pattern)
            for item in wrds:
                # Extend the words list with the individual words
                words.extend(wrds)
                # Append the pattern and corresponding intent tag to their respective lists
                docs_patt.append(wrds)
                docs_tag.append(intent["tag"])
                if intent["tag"] not in labels:
                    labels.append(intent["tag"])
    
    # Stem and preprocess the words list
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    
    # Create training data and output data using bag-of-words approach
    for x, doc in enumerate(docs_patt):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_tag[x])] = 1
        training.append(bag)
        output.append(output_row)
    
    # Convert the training and output lists to NumPy arrays
    training = numpy.array(training)
    output = numpy.array(output)
    
    # Save the preprocessed data to the pickle file for future use
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


from tensorflow.python.framework import ops
ops.reset_default_graph()

# Define the neural network architecture using tflearn
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    # Try to load the trained model from the saved file
    model.load("model.tflearn")
except:
    # If the trained model doesn't exist or there's an error, train a new model
    model = tflearn.DNN(net)
    history = model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    # Convert a sentence into a bag-of-words representation using a given vocabulary
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def words_to_list(s):
    # Convert a sentence into a list of words
    a = []
    ns = ""
    s = s + " " 
    for i in range(len(s)):
        if s[i] == " ":
            a.append(ns)
            ns = ""
        else:
            ns = ns + s[i]
    a = list(set(a))
    return a

def json_to_dictionary(data):
    # Convert intents data into a vocabulary list
    dictionary = []
    fil_dict= []
    vocabulary = []
    for i in data["intents"]:
        for pattern in i["patterns"]:
            vocabulary.append(pattern.lower())
    for i in vocabulary:
        dictionary.append(words_to_list(i))
    for i in range(len(dictionary)):
        for word in dictionary[i]:
            fil_dict.append(word)
    return list(set(fil_dict))

# Create a vocabulary list for the chatbot
chatbot_vocabulary = json_to_dictionary(data)

def word_checker(s):
    # Check and correct words in a sentence based on the chatbot's vocabulary
    correct_string = ""
    for word in s.casefold().split():
        if word not in chatbot_vocabulary:
            # Get a suggested word if the current word is not in the vocabulary
            suggestion = difflib.get_close_matches(word, chatbot_vocabulary)
            for x in suggestion:
                pass
            if len(suggestion) == 0:
                pass
            else:
                correct_string = correct_string + " " + str(suggestion[0])
        else:
            correct_string = correct_string + " " + str(word)
    return correct_string

MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
DATA_DIR_PATH = 'data'
MAX_VOCAB_SIZE = 30000

# Define markers for special tokens
marker_start = '<begin>'
marker_end = '<end>'
marker_unknown = '<unk>'
marker_pad = '<pad>'

# Set the input and output sequence lengths
input_seq_len = 15
output_seq_len = input_seq_len + 2

# Initialize dictionaries for word-to-index and index-to-word mappings
input_w2i, input_i2w, target_w2i, target_i2w = {}, {}, {}, {}

# Initialize special tokens in the input word-to-index dictionary
input_w2i[marker_unknown] = 0
input_w2i[marker_pad] = 1

# Count occurrences of words in the input and target data
target_counter = Counter()
input_counter = Counter()

# Filter out rare words based on the maximum vocabulary size
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_w2i[word[0]] = idx + 2

# Create the inverse dictionary for input vocabulary index-to-word mapping
input_i2w = dict([(idx, word) for word, idx in input_w2i.items()])

# Initialize special tokens in the target word-to-index dictionary
target_w2i[marker_unknown] = 0
target_w2i[marker_pad] = 1
target_w2i[marker_start] = 2
target_w2i[marker_end] = 3

# Assign unique indices to the target words based on their occurrence frequency
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_w2i[word[0]] = idx + 4

# Create the inverse dictionary for target vocabulary index-to-word mapping
target_i2w = dict([(idx, word) for word, idx in target_w2i.items()])

# Get the lengths of input and target vocabularies
inputVocabLen = len(input_w2i)
targetVocabLen = len(target_w2i)

# Define the encoder and decoder inputs as placeholders
encoder_inputs = [tf.keras.Input(dtype=tf.int32, shape=[None], name='encoder{}'.format(i)) 
                  for i in range(input_seq_len)]
decoder_inputs = [tf.keras.Input(dtype=tf.int32, shape=[None], name='decoder{}'.format(i)) 
                  for i in range(output_seq_len)]

# Prepare the target inputs and weights for training
targets = [decoder_inputs[i + 1] for i in range(output_seq_len - 1)]
targets.append(tf.keras.Input(dtype=tf.int32, shape=[None], name='last_target'))
target_weights = [tf.keras.Input(dtype=tf.float32, shape=[None], name='target_w{}'.format(i)) 
                  for i in range(output_seq_len)]

# Define the output projection variables
size = 512
targetVocabLen = 1000
w_t = tf.Variable(tf.random.truncated_normal([targetVocabLen, size], stddev=0.1), name='proj_w')
b = tf.Variable(tf.zeros([targetVocabLen]), name='proj_b')
w = tf.transpose(w_t)
output_projection = (w, b)

# Define the model inputs
encoder_inputs = tf.keras.Input(shape=(None,))
decoder_inputs = tf.keras.Input(shape=(None,))

# Define the embedding layers
embedding_size = 100
encoder_embedding = tf.keras.layers.Embedding(inputVocabLen, embedding_size)
decoder_embedding = tf.keras.layers.Embedding(targetVocabLen, embedding_size)

# Embed the inputs
encoder_inputs_embedded = encoder_embedding(encoder_inputs)
decoder_inputs_embedded = decoder_embedding(decoder_inputs)

# Define the encoder LSTM
encoder_lstm = tf.keras.layers.LSTM(size, return_state=True)
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs_embedded)

# Define the decoder LSTM
decoder_lstm = tf.keras.layers.LSTM(size, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_embedded, initial_state=[encoder_state_h, encoder_state_c])

# Define the output layer
output_projection = tf.keras.layers.Dense(targetVocabLen)

# Apply the output projection to the decoder outputs
outputs = output_projection(decoder_outputs)

# Define the model
targets = tf.keras.Input(dtype=tf.int32, shape=(None, None))
target_weights = tf.keras.Input(dtype=tf.float32, shape=(None, None))
loss = tf.reduce_mean(
    tf.keras.metrics.sparse_categorical_crossentropy(targets, outputs) * target_weights
)

# Define training parameters
learning_rate = 7e-3
batch_size = 96
steps = 25501


# Initialize the speech recognizer
r = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function for the bot to speak
def bot_speaking(message):
    engine.say(message)
    engine.runAndWait()
    if engine._inLoop:
        engine.endLoop()

# Function to get user input through microphone
def get_input():
    with sr.Microphone() as source:
        # Prompt the user to say something
        bot_speaking("Hey mate, say something")
        
        # Listen to the audio input
        audio = r.listen(source, timeout=0)
        
        # Provide feedback to the user
        bot_speaking("Perfect, Thanks!")
    
    try:
        # Recognize the speech using Google Speech Recognition
        msg = r.recognize_google(audio)
        
        # Print the recognized text
        print("TEXT: " + msg)
        
        # Speak out the recognized text
        bot_speaking("You said: " + msg)
        
        return msg
    except:
        # If speech recognition fails, inform the user
        bot_speaking("Sorry mate! It's not working")
        pass


import openai
openai.api_key = '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'

def get_response(msg):
    while True:
        inp = msg
        
        # Check if the user wants to quit or if there is no input
        if inp.lower() == "quit" or inp is None:
            break
        
        # Check and correct words in the input using the word_checker function
        inp_x = word_checker(inp)
        
        # Predict the intent of the user's input using the trained model
        results = model.predict([bag_of_words(inp_x, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        # Check if the query is related to computer science
        is_computer_science_query = is_computer_science_related(msg)
        
        if is_computer_science_query:
            # Generate a chatbot response based on the user's message and computer science context
            context = "You are a student interested in computer science. You ask: " + msg
            response = openai.Completion.create(
                engine='text-davinci-003',
                prompt=context,
                max_tokens=1000,
                temperature=0.7,
                n=1,
                stop=None
            )
            return response.choices[0].text.strip()
        elif tag in ["greeting", "goodbye", "thanks", "options"]:
            # Use the trained model to generate a response based on specific intent tags
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    ms = random.choice(responses)
                    return ms
        elif "roadmap" in msg.lower() or "open source tools" in msg.lower():
            # Use the trained model to generate a response based on specific keywords in the user's message
            inp_x = word_checker(inp)
            results = model.predict([bag_of_words(inp_x, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            if results[results_index] >= 0.9:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                        ms = random.choice(responses)
                        return ms
        else:
            # Provide a default response if the query is not related to computer science
            return "Arto chatbot is programmed to respond to queries related to computer science only."

    
def app():
    # Set the configuration for the Streamlit app
    st.set_page_config(
        page_title="Arto Chatbot",
        page_icon="asset\ARTO.ico",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.google.com/',
            'Report a bug': 'https://www.google.com/',
            'About': "# chatbot app"
        })

    # Display the header image
    header_image = 'header_image.png'
    st.image(header_image, width=96)

    # Set the app's header
    st.header("Arto Chatbot")


    # Set the app's background color and font
    st.markdown(
        """
        <style>
            body {
                background-color: #1f243a;
                font-family: sans-serif;
            }
            .reportview-container {
            background: #1f243a
            }

            .sidebar .sidebar-content {
            background: #1f243a
            }


            .chat-wrapper {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                justify-content: flex-start;
                margin-top: 10px;
                margin-bottom: 10px;
            }

            .loader {
            border: 16px solid #f3f3f3; 
            border-top: 16px solid #3498db; 
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            }
            
            @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
            }

            .chat-bubble {
            background-color: #f1f1f1;
            border-radius: 18px;
            padding: 10px;
            margin: 10px 0;
            position: relative;
            max-width: 60%;
            height: auto;
            }
            
            .chat-bubble::after {
            content: "";
            clear: both;
            display: table;
            }
            
            .user-bubble {
            background-color: #2196F3;
            color: white;
            align-self:flex-end;
            margin-left: 60%;

            }
            
            .user-bubble::after {
            content: "";
            position: absolute;
            bottom: 12px;
            right: -15px;
            border-style: solid;
            border-width: 15px 0 15px 20px;
            border-color: transparent transparent transparent #2196F3;
            }

            .chat-bubble.chat-bot {
            background-color: #e9e9eb;
            color: black;
            align-self:flex-end;
            margin-right: 60%;

            }
            
            .chat-bubble.chat-bot::after {
            content: "";
            position: absolute;
            top: 12px;
            left: -15px;
            border-style: solid;
            border-width: 15px 20px 15px 0;
            border-color: transparent #e9e9eb transparent transparent;
            }


            /* Sidebar styling */
            .sidebar .sidebar-content {
                background-color: #f1f1f6;
            }

            .sidebar .sidebar-content a {
                color: #1f243a;
            }

            .sidebar .sidebar-content a:hover {
                color: #1f243a;
            }            

            
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to scroll the chat history to the bottom using JavaScript
    def scroll_to_bottom():
        scroll_script = """
            <script>
            var chatHistory = document.getElementById("chat-history");
            chatHistory.scrollTop = chatHistory.scrollHeight;
            </script>
        """
        st.markdown(scroll_script, unsafe_allow_html=True)


    #Create a folder for storing chat history files
    if not os.path.exists("chat history"):
        os.makedirs("chat history")

    # Get all chat history files
    chat_history_files = [f for f in os.listdir("chat history") if f.endswith('.json')]

    # Select chat history file
    selected_file = st.sidebar.selectbox('Select chat history file', chat_history_files)

    # Load the selected chat history from a file
    if selected_file:
        file_path = os.path.join("chat history", selected_file)
        if os.stat(file_path).st_size == 0:
            chat_history_list = []
        else:
            with open(file_path, "r") as f:
                chat_history_list = json.load(f)
    else:
        chat_history_list = []

    # Create a placeholder for the chat history
    chat_history = st.empty()

    # Create a placeholder for the user input
    user_input = st.text_input("User Input", "")

    # Create a button to submit user input
    submit_button = st.button("Send")
    new_chat_button = st.button("New Chat")
    delete_history_button = st.button("Delete All Chat History")

    # If new chat button is clicked
    if new_chat_button:
        # Prompt user to enter file name for chat history and store it in new_file_name variable
        new_file_name = st.text_input("Enter chat history file name", "") or "default"
        # Create a path for the file with provided name
        file_path = os.path.join("chat history", new_file_name + ".json")

        # Check if a file with the same name already exists 
        if os.path.isfile(file_path):
            # Display error message when file with same name already exists
            st.error("A file with the same name already exists.")
        else:
            try:
                # Create a new empty chat history file with the given name
                with open(file_path, "w") as f:
                    json.dump([], f)
                
                # Refresh the list of available chat history files
                chat_history_files = [f for f in os.listdir("chat history") if f.endswith('.json')]
                
                # Show success message with file path if file was created successfully
                st.success(f"Chat history file {new_file_name} created successfully at {file_path}!")
                
                # Set newly created file as selected file and initialize an empty array to hold chat history
                selected_file = new_file_name + ".json"
                chat_history_list = []
            except (IOError, ValueError) as e:
                # Display error message if an exception is thrown while creating the file
                st.error(f"Error occurred: {str(e)}")
    if delete_history_button:
            # Delete all chat history files
            try:
                shutil.rmtree("chat history")
                os.makedirs("chat history")
                st.success("All chat history files deleted successfully!")
                # Refresh the list of chat history files
                chat_history_files = [f for f in os.listdir("chat history") if f.endswith('.json')]
            except (OSError, IOError) as e:
                st.error(f"Error occurred while deleting chat history files: {str(e)}")

            # Refresh the list of chat history files
            chat_history_files = [f for f in os.listdir("chat history") if f.endswith('.json')]
    if submit_button:
        try:
            # Get user input and add it to chat history list
            user_message = user_input.strip()
            chat_history_list.append(('user', user_message))

            # Send user input to chatbot for processing
            chatbot_output = get_response(user_message)

            # Add chatbot response to chat history
            chat_history_list.append(('chatbot', chatbot_output))

            # Clear user input
            user_input = ""

            # Save the updated chat history to the selected file
            with open(os.path.join("chat history", selected_file), "w") as f:
                json.dump(chat_history_list, f)

        except Exception as e:
            # Handle exceptions gracefully by displaying an error message
            error_msg = f"Error occurred: {str(e)}"
            chat_history_list.append(('error', error_msg))

        # Display chat history
        conversation_html = "<div class='chat-wrapper'>"
        for msg_type, msg_text in reversed(chat_history_list):
            if msg_type == 'user':
                # Display the user's input in the chat history, aligned to the right
                user_bubble = f"<div class='chat-bubble user-bubble align-right'>{msg_text}</div><br>"
                conversation_html = user_bubble + conversation_html
            elif msg_type == 'chatbot':
                # Display chatbot's response in chat history, aligned to the left
                bot_bubble = f"<div class='chat-bubble chat-bot align-left'>{msg_text}</div><br>"
                conversation_html = bot_bubble + conversation_html
            elif msg_type == 'error':
                # Display error message in chat history, aligned to the right
                error_bubble = f"<div class='chat-bubble error-bubble align-right'>{msg_text}</div><br>"
                conversation_html = error_bubble + conversation_html
            conversation_html += "</div>"
        chat_history.markdown(conversation_html, unsafe_allow_html=True)
        # Scroll to the bottom of the chat history
        scroll_to_bottom()


    # About section
    st.sidebar.markdown("<p style='text-align: center;'><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>This project was crafted with 🤍 by<br>Pulkit, Manan, Prince and Roshni<br><br></p> ",unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>Copyright © 2023 Arto Chatbot. All rights reserved.</p>",unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    app()