import json
import numpy as np
import gensim
import nltk
nltk.data.path.append("./nltk_data/")
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import streamlit as st


# Load the intents from the JSON file
with open('intents.json') as file:
    data = json.load(file)

# Preprocess the data
lemmatizer = WordNetLemmatizer()
intents = data['intents']
tags = []
documents = []
responses = []
for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        documents.append(tokens)
        responses.append(tag)

# Train the LDA model to get topic distributions for each document
dictionary = gensim.corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=50, id2word=dictionary)

# Convert the topic distributions into features
lda_features = []
for document in documents:
    bow = dictionary.doc2bow(document)
    topic_dist = np.array([score for _, score in lda_model.get_document_topics(bow)])
    lda_features.append(topic_dist)

# Convert the responses into one-hot encoded labels
labels = np.zeros((len(responses), len(tags)))
for i, tag in enumerate(tags):
    labels[:, i] = np.array(responses) == tag

# Split the data into training and testing sets
train_size = int(0.8 * len(lda_features))
train_features = lda_features[:train_size]
train_labels = labels[:train_size, :]
test_features = lda_features[train_size:]
test_labels = labels[train_size:, :]

# Build and train the LSTM model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(doc) for doc in documents)
train_sequences = tokenizer.texts_to_sequences(documents[:train_size])
test_sequences = tokenizer.texts_to_sequences(documents[train_size:])
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

def create_lstm_model(units=128, dropout=0.2):
    lstm_model = Sequential()
    lstm_model.add(Embedding(vocab_size, 128, input_length=max_len))
    lstm_model.add(LSTM(units, dropout=dropout))
    lstm_model.add(Dense(len(tags), activation='softmax'))

    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return lstm_model

param_grid = {'units': [64, 128, 256], 'dropout': [0.2, 0.3, 0.4]}
best_lstm_accuracy = 0.0
best_lstm_params = {'units': param_grid['units'][0], 'dropout': param_grid['dropout'][0]}

for units in param_grid['units']:
    for dropout in param_grid['dropout']:
        lstm_model = create_lstm_model(units=units, dropout=dropout)
        lstm_model.fit(train_padded, train_labels, epochs=1000, batch_size=32, verbose=0)
        _, accuracy = lstm_model.evaluate(test_padded, test_labels, verbose=0)
        if accuracy > best_lstm_accuracy:
            best_lstm_accuracy = accuracy
            best_lstm_params = {'units': units, 'dropout': dropout}

# Train the LSTM model with the best parameters
best_lstm_model = create_lstm_model(units=best_lstm_params['units'], dropout=best_lstm_params['dropout'])
best_lstm_model.fit(train_padded, train_labels, epochs=10, batch_size=32, verbose=0)
best_lstm_accuracy = best_lstm_model.evaluate(test_padded, test_labels, verbose=0)[1]

# Build and train the SVM model

# Pad sequences in train_features and test_features
train_features = pad_sequences(train_features)
test_features = pad_sequences(test_features)
# Convert train_features and test_features to numpy arrays
train_features = np.array(train_features)
test_features = np.array(test_features)

# Reshape train_features and test_features to be 2-dimensional arrays
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)

svm_model = SVC()
svm_model.fit(train_features, np.argmax(train_labels, axis=1))

# Evaluate the SVM model
svm_accuracy = svm_model.score(test_features, np.argmax(test_labels, axis=1))

# Build and train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(train_features, np.argmax(train_labels, axis=1))

# Evaluate the Random Forest model
rf_accuracy = rf_model.score(test_features, np.argmax(test_labels, axis=1))

# Comparing the models
model_comparison = {
    'SVM': svm_accuracy,
    'Random Forest': rf_accuracy,
    'LSTM': best_lstm_accuracy
}

best_model = max(model_comparison, key=model_comparison.get)

# Streamlit UI
def get_bot_response(message):
    tokens = nltk.word_tokenize(message.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    bow = dictionary.doc2bow(tokens)
    topic_dist = np.array([score for _, score in lda_model.get_document_topics(bow)])
    features = topic_dist.reshape(1, -1)
    intent = tags[svm_model.predict(features)[0]]
    for intent_data in intents:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            return np.random.choice(responses)

st.title('CS Chatbot')

user_input = st.text_input('You:')

if st.button('Send'):
    bot_response = get_bot_response(user_input)
    st.text(f'Bot: {bot_response}')

st.text(f'Best Model: {best_model}')
st.text('Model Comparison:')
for model, accuracy in model_comparison.items():
    st.text(f'{model}: {accuracy}')
