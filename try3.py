import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

# Load intents from JSON
with open('intents.json') as file:
    data = json.load(file)

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()
words = []
labels = []
documents = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))

# Create training data
training = []
output = []
output_empty = [0] * len(labels)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy arrays
np.random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Convert train_y to a list of tuples
train_y = np.argmax(train_y, axis=1)

# Handle missing labels
missing_labels = set(labels) - set(train_y)
for missing_label in missing_labels:
    output_row = list(output_empty)
    output_row[labels.index(missing_label)] = 1
    training = np.concatenate((training, [[[0] * len(words), output_row]]))

# Shuffle the updated training data
np.random.shuffle(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Split the training data into train and test
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.3)

# Create a dictionary to store the models
models = {
    "Neural Network": Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(labels), activation='softmax')
    ]),

    "LSTM": Sequential([
        Embedding(len(words), 128, input_length=len(train_x[0])),
        LSTM(128),
        Dense(len(labels), activation='softmax')
    ])
}

# Train the models
for model_name, model in models.items():
    print(f"Training {model_name}...")
    if model_name == "Neural Network":
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=16, verbose=1, validation_split=0.1, callbacks=[early_stopping])

    elif model_name == "LSTM":
        tokenizer = Tokenizer(num_words=len(words))
        tokenizer.fit_on_texts([str(x) for x in train_x])  # Convert train_x to strings
        train_sequences = tokenizer.texts_to_sequences([str(x) for x in train_x])  # Convert train_x to strings
        train_sequences_padded = pad_sequences(train_sequences, maxlen=len(train_x[0]), padding='post')
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(np.array(train_sequences_padded), np.array(train_y), epochs=200, batch_size=16, verbose=1, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the models
for model_name, model in models.items():
    print(f"\n{model_name} Evaluation:")
    if model_name == "Neural Network" or model_name == "LSTM":
        if model_name == "Neural Network":
            _, acc = model.evaluate(np.array(test_x), np.array(test_y), verbose=0)
        elif model_name == "LSTM":
            test_sequences = tokenizer.texts_to_sequences([str(x) for x in test_x])  # Convert test_x to strings
            test_sequences_padded = pad_sequences(test_sequences, maxlen=len(train_x[0]), padding='post')
            _, acc = model.evaluate(np.array(test_sequences_padded), np.array(test_y), verbose=0)
        print("Accuracy:", acc)
