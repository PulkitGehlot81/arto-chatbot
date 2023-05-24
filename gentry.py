import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Model

from collections import Counter
import os
import nltk
alphas = 'abcdefghijklmnopqrstuvwxyz1234567890'

MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
DATA_DIR_PATH = 'data'
MAX_VOCAB_SIZE = 30000

marker_start = '<begin>'
marker_end = '<end>'
marker_unknown = '<unk>'
marker_pad = '<pad>'

# standard step - reset computation graphs
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 2 more for start and stop markers
input_seq_len = 15
output_seq_len = input_seq_len+2

#defines permissible characters for the chatbot
def permissible_chars(word):

    for char in word:
        if char in alphas:
            return True

    return False

# Compute softmax values for each sets of scores in x.
def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# To parse the input yml files and create word2index and index2word mappings

target_counter = Counter()
input_counter = Counter()

input_texts = []
target_texts = []

# Parser base code of GuntherCox dataset obtained from link below and modified as per requirement.
# https://github.com/kushagra2101/ChatCrazie/blob/master/train_seq2seq.py

for file in os.listdir(DATA_DIR_PATH):
    filepath = os.path.join(DATA_DIR_PATH, file)
    if os.path.isfile(filepath):
        print('processing file: ', file)
        lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
        prev_words = []
        for line in lines:

            if line.startswith('- - '):
                prev_words = []

            if line.startswith('- - ') or line.startswith('  - '):
                line = line.replace('- - ', '')
                line = line.replace('  - ', '')
                next_words = [w.lower() for w in nltk.word_tokenize(line)]
                next_words = [w for w in next_words if permissible_chars(w)]
                if len(next_words) > MAX_TARGET_SEQ_LENGTH:
                    next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

                if len(prev_words) > 0:
                    input_texts.append(prev_words)
                    for w in prev_words:
                        input_counter[w] += 1

                    target_words = next_words[:]
                    for w in target_words:
                        target_counter[w] += 1
                    target_texts.append(target_words)

                prev_words = next_words


for idx, (input_words, target_words) in enumerate(zip(input_texts, target_texts)):
    if idx < 20:
        print([input_words, target_words])

input_w2i, input_i2w, target_w2i, target_i2w = {},{},{},{}

### Creating Word2index and Index2word, forward and reverse mapping ###
## we will create dictionaries to provide a unique integer for each word.
input_w2i[marker_unknown] = 0
input_w2i[marker_pad] = 1
# filter out the rare words
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_w2i[word[0]] = idx+2

# inverse dictionary for vocab_to_int.
input_i2w = dict([(idx, word) for word, idx in input_w2i.items()])

## we will create dictionaries to provide a unique integer for each word.
target_w2i[marker_unknown] = 0
target_w2i[marker_pad] = 1
target_w2i[marker_start] = 2
target_w2i[marker_end] = 3
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_w2i[word[0]] = idx+4

# inverse dictionary for vocab_to_int.
target_i2w = dict([(idx, word) for word, idx in target_w2i.items()])


###########################################

# inputVocabLen = len(input_word2idx)
# targetVocabLen = len(target_word2idx)

###########################################

# if the word is not found then default with 0. 
# 0 in index means the word is unknown (<unk>)
x = [[input_w2i.get(word, 0) for word in sentence] for sentence in input_texts]
y = [[target_w2i.get(word, 0) for word in sentence] for sentence in target_texts]

inputVocabLen = len(input_w2i)
targetVocabLen = len(target_w2i)

## Padding and Splitting

# Pad all the sequences to same length
for i in range(len(x)):

    if (len(x[i]) > input_seq_len):
        x[i] = x[i][:input_seq_len-1]

    # Fill in with padding marker
    for k in range(input_seq_len - len(x[i])):
        x[i] = x[i] + [input_w2i[marker_pad]]
            
    if (len(y[i]) > output_seq_len-2):
        y[i] = y[i][:output_seq_len-3]

    # Add end and begin marker
    y[i] = [target_w2i[marker_start]] + y[i] + [target_w2i[marker_end]]

    # Fill in with padding marker
    for k in range(output_seq_len - len(y[i])):
        y[i] = y[i] + [input_w2i[marker_pad]]

    if (i < 10):
        print(x[i])
        print(y[i])

        
# Train Test Split
X_train,  X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.05)
## Model Helper Functions Batch feeding, Decoding & Loss
# Stub code sourced from Neural machine translator for English2German translation
# https://github.com/Nemzy/language-translation. Modified to suit requirements.

# feed data into placeholders
def feed_dict(x, y, batch_size = 64):
    feed = {}
    
    idxes = np.random.choice(len(x), size = batch_size, replace = False)
    
    for i in range(input_seq_len):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes], dtype = np.int32)
        
    for i in range(output_seq_len):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes], dtype = np.int32)
        
    feed[targets[len(targets)-1].name] = np.full(shape = [batch_size], fill_value = target_w2i[marker_pad], dtype = np.int32)
    
    for i in range(output_seq_len-1):
        batch_weights = np.ones(batch_size, dtype = np.float32)
        target = feed[decoder_inputs[i+1].name]
        for j in range(batch_size):
            if target[j] == target_w2i[marker_pad]:
                batch_weights[j] = 0.0
        feed[target_weights[i].name] = batch_weights
        
    feed[target_weights[output_seq_len-1].name] = np.zeros(batch_size, dtype = np.float32)
    
    return feed

# define our loss function

# sampled softmax loss - returns: A batch_size 1-D tensor of per-example sampled softmax losses
def sampled_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
                        weights = w_t,
                        biases = b,
                        labels = tf.reshape(labels, [-1, 1]),
                        inputs = logits,
                        num_sampled = 512,
                        num_classes = targetVocabLen)

# decode output sequence
def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(target_i2w[idx])
    return words
## Model Building
# Stub code sourced from Neural machine translator for English2German translation
# https://github.com/Nemzy/language-translation. Modified to suit requirements.

# Defining placeholders
# The first None means the batch size, and the batch size is unknown since user can set it. 
# The second None means the lengths of sentences. 

encoder_inputs = [tf.keras.Input(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) 
                  for i in range(input_seq_len)]
decoder_inputs = [tf.keras.Input(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) 
                  for i in range(output_seq_len)]

targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]
# add one more target
targets.append(tf.keras.Input(dtype = tf.int32, shape = [None], name = 'last_target'))
target_weights = [tf.keras.Input(dtype = tf.float32, shape = [None], 
                                 name = 'target_w{}'.format(i)) for i in range(output_seq_len)]

# output projection
size = 512
targetVocabLen = 1000
w_t = tf.Variable(tf.random.truncated_normal([targetVocabLen, size], stddev=0.1), name='proj_w')
b = tf.Variable(tf.zeros([targetVocabLen]), name='proj_b')
w = tf.transpose(w_t)
output_projection = (w, b)



# Define the model inputs
encoder_inputs = tf.keras.Input(shape=(None,))
decoder_inputs = tf.keras.Input(shape=(None,))

# Define the embedding layer
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
model = Model([encoder_inputs, decoder_inputs], outputs)

# Weighted cross-entropy loss for a sequence of logits
targets = tf.keras.Input(dtype=tf.int32, shape=(None, None))
target_weights = tf.keras.Input(dtype=tf.float32, shape=(None, None))
loss = tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(targets, outputs) * target_weights)
## Training & Plotting
# Stub code sourced from Neural machine translator for English2German translation
# https://github.com/Nemzy/language-translation. Modified to suit requirements.

# ops and hyperparameters
learning_rate = 7e-3
batch_size = 96
steps = 25501

# ops for projecting outputs
#outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

# training op
optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate).minimize(loss)
# tf.train.RMSPropOptimizer

# init op
init = tf.Variable()


# Loss values appended to plot diagram
losses = []

# Save checkpoint to restore the model later 
checkpoint = tf.train.Checkpoint()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(steps):
        feed = feed_dict(X_train, Y_train, batch_size)
        sess.run(optimizer, feed_dict = feed)
        
        if step % 500 == 0:
            loss_value = sess.run(loss, feed_dict = feed)
            print('step: {}, loss: {}'.format(step, loss_value))
            losses.append(loss_value)
            
    checkpoint.save(sess, 'checkpoints/', global_step=step)
    print('Checkpoint is saved')
    
# plot the losses
with plt.style.context('fivethirtyeight'):
    plt.plot(losses, linewidth = 1)
    plt.xlabel('Steps')
    plt.ylabel('Losses')
plt.show()
## Generate Response Prediction
# To predict response (inference) use the same model as defined above with forward feed

def generateReply(humanMsg):
    
    if (len(humanMsg) == 0):
        return ''
    
    with tf.Graph().as_default():
        
        replyMsg = ""

        # same format as in model building
        encoder_inputs = [tf.keras.Input(dtype = tf.int32, shape = [None], 
                                         name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
        decoder_inputs = [tf.keras.Input(dtype = tf.int32, shape = [None], 
                                         name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

        # output projection
        size = 512
        w_t = tf.Variable('proj_w', [targetVocabLen, size], tf.float32)
        b = tf.Variable('proj_b', [targetVocabLen], tf.float32)
        w = tf.transpose(w_t)
        output_projection = (w, b)

        # feed_previous is set to true so that output at time t can be fed as input at time t+1
        outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                    encoder_inputs,
                                                    decoder_inputs,
                                                    tf.contrib.rnn.BasicLSTMCell(size),
                                                    num_encoder_symbols = inputVocabLen,
                                                    num_decoder_symbols = targetVocabLen,
                                                    embedding_size = 100,
                                                    feed_previous = True,
                                                    output_projection = output_projection,
                                                    dtype = tf.float32)
        # ops for projecting outputs
        outputs_proj = [tf.matmul(outputs[i], 
                        output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

        ## Clean and Format incoming msg by humans.
        ## It is better to do the same clean/format as the data preprocessing steps
        ## for the algorithm to predict next words more accurately
        msgLowerCase = [w.lower() for w in nltk.word_tokenize(humanMsg)]
        msg = [w for w in msgLowerCase if permissible_chars(w)]
        if len(msg) > input_seq_len:
            msg = msg[0:input_seq_len-1]

        human_msg_encoded = [input_w2i.get(word, 0) for word in msg]

        # Fill in with padding marker
        for k in range(input_seq_len - len(human_msg_encoded)):
            human_msg_encoded = human_msg_encoded + [input_w2i[marker_pad]]

        # restore all variables - use the last checkpoint saved
        saver = tf.train.Saver()
        path = tf.train.latest_checkpoint('checkpoints')

        with tf.Session() as sess:
            # restore
            saver.restore(sess, path)

            # feed data into placeholders
            feed = {}
            for i in range(input_seq_len):
                feed[encoder_inputs[i].name] = np.array([human_msg_encoded[i]], dtype = np.int32)

            feed[decoder_inputs[0].name] = np.array([target_w2i[marker_start]], dtype = np.int32)

            # translate
            output_sequences = sess.run(outputs_proj, feed_dict = feed)

            ouput_seq = [output_sequences[j][0] for j in range(output_seq_len)]
            #decode output sequence
            words = decode_output(ouput_seq)

            for i in range(len(words)):
                if words[i] not in [marker_end, marker_pad, marker_start]:
                    replyMsg += words[i] + ' '
                             
        print(replyMsg)
        return replyMsg
## Chatbot Interface for Human-Bot interaction
import tkinter 

def Enter_pressed(event):
    input_get = input_field.get()
    print(input_get)
    bot_reply = generateReply(input_get)
    if (len(input_get.strip()) > 0):
        messages.insert(INSERT, '\nYou says: \t%s' % input_get)
    if (len(bot_reply.strip()) > 0):
        messages.insert(INSERT, '\nBot says: \t%s' % bot_reply)
    input_user.set('')
    messages.see(tkinter.END)
    return "break"

from ttkthemes import ThemedTk
window = ThemedTk()
window.set_theme("blue")

# window = Tk()
window.geometry('300x450')
window.title("Digital Imprint of You!")

messages = Text(window)
messages.insert(INSERT, '')
messages.pack()

input_user = StringVar()
input_field = ttk.Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)

# frame = Frame(window)
input_field.bind("<Return>", Enter_pressed)
input_field.pack()


btn = Button(window,text='Send', command=Enter_pressed(''))
btn.bind('<Button-1>', Enter_pressed)
btn.pack(side=RIGHT, fill=X)


window.mainloop()

