# Sentiment analysis functions
def clean_text(text):
    """
    This function cleans the text data by removing unnecessary characters
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().strip()
    return text

def get_sentiment(text):
    """
    This function returns the sentiment score (positive, negative or neutral) of a given text
    """
    text = clean_text(text)
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Topic modelling functions
def train_topic_model(documents):
    """
    This function trains an LDA topic model on the given list of documents
    """
    processed_docs = [nlp(doc).noun_chunks for doc in documents]
    dictionary = corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = LdaMulticore(bow_corpus, num_topics=5, id2word=dictionary, passes=2, workers=2)
    return lda_model, dictionary

def get_topics(text, lda_model, dictionary):
    """
    This function returns the topics of a given text using the trained LDA model and dictionary
    """
    processed_doc = [nlp(text).noun_chunks]
    bow_vector = dictionary.doc2bow(processed_doc[0])
    topics = lda_model.get_document_topics(bow_vector)
    return topics

# Updated get_response function
def get_response(msg, context={}):
    """
    This function returns the response of the chatbot to a given message
    """
    while True:
        inp = msg.lower()
        if inp == "quit" or inp == None:
            break
        
        # Check if message is part of a multi-turn conversation
        if 'context' in context and 'last_intent' in context:
            if context['last_intent'] in ['greet', 'goodbye', 'thanks']:
                context = {}
            else:
                context['counter'] += 1
        
        # Apply word correction function to message
        inp_x = word_checker(inp)
        
        # Get bag of words representation of message
        bag = bag_of_words(inp_x, words)
        
        # Predict intent of message using trained model
        results = model.predict([bag])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        # Check if intent has a confidence score of 0.9 or higher
        if results[results_index] >= 0.9:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    ms = random.choice(responses)
                    context['last_intent'] = tg['tag']
                    if tg['tag'] == 'ask_sentiment':
                        sentiment = get_sentiment(msg)
                        ms = ms.replace('{sentiment}', sentiment)
                    elif tg['tag'] == 'ask_topic':
                        lda_model, dictionary = train_topic_model([doc['text'] for doc in data['docs']])
                        topics = get_topics(msg, lda_model, dictionary)
                        ms = ms.replace('{topic}', str(topics[0][0]))
                    return ms, context
        else:
            if 'context' in context and 'counter' in context and context['counter'] >= 2:
                responses = ["I'm sorry, I don't understand what you're asking. Let's start over."]
