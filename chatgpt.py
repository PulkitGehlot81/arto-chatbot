import openai

# Set up your OpenAI API credentials
openai.api_key = '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'

def generate_chatbot_response(user_message):
    # Check if the user's message is related to computer science
    is_computer_science_query = is_computer_science_related(user_message)

    if is_computer_science_query:
        # Generate a chatbot response based on the user's message and computer science context
        context = "You are a student interested in computer science. You ask: " + user_message
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=context,
            max_tokens=1000,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
    else:
        return "I am programmed to respond to queries in the field of computer science only."

def is_computer_science_related(message):
    # Perform a check to determine if the message is related to computer science
    # You can customize this check based on specific keywords, patterns, or machine learning models
    computer_science_keywords = ["computer science", "programming", "algorithms", "data structures"]
    if any(keyword in message.lower() for keyword in computer_science_keywords):
        return True
    else:
        return False

# Example usage
user_message = "give me some best course link for learning Python"
chatbot_response = generate_chatbot_response(user_message)
print(chatbot_response)
