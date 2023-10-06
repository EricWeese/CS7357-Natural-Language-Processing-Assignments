import nltk
from datasets import load_dataset

# Download the necessary NLTK data
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# Load the dataset
dataset = load_dataset("sidhq/email-thread-summary")

print(f"{dataset['train']['thread'][0]}")
# Define a function to extract action items from text
def extract_action_items(text):
    action_items = []

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Define action verbs that are commonly used to start action items
    action_verbs = ["please", "kindly", "ensure", "verify", "complete", "confirm", "prepare", "review", "create", "update"]

    # Iterate over sentences
    for sentence in sentences:
        # Tokenize the sentence into words and tag their parts of speech
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)

        # Check if the sentence starts with an action verb
        if tagged_words[0][0].lower() in action_verbs:
            action_items.append(sentence)

    return action_items

# Iterate over the dataset and extract action items from each email
# for idx, example in enumerate(dataset['train']):
#     email_body = example['thread']['messages'][0]['body']
#     action_items = extract_action_items(email_body)
#     if action_items:
#         print(f"Email {idx + 1} Action Items:")
#         for item in action_items:
#             print(f"- {item}")
#         print()
#     if idx > 100:
#         break

# Note: Adjust the loop and indexing based on the structure of the dataset
