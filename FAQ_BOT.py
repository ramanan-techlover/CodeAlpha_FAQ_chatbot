import tkinter as tk
from tkinter import scrolledtext
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize SpaCy and download required NLTK packages
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

# Define a list of FAQ pairs (questions and answers)
faq_data = {
    "What is the return policy?": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "How can I track my order?": "You can track your order through the 'Orders' section in your account.",
    "What are the payment options?": "We accept credit cards, debit cards, and PayPal.",
    "How do I contact customer support?": "You can contact customer support by emailing support@ourcompany.com or calling +1-234-567-890.",
    "Are there any discounts available?": "Yes, we have seasonal discounts and offer discounts for members."
}

# Pre-process the FAQ data into lists for easier TF-IDF usage
faq_questions = list(faq_data.keys())
faq_answers = list(faq_data.values())

# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the FAQ questions
vectorizer.fit(faq_questions)

def chatbot_response(user_query):
    # Process the user query
    query_vector = vectorizer.transform([user_query])
    faq_vectors = vectorizer.transform(faq_questions)

    # Compute similarity
    similarities = cosine_similarity(query_vector, faq_vectors).flatten()
    closest_match_idx = np.argmax(similarities)

    # Define a similarity threshold
    threshold = 0.3
    if similarities[closest_match_idx] >= threshold:
        return faq_answers[closest_match_idx]
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase your question?"

# Set up the main window for the chatbot
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("600x600")

# Create a Scrollable Text Box for Chat History
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, state='disabled')
chat_history.grid(row=0, column=0, padx=10, pady=10)

# Function to handle the sending of messages
def send_message():
    user_query = user_input.get()
    if user_query:
        chat_history.config(state='normal')
        chat_history.insert(tk.END, "You: " + user_query + "\n")
        response = chatbot_response(user_query)
        chat_history.insert(tk.END, "Bot: " + response + "\n\n")
        chat_history.config(state='disabled')
        user_input.delete(0, tk.END)  # Clear the input field

# Function to set placeholder text when input box is empty
def on_click(event):
    if user_input.get() == "Enter your question here...":
        user_input.delete(0, tk.END)  # Clear the placeholder text
        user_input.config(fg="black")  # Change text color to black

# Function to reset placeholder text if the user leaves the input field empty
def on_focusout(event):
    if user_input.get() == "":
        user_input.insert(0, "Enter your question here...")
        user_input.config(fg="grey")  # Change text color to grey

# Create an Entry Box for User Input with placeholder
user_input = tk.Entry(root, width=70, fg="grey")
user_input.insert(0, "Enter your question here...")
user_input.grid(row=1, column=0, padx=10, pady=10)

# Bind the functions to the input box for handling placeholder text
user_input.bind("<FocusIn>", on_click)
user_input.bind("<FocusOut>", on_focusout)

# Create a Send Button
send_button = tk.Button(root, text="Send", width=20, height=2, command=send_message, bg="lightblue")
send_button.grid(row=2, column=0, pady=10)

# Run the Tkinter event loop
root.mainloop()
