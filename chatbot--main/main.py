from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import time

# Intent Classification (Zero-shot classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Intent descriptions (natural language)
intent_labels = [
    "User wants to track an order",
    "User is asking about a product",
    "User needs technical support",
    "User is greeting",
    "User is saying goodbye",
    "User message doesn't match any known intent"
]

def classify_intent(user_message):
    result = classifier(user_message, intent_labels)
    top_intent = result["labels"][0]
    confidence = result["scores"][0]
    return top_intent, confidence

# Order Tracking Module
dummy_orders = {
    "123456": {"status": "Out for delivery", "expected_date": "April 12th"},
    "789012": {"status": "Shipped", "expected_date": "April 14th"},
    "345678": {"status": "Processing", "expected_date": "April 16th"},
    "901234": {"status": "Delivered", "expected_date": "April 10th"},
}

def fetch_order_status(order_number):
    print("Fetching order details...")
    time.sleep(2)  # Simulate network delay
    order_info = dummy_orders.get(order_number)
    if order_info:
        return f"Your order #{order_number} is {order_info['status'].lower()} and should arrive by {order_info['expected_date']}."
    else:
        return f"Order #{order_number} not found. Please check the number and try again."

# Product Questions Module
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample product FAQ or catalog
product_data = {
    "Model X battery": "Model X has a battery life of up to 12 hours.",
    "Model Y weight": "Model Y weighs just 1.2 kg, making it ultra-portable.",
    "Model Z screen": "Model Z features a 14-inch 4K display with touch support.",
    "Model X warranty": "Model X comes with a 2-year limited warranty."
}

# Pre-encode product entries
keys = list(product_data.keys())
values = list(product_data.values())
catalog_embeddings = model.encode(keys, convert_to_tensor=True)

def get_product_answer(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, catalog_embeddings)[0]
    best_match_idx = scores.argmax()
    return values[best_match_idx]

# Technical Support Module
class TechSupportChatbot:
    def __init__(self):
        self.escalated = False
        self.product = None
        self.issue = None

    def greet_user(self):
        print("Hello! I'm your technical support assistant. How can I help you today?")

    def ask_product(self):
        self.product = input("Which device is having issues? ")

    def ask_issue(self):
        self.issue = input(f"Can you describe the issue you're facing with your {self.product}? ")

    def provide_fix(self):
        if "restart" in self.issue.lower():
            print("It seems like you may need to restart your device. Please try the following steps:")
            print("1. Turn off your device and wait for 10 seconds.")
            print("2. Press and hold the power button for 5 seconds to restart.")
            print("3. Check if the issue persists.")
        else:
            print("I suggest checking for any recent software updates or reinstalling the app. If that doesn't resolve the issue, we may need to look into further options.")

    def ask_for_details(self):
        additional_info = input("When did the issue first occur? ")
        print(f"Thanks for providing that information: {additional_info}")

    def escalate_issue(self):
        self.escalated = True
        print("I'm sorry, but it seems like this issue requires further assistance. I will escalate your request to a technical specialist.")
        print("Please hold on for a moment.")

    def confirm_escalation(self):
        if self.escalated:
            contact_info = input("Can you provide your contact details to ensure a quick resolution? ")
            print(f"Your issue has been escalated to our support team. A technician will contact you at {contact_info}.")

    def close_conversation(self):
        print("Thank you for reaching out! Please let us know if you need further assistance.")

# Main Function to Integrate All Components
def main():
    print("Welcome to the Customer Support Chatbot!")
    while True:
        user_message = input("You: ")

        # Classify the intent of the message
        intent, confidence = classify_intent(user_message)
        print(f"Intent: {intent}, Confidence: {confidence:.2f}")

        if intent == "User wants to track an order":
            order_number = input("Please enter your order number: ")
            print(fetch_order_status(order_number))

        elif intent == "User is asking about a product":
            print(get_product_answer(user_message))

        elif intent == "User needs technical support":
            chatbot = TechSupportChatbot()
            chatbot.greet_user()
            chatbot.ask_product()
            chatbot.ask_issue()
            chatbot.provide_fix()

            ask_more_info = input("Would you like to provide more details? (yes/no): ")
            if ask_more_info.lower() == "yes":
                chatbot.ask_for_details()

            unresolved = input("Is the issue resolved? (yes/no): ")
            if unresolved.lower() == "no":
                chatbot.escalate_issue()
                chatbot.confirm_escalation()

            chatbot.close_conversation()

        elif intent == "User is greeting":
            print("Bot: Hello! How can I assist you today?")

        elif intent == "User is saying goodbye":
            print("Bot: Goodbye! Have a great day!")
            break

        else:
            print("Bot: I'm sorry, I couldn't understand that. Can you please rephrase?")

# Run the chatbot
if __name__ == "__main__":
    main()
