🛠 Customer Support Chatbot

This project is a Customer Support Chatbot built using Hugging Face Transformers and Sentence-Transformers.
It can handle intent classification, order tracking, product inquiries, and technical support in an interactive way.


---

📌 Features

1. Intent Classification (Zero-shot Classification)

Uses facebook/bart-large-mnli model to understand user messages.

Supports the following intents:

Track an order

Ask about a product

Get technical support

Greeting

Goodbye

Unknown intent




2. Order Tracking Module

Simulates fetching order details from a dummy database.

Provides order status and expected delivery date.



3. Product Questions Module

Uses all-MiniLM-L6-v2 embeddings with cosine similarity to match user queries with product FAQ.

Returns the most relevant product information.



4. Technical Support Module

Guides the user through troubleshooting steps.

Asks about the product, issue description, and additional details.

Provides basic fixes or escalates the issue to a support team.



5. Interactive Chat Flow

Greets the user.

Classifies intent dynamically.

Routes conversation to the right module.

Escalates unresolved issues.





---

🚀 Installation

Make sure you have Python 3.8+ installed.

# Clone the repository
git clone https://github.com/your-username/customer-support-chatbot.git
cd customer-support-chatbot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


---

📦 Dependencies

transformers

sentence-transformers

torch

time (built-in Python)


You can install them via:

pip install transformers sentence-transformers torch


---

▶️ Usage

Run the chatbot:

python chatbot.py

Sample flow:

Welcome to the Customer Support Chatbot!
You: Hi
Intent: User is greeting, Confidence: 0.95
Bot: Hello! How can I assist you today?

You: I want to track my order
Please enter your order number: 123456
Fetching order details...
Your order #123456 is out for delivery and should arrive by April 12th.


---

🧠 How It Works

1. Intent Detection → Classifies user message into one of the predefined intents.


2. Routing → Based on the intent, forwards the query to the correct module:

Order Tracking → fetches from dummy_orders.

Product Inquiry → uses semantic search over product_data.

Tech Support → guides the user through troubleshooting.

Greeting / Goodbye → simple responses.



3. Conversation Flow → Keeps interacting until user says goodbye.




---

📚 Future Improvements

Add real database/API for order tracking.

Expand product catalog dynamically.

Improve tech support with knowledge base & FAQs.

Add sentiment analysis for better responses.

Deploy chatbot as a web app or integrate into WhatsApp/Slack.



---

📝 License

This project is licensed under the MIT License.
