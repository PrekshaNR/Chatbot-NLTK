import joblib
import numpy as np

import joblib

model_path = r"C:\Users\User\anaconda3\envs\Chatbot\restaurant_intent_classifier.joblib"
model = joblib.load(model_path)

print("Model loaded from:", model_path)
print("Has idf_ after loading?:", hasattr(model.named_steps['tfidf'], 'idf_'))

# Intent prediction with confidence threshold
def predict_intent(user_text: str, threshold=0.4):
    # distance from decision boundary for each class
    scores = model.decision_function([user_text])
    if scores.ndim > 1:
        scores = scores[0]

    max_score = float(scores.max())
    intent = model.predict([user_text])[0]

    # if the best score is small, model is not very confident
    if max_score < threshold:
        return "fallback"
    return intent

# Predefined responses for each intent
responses = {
    "greeting": "Hello 👋! How can I assist you today?",
    "opening_hours": "We are open from 10 AM to 11 PM every day!",
    "location": "We are located on MG Road, Bangalore 📍",
    "menu_inquiry": "We have a variety of dishes including pizzas, pastas and desserts 🍽️",
    "price_inquiry": "Our meals start from ₹199 onwards 💰",
    "table_reservation": "Sure! For how many people and at what time? 😊",
    "wait_time": "Right now the wait time is around 10–15 minutes ⏳",
    "special_offers": "We have 20% off on combo meals today 🎉",
    "dietary_options": "We have vegan, Jain and gluten-free options 🥗",
    "chef_recommendations": "The chef recommends our Peri Peri Pasta 👨‍🍳",
    "contact_information": "You can reach us at +91-9876543210 ☎",
    "ambience": "We have cozy indoor and rooftop seating 🌃",
    "goodbye": "Goodbye! Have a great day 😄",
    "fallback": "I am not fully sure I understood that. Could you please rephrase? 🙂"
}

def chat():
    print("🤖 Restaurant Bot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye 👋!")
            break

        intent = predict_intent(user_input)
        response = responses.get(intent, responses["fallback"])
        
        print(f"Bot: {response}")

chat()
