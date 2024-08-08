import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("fine-tuned-bert-lora")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-bert-lora")

# Define a function to predict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id

# Streamlit UI
st.title("Text Classification with BERT")
user_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    prediction = predict(user_input)
    st.write(f"Predicted class: {prediction}")
