import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the trained model
model_path = "feedback_model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Prediction function
def predict_alignment(text, reason):
    inputs = tokenizer(text, reason, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "✅ Feedback matches reason" if prediction == 1 else "❌ Feedback does NOT match reason"

# Gradio Interface
interface = gr.Interface(
    fn=predict_alignment,
    inputs=[
        gr.Textbox(label="User Feedback Text"),
        gr.Textbox(label="Dropdown Reason"),
    ],
    outputs=gr.Label(label="Alignment Result"),
    title="MyZoom Feedback Validator",
    description="Enter a user's feedback and dropdown reason to check if they align (DistilBERT-based classifier)."
)

# Launch app
interface.launch(share=True)
