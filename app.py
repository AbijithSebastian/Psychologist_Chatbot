import gradio as gr
from groq import Groq
import os

# Load API key from environment variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def initialize_messages():
    return [{"role": "system",
             "content": """You are a compassionate and experienced psychologist with expertise in mental health, 
             emotional well-being, and behavioral therapy. Your role is to assist individuals by providing supportive, 
             evidence-based psychological advice. You offer guidance on coping strategies, mental health issues, 
             and emotional challenges in a professional, empathetic, and non-judgmental manner. Always prioritize the user’s safety, 
             confidentiality, and well-being. Avoid diagnosing or prescribing medication; instead, encourage professional consultation 
             when necessary.""" }]

messages_prmt = initialize_messages()

def customLLMBot(user_input, history):
    global messages_prmt

    messages_prmt.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        messages=messages_prmt,
        model="llama3-8b-8192",
    )
    LLM_reply = response.choices[0].message.content
    messages_prmt.append({"role": "assistant", "content": LLM_reply})

    return LLM_reply

iface = gr.ChatInterface(
    customLLMBot,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question related to Psychology"),
    title="Psychologist ChatBot",
    description="""A supportive chatbot designed to help you with emotional well-being, stress, anxiety, 
                   and everyday mental health concerns. This is not a substitute for professional therapy 
                   or crisis support.""",
    theme="soft",
    examples=["I'm feeling anxious lately", "How can I manage exam stress?", "I feel overwhelmed, what should I do?"],
    submit_btn=None
)

iface.launch()