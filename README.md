# AI Chatbot with LangChain and Flask

## Project Overview
This project implements a custom chatbot that can answer questions about technical courses from brainlox.com. It uses LangChain for document processing and conversation management, Flask-RESTful for the API, and HuggingFace models for embeddings and text generation.

## Features
- Web scraping of course data using LangChain's WebBaseLoader
- Text processing and chunking for better context understanding
- Vector storage using FAISS for efficient similarity search
- Conversational AI using HuggingFace's flan-t5-small model
- RESTful API endpoint for chat interactions
- Chat history management

## Technical Stack
- **Framework**: Flask & Flask-RESTful
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Language Model**: google/flan-t5-small
- **Document Loading**: LangChain WebBaseLoader
- **Text Processing**: LangChain CharacterTextSplitter

## Installation

1. Clone the repository:
git clone https://github.com/AlapatiSreeHarsha/ai_task.git
2.Files running
open Terminal and run 
python app.py
