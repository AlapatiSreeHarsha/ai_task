from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
import os

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Initialize URL and vector store
URL = "https://brainlox.com/courses/category/technical"
vector_store = None

def initialize_vector_store():
    global vector_store
    
    # Load data from URL using WebBaseLoader
    loader = WebBaseLoader(URL)
    data = loader.load()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = text_splitter.split_documents(data)
    
    # Create embeddings and vector store using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

# Initialize conversation chain
def get_conversation_chain():
    # Using a free model from HuggingFace
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return conversation_chain

class ChatbotAPI(Resource):
    def post(self):
        try:
            data = request.get_json()
            question = data.get('question')
            chat_history = data.get('chat_history', [])
            
            if not question:
                return {'error': 'Question is required'}, 400
            
            if vector_store is None:
                initialize_vector_store()
            
            conversation_chain = get_conversation_chain()
            response = conversation_chain({
                'question': question,
                'chat_history': chat_history
            })
            
            return {
                'answer': response['answer'],
                'chat_history': chat_history + [(question, response['answer'])]
            }
            
        except Exception as e:
            return {'error': str(e)}, 500

# Add resource to API
api.add_resource(ChatbotAPI, '/chat')

if __name__ == '__main__':
    # Initialize vector store on startup
    initialize_vector_store()
    app.run(debug=True)
