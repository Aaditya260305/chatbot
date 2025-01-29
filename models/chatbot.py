# models/chatbot.py
from llama_index.core import VectorStoreIndex, StorageContext , SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings, PromptTemplate
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import os
from datetime import datetime
import pickle

class CryptoChatbot:
    def __init__(self, api_key, model_path="models/saved"):
        self.api_key = api_key
        self.model_path = model_path
        self.setup_model()
        
    def setup_model(self):
        self.client = qdrant_client.QdrantClient(path=f"qdrant_gemini_{datetime.now().timestamp()}")
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name="collection"
        )

        # Configure settings
        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=self.api_key
        )

        # Settings.llm = Gemini(api_key=self.api_key)
        Settings.llm = Gemini(
            api_key=self.api_key,
            temperature=0.3,
            request_timeout=60.0
        )
        
        # Set up storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        documents = SimpleDirectoryReader("models\data").load_data()
        # print(documents)

        self.index = VectorStoreIndex(
            documents,
            storage_context=self.storage_context,
        )
        
        # Define QA template
        self.qa_template = PromptTemplate(
            template=(
                "Your are Crypta-AI specialised in cryptocurrencies and related stuff.\n"
                "Don't tell about the dataset you are trained on.\n"
                "I am not a financial advisor. Always do your own research before making investment decisions.\n"
                "Cryptocurrency markets are highly volatile and risky. Only invest what you can afford to lose.\n"
                "The information provided is for educational purposes only.\n"
                "We have provided context information below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given this information, please answer the question: {query_str}\n"
            )
        )

        self.save_model()

    def load_model(self):
        """Load saved model state"""
        try:
            with open(f"{self.model_path}/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

    def save_model(self):
        """Save current model state"""
        os.makedirs(self.model_path, exist_ok=True)
        with open(f"{self.model_path}/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
    
    def query(self, question):
        """Process a query and return response"""
        if not hasattr(self, 'index'):
            if not self.load_model():
                raise Exception("Model not trained yet!")
        
        # print(question)
        query_engine = self.index.as_query_engine(
            text_qa_template=self.qa_template
        )
        response = query_engine.query(question)
        print(response)

        return str(response)