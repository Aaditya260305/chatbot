# utils/model_manager.py

from models.chatbot import CryptoChatbot

class ModelManager:
    def __init__(self):
        self.chatbot = None
        
    def initialize_chatbot(self, api_key):
        self.chatbot = CryptoChatbot(api_key)
        return self.chatbot
    
    def get_chatbot(self):
        if not self.chatbot:
            raise Exception("Chatbot not initialized!")
        return self.chatbot
