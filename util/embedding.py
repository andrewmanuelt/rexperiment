import torch

from langchain_huggingface import HuggingFaceEmbeddings

class Embedding():
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.config = {
            'model_name': 'sentence-transformers/all-mpnet-base-v2', 
            'model_kwargs': {'device': self.device},
            'encode_kwargs': {'normalize_embeddings': True},
        }
        
    def load_embedding_function(self):
        embedding = HuggingFaceEmbeddings(
            model_name = self.config['model_name'], 
            model_kwargs = self.config['model_kwargs'],
            encode_kwargs = self.config['encode_kwargs']
        )

        return embedding