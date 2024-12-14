import faiss 

from tqdm import tqdm
from uuid import uuid4

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class VectorDatabase():
    def __init__(self, folder_path, index_name, embedding_function) -> None:
        self.folder_path = folder_path
        self.index_name = index_name
        self.ef = embedding_function
    
    def store_client(self):
        index = faiss.IndexFlatL2(len(self.ef.embed_query('hello')))
        client = FAISS(
            docstore = InMemoryDocstore(),
            embedding_function = self.ef, 
            index = index,  
            index_to_docstore_id = {}
        )

        return client

    def store(self, client, documents: list):
        print('Waiting for storing documents...')

        ids = [str(uuid4()) for x in range(0, len(documents))]

        client.add_documents(
            documents=documents,
            ids=ids
        ) 

        client.save_local(
            folder_path = self.folder_path,
            index_name = self.index_name
        )   
    
    def load(self):
        client = FAISS.load_local(
            index_name = self.index_name,
            folder_path = self.folder_path,
            embeddings = self.ef, 
            allow_dangerous_deserialization = True,
        )       

        return client 