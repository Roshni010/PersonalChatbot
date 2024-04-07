from langchain.vectorstores.milvus import Milvus 
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
load_dotenv()
host = os.getenv('HOST')
port = os.getenv('PORT')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
database = Milvus(embeddings,connection_args={'host':host,'port':port},collection_name='cv_roshni')

# print(database.similarity_search("what is my name?"),k=2)


