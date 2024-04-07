from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.milvus import Milvus
from dotenv import load_dotenv
import os
# To connect the translatorchain.py with main.py and perform prompt chaining where you first translate the query and then perform similarity search, you can follow these steps:

# Import the translatedQuestion function from translatorchain.py into main.py.
# Modify the result function in main.py to first call the translatedQuestion function to translate the query.
# Use the translated query as input for the similarity search.
from translatorchain import translatedQuestion  # Importing the translatedQuestion function

# Load environment variables
load_dotenv()

# Define the connection parameters
host = os.getenv('HOST')
port = os.getenv('PORT')

# Initialize the generative AI model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
llmG = ChatGoogleGenerativeAI(model='gemini-1.0-pro', generation_config=generation_config)

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the output parser
output_parser = StrOutputParser()

# Load the document data
loader = CSVLoader('./docs/Roshani Information - Sheet1.csv', encoding="utf-8")
data = loader.load()

# Initialize the Milvus vector store
database = Milvus.from_documents(data, embeddings, connection_args={'host': host, 'port': port}, collection_name='cv_roshni')

# Define the prompt template
template = """
### Instruction ###
You are Roshani Thapa, so remember you are a Human.
Below is the query by the user delimited by triple backticks.
JUST PROVIDE THE GENERATIVE RESPONSE TO THE QUERY POSITIVELY.
DO NOT PROVIDE THE INFORMATION OUT OF THE CONTEXT.
DONOT EXCEED MORE THAN 30 WORDS.
query:```{question}```
context:```{context}```
Provide reply only without backticks and nothing else.
If you don't find any relevant information in the context just give phone number from the document and say "Call me".
If the question contains statements like wow , oh my god,oh okay then response to the query with "Yes. Thankyou".
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the prompt chaining pipeline
chain = RunnableMap({
    "context": lambda x: database.similarity_search(x['question'], k=8),
    "question": lambda x: x['question']
}) | prompt | llmG | output_parser

# Define the result function
def result(query: str):
    # Translate the query
    translated_query = translatedQuestion(query)["result"]

    # Perform the prompt chaining with the translated query
    # context = database.similarity_search(translated_query, k=8)
    # print("Context:", context)
    result = chain.invoke({'question': translated_query})
    return {"text": result}
def ask_for_info(ask_for=['name', 'age', 'location']):
    # If there are no more items to ask for, thank the user and ask how you can help them
    if not ask_for:
        return "Thank you for providing the information. How can I assist you further?"

    # Prompt the user for the next item in the ask_for list
    question = f"May I ask, what is your {ask_for[0]}?"
    response = input(question)

    # Remove the item from the ask_for list
    ask_for = ask_for[1:]

    # Recursively call the function to ask for the remaining items
    return ask_for_info(ask_for)
# Main block to take user input and show response
if __name__ == "__main__":
    query = input("Enter your query: ")

    # Check if the user query is "i want to give you my details"
    if query.lower() == "i want to give you my details":
        # Execute ask_for_info function
        response = ask_for_info()
    else:
        
        # Execute result function
        response = result(query)

    print("Response:", response)

