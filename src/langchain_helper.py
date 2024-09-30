from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import shutil
from dotenv import load_dotenv
load_dotenv()

llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"], temperature=0.1)
instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path=r"D:\generativeai\customer_service_chatbot_LLM\dataset\dataset.csv", source_column="prompt")
    try:
        data = loader.load()
        if not data:
            print("No data loaded from the CSV file.")
            return

        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
        
        if os.path.exists(vectordb_file_path):
            shutil.rmtree(vectordb_file_path)  # Remove existing index
            print(f"Removed existing directory: {vectordb_file_path}")

        vectordb.save_local(vectordb_file_path)
        print(f"Vector database saved to {vectordb_file_path}.")
        
    except Exception as e:
        print(f"Error during vector database creation: {e}")

    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    prompt_template="""Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever= retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain

if __name__=="__main__" :
    create_vector_db()
    chain = get_qa_chain()
    if chain:
        print(chain("hello?"))
