from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langsmith import Client
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# LangSmith istemcisi
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
if LANGCHAIN_API_KEY is None:
    raise ValueError("LANGCHAIN_API_KEY is not set.")

client = Client(api_key=LANGCHAIN_API_KEY)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})

class QueryRequest(BaseModel):
    query_text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the API!"}

@app.post("/query")
async def query(request: QueryRequest):
    query_text = request.query_text
    
    # Veritabanını hazırla
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Veritabanında ara
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return {"response": response_text, "sources": sources}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
