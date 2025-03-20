from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import openai
import nltk
import spacy
from nltk.chat.util import Chat, reflections
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import random
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Set OpenAI API key
openai.api_key = 'your_openai_api_key_here'

# SQLAlchemy setup for query logging
DATABASE_URL = "sqlite:///./legal_queries.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define a query log model
class QueryLog(Base):
    __tablename__ = 'queries'
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, index=True)
    response = Column(String)

Base.metadata.create_all(bind=engine)

# Simple chatbot using NLP (to simulate legal advice)
legal_patterns = [
    (r"what is (.*) law", ["The law related to %s is...", "In legal terms, %s refers to..."]),
    (r"how to (.*) case", ["To proceed with a %s case, you need to..."]),
    (r"what is (.*) contract", ["A %s contract refers to...", "In contract law, %s is a formal agreement..."]),
    # Add more patterns as needed
]

legal_chatbot = Chat(legal_patterns, reflections)

# Function to generate legal support from OpenAI
def get_legal_support(query: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Provide a brief explanation for: {query}",
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Function to log queries and responses to database
def log_query(session, user_query: str, response_text: str):
    query_log = QueryLog(query=user_query, response=response_text)
    session.add(query_log)
    session.commit()

# Define a request model using Pydantic
class QueryRequest(BaseModel):
    query: str

class ResponseModel(BaseModel):
    response: str
    suggested_actions: List[str] = []
    related_links: List[str] = []

# Function for classifying legal questions and providing additional suggestions
def classify_query(query: str):
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]
    actions = []
    links = []

    if "contract" in query.lower():
        actions.append("Review the contract terms thoroughly.")
        links.append("https://www.law.com/contract-terms/")
    
    if "law" in query.lower():
        actions.append("Consult with a lawyer to discuss your legal situation.")
        links.append("https://www.legaldictionary.com/")
    
    return actions, links

# FastAPI route to handle legal queries
@app.post("/ask", response_model=ResponseModel)
async def ask_question(request: QueryRequest):
    user_query = request.query

    # Check if query matches predefined legal patterns
    response_text = legal_chatbot.respond(user_query)
    
    # If no response from chatbot, use OpenAI for AI-based legal support
    if not response_text:
        response_text = get_legal_support(user_query)

    # Classify the query for additional info
    suggested_actions, related_links = classify_query(user_query)

    # Log the query and response to the database
    session = SessionLocal()
    log_query(session, user_query, response_text)

    return {"response": response_text, "suggested_actions": suggested_actions, "related_links": related_links}

@app.get("/queries")
def get_queries(skip: int = 0, limit: int = 10):
    session = SessionLocal()
    queries = session.query(QueryLog).offset(skip).limit(limit).all()
    return queries

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
