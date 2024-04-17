import nltk
from nltk.tokenize import sent_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
from transformers import pipeline
import asyncio
import pymongo
from fastapi import HTTPException
from fastapi import Query
from bson import ObjectId
from fastapi.responses import JSONResponse

# MongoDB connection details
MONGODB_URL = "mongodb+srv://prkskrs:1JRRLP0TScJtklaB@cluster0.fncdhdb.mongodb.net/myPrjmtDB?retryWrites=true&w=majority"
DB_NAME = "siddagangaDB"
USER_COLLECTION = "users"

# Connect to MongoDB
client = pymongo.MongoClient(MONGODB_URL)
db = client[DB_NAME]
user_collection = db[USER_COLLECTION]


nltk.download('punkt')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

with open("academics.json", "r") as file:
    data = json.load(file) 

def encode_and_compute_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

def merge_and_rephrase(similar_answers):
    merged_answer = " ".join(similar_answers)
    sentences = sent_tokenize(merged_answer)
    rephrased_answer = ". ".join([sentence.capitalize() for sentence in sentences])
    return rephrased_answer

def get_similar_question_and_answer(question):
    similarities = [(q, encode_and_compute_similarity(question, q)) for q in data.keys()]
    max_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:1]
    similar_questions = [sim[0] for sim in max_similarities]
    similar_answers = [data[q] for q in similar_questions]
    
    return similar_questions, similar_answers


class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Question(BaseModel):
    question: str

class Feedback(BaseModel):
    question: str
    answer: str
    company: str
    email: str
    difficulty: str

@app.post("/get_answer/")
async def get_answer(question: Question):
    similar_questions, similar_answers = get_similar_question_and_answer(question.question)
    merged_and_rephrased_answer = merge_and_rephrase(similar_answers)
    loop = asyncio.get_event_loop()
    summary_task = loop.run_in_executor(None, summarizer, merged_and_rephrased_answer, 130, 30, False)
    summary = await summary_task
    summarized_text = summary[0]['summary_text']
    return {"similar_questions": similar_questions, "merged_and_rephrased_answer": merged_and_rephrased_answer, "answer": summarized_text}

@app.post("/signup/")
async def signup(user: UserSignup):
    print(user)
    if user_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="User already exists with this email")

    user_collection.insert_one({"email": user.email, "password": user.password})
    return {"message": "User created successfully"}

@app.post("/login/")
async def login(user: UserLogin):
    stored_user = user_collection.find_one({"email": user.email})
    if user.password == stored_user["password"]:
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=404, detail="Invalid Username or Password")

@app.post("/store_feedback/")
async def store_feedback(feedback: Feedback):
    existing_feedback = db.feedback.find_one({"email": feedback.email, "company": feedback.company, "difficulty": feedback.difficulty})
    if existing_feedback:
        raise HTTPException(status_code=400, detail="Feedback already exists for this email and company")
    existing_user = user_collection.find_one({"email": feedback.email})
    
    if not existing_user:
        raise HTTPException(status_code=400, detail="User with this email does not exist")
    
    db.feedback.insert_one(feedback.dict())
    return {"message": "Feedback stored successfully"}

@app.get("/get_feedback/")
async def get_feedback():
    feedback_cursor = db.feedback.find({})
    feedback_list = []
    for feedback in feedback_cursor:
        feedback_list.append({
            "username": feedback["email"].split("@")[0],
            "answer": feedback["answer"],
            "company": feedback["company"],
            "difficulty": feedback["difficulty"]
        })

    if not feedback_list:
        return JSONResponse(content={"message": "No feedback available for this company"}, status_code=404)

    return feedback_list



# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
