from fastapi import FastAPI, File, HTTPException, Depends, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
from pymongo import MongoClient
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Connect to MongoDB
client = MongoClient(
    "mongodb+srv://atharvdesai:ahrAA7kOTdZfyur9@cluster0.smf3kdb.mongodb.net/Memeingle?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["Memeingle"]
collection = db["memes"]

# Secret key for JWT token
SECRET_KEY = "Tatakae"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Token Data model
class TokenData(BaseModel):
    username: str


# Token model
class Token(BaseModel):
    access_token: str
    token_type: str


# Function to create access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Function to verify token
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username)
        return token_data
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Endpoint to retrieve a list of memes
@app.get("/memelist/", response_model=List[dict])
def get_memelist():
    memes = collection.find({}, {"_id": 1, "Url": 1})
    meme_list = [{"id": str(meme["_id"]), "Url": meme["Url"]} for meme in memes]
    return meme_list

    # Read CSV file into pandas DataFrame


df = pd.read_csv("./downloads.csv")

# Convert DataFrame to list of dictionaries (each dictionary represents a record)
memes_data = df.to_dict(orient="records")

# Insert data into MongoDB collection
collection.insert_many(memes_data)

print("Data inserted into MongoDB collection.")


@app.post("/uploadfile/")
async def upload_csv_file(file: UploadFile = File(...)):
    # Check if the uploaded file is a CSV file
    if file.filename.endswith(".csv"):
        # Read CSV file contents
        contents = await file.read()

        return {
            "status": "success",
            "message": "CSV file uploaded and data inserted into MongoDB collection.",
        }
    else:
        return {"status": "error", "message": "Please upload a CSV file."}


# Endpoint to like a meme
@app.post("/like/{meme_id}")
def like_meme(meme_id: str, token_data: TokenData = Depends(verify_token)):
    user_id = token_data.username
    # Add logic to record the user's like for the meme with ID `meme_id`
    # For example, update the user's preferences in the database
    user_preferences = db.user_preferences.find_one({"user_id": user_id})
    if not user_preferences:
        # If the user's preferences document doesn't exist, create a new one
        user_preferences = {"user_id": user_id, "liked_memes": []}
    # Check if the meme ID is not already in the user's liked memes
    if meme_id not in user_preferences["liked_memes"]:
        user_preferences["liked_memes"].append(meme_id)
        db.user_preferences.update_one(
            {"user_id": user_id}, {"$set": user_preferences}, upsert=True
        )
    return {"message": f"Meme {meme_id} liked by user {user_id}"}


# Function to find similar users
def find_similar_users(user_id: str):
    # Fetch liked memes for the current user
    user_preferences = db.user_preferences.find_one({"user_id": user_id})
    if not user_preferences:
        return []  # No preferences found for the user

    liked_memes = user_preferences.get("liked_memes", [])

    # Fetch liked memes for all users
    all_users = db.user_preferences.find({}, {"user_id": 1, "liked_memes": 1})
    all_users_preferences = {
        user["user_id"]: user.get("liked_memes", []) for user in all_users
    }

    # Create a matrix of liked memes for all users
    user_matrix = []
    for user, preferences in all_users_preferences.items():
        user_vector = [1 if meme in preferences else 0 for meme in liked_memes]
        user_matrix.append(user_vector)

    # Calculate cosine similarity between the current user and all other users
    similarity_scores = cosine_similarity(
        [user_matrix[liked_memes.index(meme_id)] for meme_id in liked_memes],
        user_matrix,
    )

    # Find users with high similarity scores
    similar_users_indices = np.argsort(similarity_scores)[
        -2:-6:-1
    ]  # Get top 5 similar users (excluding the current user)
    similar_users = [
        (all_users_preferences[liked_memes[user_index]], similarity_scores[user_index])
        for user_index in similar_users_indices
    ]

    return similar_users


# Endpoint to get similar users for the current user
@app.get("/similar_users/")
def get_similar_users(token_data: TokenData = Depends(verify_token)):
    user_id = token_data.username
    similar_users = find_similar_users(user_id)
    return {"similar_users": similar_users}


# Endpoint to generate token
@app.post("/token/", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Authenticate user here, replace with your authentication logic
    # For simplicity, let's just check if the username and password are correct
    if form_data.username == "user" and form_data.password == "password":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
