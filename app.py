from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import redis
import logging


from bson import json_util
from bson.objectid import ObjectId

from concurrent.futures import ThreadPoolExecutor, as_completed
from apscheduler.schedulers.background import BackgroundScheduler

import random
import math
import numpy as np
import json

import pandas as pd
from sklearn.base import check_is_fitted
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi.middleware.cors import CORSMiddleware
from functools import cache
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Async function to connect to Redis
def redis_connect():
    try:
        client = redis.StrictRedis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASS"),
        )
        # Optionally, test the connection by pinging the server
        client.ping()
        print("Successfully connected to Redis Cloud!")
        return client
    except redis.ResponseError as e:
        print(f"Redis response error: {e}")
        raise
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        raise


# * ? Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URL"))
db = client["Memeingle"]

MEMES = db["memes"]
USERS = db["users"]


# Global Redis client
redis_client = None

app = FastAPI(
    title="Memeingle API",
    description="This API provides endpoints to manage memes, users, and recommendations in the Memeingle application.",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    tags=[
        {"name": "Memes", "description": "Endpoints related to memes"},
        {"name": "Users", "description": "Endpoints related to users"},
        {"name": "Recommendations", "description": "Endpoints for recommendations"},
    ],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_redis():
    global redis_client
    redis_client = redis_connect()


initialize_redis()

CACHE_EXPIRE_IN_SECONDS = 200  # * 5 mins


# * Redis


# Function to generate Redis key
def generate_redis_key(val, key):
    return f"{val}:{key}"  # Example: "likes:123"


def get_cached_data(key, val):
    redis_key = generate_redis_key(key, val)
    try:
        data = redis_client.get(redis_key)
        print(data)
        return json.loads(data) if data else None
    except Exception as e:
        print(f"Error getting cached data: {e}")
        return None


# Set cache data in Redis
def set_cache_data(key, data, val, expire=CACHE_EXPIRE_IN_SECONDS):
    redis_key = generate_redis_key(key, val)
    redis_client.set(redis_key, json.dumps(data), ex=expire)


global_model = RandomForestClassifier(n_estimators=50, random_state=42)
imputer = SimpleImputer(strategy="mean")
accuracy = None
# * FAST-API


def load_user_similarity_matrix():
    # * ? Load user-item interaction data
    cursor = USERS.find({}, {"_id": 1, "details.liked": 1})
    user_likes = {
        str(doc["_id"]): [
            str(like["meme"]) for like in doc.get("details", {}).get("liked", [])
        ]
        for doc in cursor
    }
    # * ? Convert to user-item matrix
    mlb = MultiLabelBinarizer()
    user_item_matrix = pd.DataFrame(
        mlb.fit_transform(user_likes.values()),
        index=user_likes.keys(),
        columns=mlb.classes_,
    )

    # * ? Calculate cosine similarities between users
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    return user_similarity_df, user_item_matrix


def get_top_n_similar_users(user_id, n=5):

    user_similarity_df = load_user_similarity_matrix()[0]

    if user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail="User not found")
    similar_users = (
        user_similarity_df[user_id].sort_values(ascending=False).head(n + 1).index[1:]
    )
    return similar_users


def recommend_memes(user_id: str, top_n: int = 20) -> List[str]:
    user_similarity_df, user_item_matrix = load_user_similarity_matrix()

    if user_id not in user_item_matrix.index:
        raise HTTPException(status_code=404, detail="User not found")

    # * Find similar users
    similar_users = get_top_n_similar_users(user_id)

    # * Aggregate memes liked by similar users
    meme_scores = user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)

    # * Filter out memes the user has already liked
    liked_memes = set(
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 1].index
    )
    recommendations = [meme for meme in meme_scores.index if meme not in liked_memes]

    print("Reccommed 1")
    print(recommendations)

    # * If not enough recommendations, widen the pool of similar users
    similar_user_count = 13
    while len(recommendations) < top_n and similar_user_count < len(user_similarity_df):
        similar_user_count += 5
        similar_users = get_top_n_similar_users(user_id, n=similar_user_count)
        meme_scores = (
            user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)
        )

        new_recommendations = [
            str(meme) for meme in meme_scores.index if meme not in liked_memes
        ]

        print(new_recommendations)

        recommendations = list(
            dict.fromkeys(recommendations + new_recommendations)
        )  # * Remove duplicates and preserve order

    # * Ensure there are at least 20 recommendations
    recommendations = recommendations[:top_n]
    print("Reccommed 2")
    print(recommendations)

    # * Fill up with random memes from the collection to make a total of 20 memes
    remaining_count = top_n - len(recommendations)
    if remaining_count > 0:
        random_memes = list(MEMES.aggregate([{"$sample": {"size": remaining_count}}]))
        random_memes = [
            str(meme["_id"])
            for meme in random_memes
            if str(meme["_id"]) not in liked_memes
            and str(meme["_id"]) not in recommendations
        ]
        recommendations += random_memes

    print("Reccommed 3")
    print(recommendations)

    # * Ensure there are at least 20 memes
    if len(recommendations) < top_n:
        additional_random_memes = list(
            MEMES.aggregate([{"$sample": {"size": top_n - len(recommendations)}}])
        )
        additional_random_memes = [
            str(meme["_id"])
            for meme in additional_random_memes
            if str(meme["_id"]) not in liked_memes
            and str(meme["_id"]) not in recommendations
        ]
        recommendations += additional_random_memes

    print(recommendations)
    return recommendations[:top_n]


@cache
@app.get(
    "/memes",
    summary="Get list of memes",
    description="Returns a list of memes from the MongoDB collection.",
)
async def list_memes():
    memes = MEMES.find()
    memes_list = [
        json.loads(json_util.dumps(meme))
        for meme in memes[: random.randrange(20, 50, 3)]
    ]
    return memes_list


@cache
@app.get(
    "/users",
    summary="Get list of users",
    description="Returns a list of users from the MongoDB collection.",
)
async def list_user():
    users = USERS.find()
    users_list = [json.loads(json_util.dumps(user)) for user in users]
    return users_list


@cache
@app.get(
    "/recommendations/{user_id}",
    summary="Get top 10 meme recommendations for a user",
    description="Returns the top 34 meme recommendations for the specified user based on collaborative filtering.",
)
async def get_recommendations(user_id: str):
    cache_key = f"recommendations:{user_id}"
    print(cache_key)
    cached_data = get_cached_data(cache_key, "likes")
    if cached_data:
        print(cached_data)
        return cached_data

    recommendations = recommend_memes(user_id.strip(), top_n=34)
    response = {"user_id": user_id.strip(), "recommendations": recommendations}

    set_cache_data(cache_key, response, "likes")
    return response


@cache
@app.get(
    "/similar/{user_id}",
    summary="Get similarity between users",
    description="Returns the similarity score between two users based on their liked memes.",
)
async def similar_users(user_id: str):
    # * ? Calculate cosine similarities between users
    user_item_matrix = load_user_similarity_matrix()[1]

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    print(user_similarity_df)

    str_user_id = str(user_id).strip()

    # * Check cache first
    cache_key = f"similar:{user_id}"
    cached_data = get_cached_data(cache_key, "similar")
    if cached_data:
        print("Cache Hit")
        return cached_data

    # * Calculate cosine similarities between users
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    print(user_similarity_df)

    if str_user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail="User not found")

    similarity_score_users = user_similarity_df.loc[str_user_id, :]

    print(user_similarity_df.loc[str_user_id, :])

    # * Sort the similarity scores in descending order and get the top 5 users
    top_5_similar_users = similarity_score_users.sort_values(ascending=False).head(6)

    # * Exclude the user itself from the top similar users
    top_5_similar_users = top_5_similar_users[
        top_5_similar_users.index != str_user_id
    ].head(5)

    # * Prepare result
    result = {"user": str_user_id, "data": top_5_similar_users.to_dict()}
    print(result)
    # * Cache the result
    set_cache_data(cache_key, result, "similar", CACHE_EXPIRE_IN_SECONDS)

    return result


# Function to extract features (optimized version from previous)
def extract_user_features_optimized(user_data, meme_data, subreddit_list):
    """Optimized feature extraction for a user."""
    sentiment_scores = []
    upvotes = []
    subreddit_vector_sums = np.zeros(len(subreddit_list), dtype=int)

    for liked_meme in user_data["details"]["liked"]:
        meme_id = str(liked_meme["meme"])
        meme = meme_data.get(meme_id)

        if meme:
            sentiment = meme.get("Sentiment", np.nan)
            upvote = meme.get("UpVotes", np.nan)
            subreddit = meme.get("Subreddit", None)

            if not np.isnan(sentiment) and not np.isnan(upvote) and subreddit:
                sentiment_scores.append(sentiment)
                upvotes.append(upvote)
                subreddit_vector_sums[subreddit_list.index(subreddit)] += 1

    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    avg_upvotes = np.mean(upvotes) if upvotes else 0
    sentiment_variance = np.var(sentiment_scores) if sentiment_scores else 0
    most_common_subreddit_vector = (subreddit_vector_sums > 0).astype(int).tolist()

    feature_vector = [
        avg_sentiment,
        avg_upvotes,
        *most_common_subreddit_vector,  # Ensure this part always has the same length
        len(sentiment_scores),
        sentiment_variance,
    ]
    return np.nan_to_num(feature_vector).tolist()


def assign_personality(avg_sentiment, avg_upvotes, liked_memes, meme_data):
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for meme in liked_memes:
        cur_meme = meme_data.get(str(meme.get("meme")))
        sentiment = cur_meme.get("Sentiment")

        if sentiment is not None and isinstance(sentiment, (float, int)):
            if sentiment > 0.5:
                positive_count += 1
            elif sentiment < 0.3:
                negative_count += 1
            else:
                neutral_count += 1

    if positive_count > negative_count:
        personality = "ENFP" if avg_upvotes > 1000 else "ESFJ"
    elif negative_count > positive_count:
        personality = "ISTP" if avg_upvotes < 100 else "INTJ"
    else:
        personality = "ISFJ" if avg_sentiment > 0.5 else "ISTJ"

    return personality, positive_count, negative_count, neutral_count


def retrain_model():
    """Retrain the global model every 30 minutes."""
    global global_model
    global imputer
    global accuracy

    logger.info("Retraining the model...")

    user_data_list = list(USERS.find({}))
    meme_data_dict = {str(meme["_id"]): meme for meme in MEMES.find({})}
    subreddit_list = MEMES.distinct("Subreddit")

    # Ensure we have enough data
    if len(user_data_list) < 3:
        return

    # Collect features and labels
    X = []
    y = []

    for user in user_data_list:
        features = extract_user_features_optimized(user, meme_data_dict, subreddit_list)
        personality, _, _, _ = assign_personality(
            features[0], features[1], user["details"]["liked"], meme_data_dict
        )
        X.append(features)
        y.append(personality)

    # Impute and train

    X_imputed = imputer.fit_transform(X)

    # Split data and train
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.3, random_state=42
    )

    # Lock the model while updating it
    new_model = RandomForestClassifier(n_estimators=50, random_state=42)
    new_model.fit(X_train, y_train)
    global_model = new_model  # Update global model

    accuracy = accuracy_score(y_test, global_model.predict(X_test))
    print("Model retrained with accuracy:", accuracy)
    logger.info("Model retraining completed successfully.")
    return accuracy


@app.get("/predict-personality/{user_id}")
async def predict_personality(user_id: str):
    global imputer
    global accuracy

    cached_data = get_cached_data(user_id, "personality")

    if cached_data:
        return cached_data

    try:
        user_data = USERS.find_one({"_id": ObjectId(user_id)})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        meme_data_dict = {str(meme["_id"]): meme for meme in MEMES.find({})}
        subreddit_list = MEMES.distinct("Subreddit")

        # Extract features for the target user
        X_user = extract_user_features_optimized(
            user_data, meme_data_dict, subreddit_list
        )

        # Ensure the global model is fitted; if not, retrain it
        try:
            check_is_fitted(global_model)
        except:
            accuracy = retrain_model()

        # Predict personality

        X_user_imputed = imputer.transform(
            [X_user]
        )  # Changed to `transform` instead of `fit_transform`
        predicted_personality = global_model.predict(X_user_imputed)[0]

        # Assign metrics for positive, negative, and neutral counts
        avg_sentiment = X_user[0]
        avg_upvotes = X_user[1]
        personality, positive_count, negative_count, neutral_count = assign_personality(
            avg_sentiment, avg_upvotes, user_data["details"]["liked"], meme_data_dict
        )

        # Prepare the response with metrics included
        response = {
            "predicted_personality": predicted_personality,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "metrics": {
                "accuracy": accuracy,
                "average_sentiment": avg_sentiment,
                "average_upvotes": avg_upvotes,
                "sentiment_variance": X_user[
                    -1
                ],  # Last element in X_user is sentiment variance
            },
        }

        set_cache_data(user_id, response, "personality", 800)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Starting FastAPI application...")
    # Set up a scheduler to retrain the model every 30 minutes
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_model, "interval", minutes=30)
    scheduler.start()
