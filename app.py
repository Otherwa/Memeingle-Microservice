import random
from fastapi import FastAPI, HTTPException, Depends
from pymongo import MongoClient
from bson import json_util
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from fastapi.middleware.cors import CORSMiddleware
from functools import cache
from typing import List
from deta import Deta


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

# ? Initialize Deta with your project key
DETA_PROJECT_KEY = "d0zuwufwggh_i5Y4sfgnP5YQg6imdd2zVqkqMRmUCCEC"
deta = Deta(DETA_PROJECT_KEY)

# Create a Base instance
cache_base_likes = deta.Base("user_likes")
cache_base_similar = deta.Base("user_similar")
CACHE_EXPIRE_IN_SECONDS = 200  # 5 mins

# ? Connect to MongoDB
client = MongoClient(
    "mongodb+srv://atharvdesai:ahrAA7kOTdZfyur9@cluster0.smf3kdb.mongodb.net/Memeingle?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["Memeingle"]

MEMES = db["memes"]
USERS = db["users"]


# DETA
async def get_cached_data(key, val):
    if val == "likes":
        data = cache_base_likes.get(key)
        if data:
            return data.get("value")
    elif val == "similar":
        data = cache_base_similar.get(key)
        if data:
            return data.get("value")
    return None


async def set_cache_data(
    key,
    data,
    val,
    expire=CACHE_EXPIRE_IN_SECONDS,
):
    if val == "likes":
        cache_base_likes.put({"key": key, "value": data}, key, expire_in=expire)
    elif val == "similar":
        cache_base_similar.put({"key": key, "value": data}, key, expire_in=expire)


# FAST-API


def load_user_similarity_matrix():
    # ? Load user-item interaction data
    cursor = USERS.find({}, {"_id": 1, "details.liked": 1})
    user_likes = {
        str(doc["_id"]): [
            str(like["meme"]) for like in doc.get("details", {}).get("liked", [])
        ]
        for doc in cursor
    }
    # ? Convert to user-item matrix
    mlb = MultiLabelBinarizer()
    user_item_matrix = pd.DataFrame(
        mlb.fit_transform(user_likes.values()),
        index=user_likes.keys(),
        columns=mlb.classes_,
    )

    # ? Calculate cosine similarities between users
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

    # Find similar users
    similar_users = get_top_n_similar_users(user_id)

    # Aggregate memes liked by similar users
    meme_scores = user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)

    # Filter out memes the user has already liked
    liked_memes = set(
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 1].index
    )
    recommendations = [meme for meme in meme_scores.index if meme not in liked_memes]

    print("Reccommed 1")
    print(recommendations)

    # If not enough recommendations, widen the pool of similar users
    similar_user_count = 13
    while len(recommendations) < top_n and similar_user_count < len(user_similarity_df):
        similar_user_count += 5
        similar_users = get_top_n_similar_users(user_id, n=similar_user_count)
        meme_scores = (
            user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)
        )
        new_recommendations = [
            str(meme["_id"]) for meme in meme_scores.index if meme not in liked_memes
        ]
        recommendations = list(
            dict.fromkeys(recommendations + new_recommendations)
        )  # Remove duplicates and preserve order

    # Ensure there are at least 20 recommendations
    recommendations = recommendations[:top_n]
    print("Reccommed 2")
    print(recommendations)

    # Fill up with random memes from the collection to make a total of 20 memes
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

    # Ensure there are at least 20 memes
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
    cached_data = await get_cached_data(cache_key, "likes")
    if cached_data:
        return cached_data

    recommendations = recommend_memes(user_id.strip(), top_n=34)
    response = {"user_id": user_id.strip(), "recommendations": recommendations}

    await set_cache_data(cache_key, response, "likes")
    return response


@cache
@app.get(
    "/similar/{user_id}",
    summary="Get similarity between users",
    description="Returns the similarity score between two users based on their liked memes.",
)
async def similar_users(user_id: str):
    # ? Calculate cosine similarities between users
    user_item_matrix = load_user_similarity_matrix()[1]

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    print(user_similarity_df)

    str_user_id = str(user_id).strip()

    # Check cache first
    cache_key = f"similar:{user_id}"
    cached_data = await get_cached_data(cache_key, "similar")
    if cached_data:
        print("Cache Hit")
        return cached_data

    # Calculate cosine similarities between users
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    print(user_similarity_df)

    if str_user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail="User not found")

    similarity_score_users = user_similarity_df.loc[str_user_id, :]

    print(user_similarity_df.loc[str_user_id, :])

    # Sort the similarity scores in descending order and get the top 5 users
    top_5_similar_users = similarity_score_users.sort_values(ascending=False).head(6)

    # Exclude the user itself from the top similar users
    top_5_similar_users = top_5_similar_users[
        top_5_similar_users.index != str_user_id
    ].head(5)

    # Prepare result
    result = {"user": str_user_id, "data": top_5_similar_users.to_dict()}
    print(result)
    # Cache the result
    await set_cache_data(cache_key, result, "similar", CACHE_EXPIRE_IN_SECONDS)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
