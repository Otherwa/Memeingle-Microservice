from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import json_util
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from fastapi.middleware.cors import CORSMiddleware
from functools import cache
from typing import List

"""
Memeingle API

Description:
The Memeingle API provides endpoints to manage memes, users, and recommendations in the Memeingle application.

Endpoints:
- Memes:
  - GET /memes: Get a list of memes from the MongoDB collection.

- Users:
  - GET /users: Get a list of users from the MongoDB collection.

- Recommendations:
  - GET /recommendations/{user_id}: Get the top 10 meme recommendations for the specified user based on collaborative filtering.
    Parameters:
      - user_id (str): The ID of the user for whom recommendations are requested.

- Similarity:
  - GET /similarity/{user_id1}/{user_id2}: Get the similarity score between two users based on their liked memes.
    Parameters:
      - user_id1 (str): The ID of the first user.
      - user_id2 (str): The ID of the second user.

Usage:
1. List Memes: Retrieve a list of memes from the MongoDB collection by sending a GET request to /memes.
2. List Users: Retrieve a list of users from the MongoDB collection by sending a GET request to /users.
3. Get Recommendations: Get the top 10 meme recommendations for a specific user by sending a GET request to /recommendations/{user_id} where {user_id} is the ID of the user for whom recommendations are requested.
4. Get Similarity: Get the similarity score between two users based on their liked memes by sending a GET request to /similarity/{user_id1}/{user_id2} where {user_id1} and {user_id2} are the IDs of the users to compare.

Data Structures:
- Meme: Each meme object consists of various attributes such as ID, title, author, URL, etc.
- User: Each user object includes attributes like ID, email, password, liked memes, etc.

Response Format:
The API returns data in JSON format.
"""


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


# ? Connect to MongoDB
client = MongoClient(
    "mongodb+srv://atharvdesai:ahrAA7kOTdZfyur9@cluster0.smf3kdb.mongodb.net/Memeingle?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["Memeingle"]

MEMES = db["memes"]
USERS = db["users"]


def load_user_similarity_matrix():
    # ? Load user-item interaction data
    cursor = USERS.find({}, {"_id": 1, "details.liked": 1})
    user_likes = {
        str(doc["_id"]): [str(like) for like in doc.get("details", {}).get("liked", [])]
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


def recommend_memes(user_id: str, top_n: int = 13) -> List[str]:
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

    # If not enough recommendations, widen the pool of similar users
    similar_user_count = 13
    while len(recommendations) < 6 and similar_user_count < len(user_similarity_df):
        similar_user_count += 5
        similar_users = get_top_n_similar_users(user_id, n=similar_user_count)
        meme_scores = (
            user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)
        )
        new_recommendations = [
            meme for meme in meme_scores.index if meme not in liked_memes
        ]
        recommendations = list(
            dict.fromkeys(recommendations + new_recommendations)
        )  # Remove duplicates and preserve order

    # Ensure there are at least 6 recommendations
    recommendations = recommendations[:6]

    # Fill up with random memes from the collection to make a total of 13 memes
    remaining_count = top_n - len(recommendations)
    random_memes = list(MEMES.aggregate([{"$sample": {"size": remaining_count}}]))
    random_memes = [
        str(meme["_id"])
        for meme in random_memes
        if str(meme["_id"]) not in liked_memes
        and str(meme["_id"]) not in recommendations
    ]

    final_recommendations = recommendations + random_memes

    # Ensure there are at least 13 memes
    if len(final_recommendations) < top_n:
        additional_random_memes = list(
            MEMES.aggregate([{"$sample": {"size": top_n - len(final_recommendations)}}])
        )
        additional_random_memes = [
            str(meme["_id"])
            for meme in additional_random_memes
            if str(meme["_id"]) not in liked_memes
            and str(meme["_id"]) not in final_recommendations
        ]
        final_recommendations += additional_random_memes

    return final_recommendations[:top_n]


@cache
@app.get(
    "/memes",
    summary="Get list of memes",
    description="Returns a list of memes from the MongoDB collection.",
)
async def list_memes():
    memes = MEMES.find()
    memes_list = [json.loads(json_util.dumps(meme)) for meme in memes]
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
    description="Returns the top 10 meme recommendations for the specified user based on collaborative filtering.",
)
async def get_recommendations(user_id: str):
    recommendations = recommend_memes(user_id.strip(), top_n=13)
    return {"user_id": user_id.strip(), "recommendations": recommendations}


@cache
@app.get(
    "/similarity/{user_id1}/{user_id2}",
    summary="Get similarity between two users",
    description="Returns the similarity score between two users based on their liked memes.",
)
async def user_similarity(user_id1: str, user_id2: str):
    # ? Calculate cosine similarities between users
    user_item_matrix = load_user_similarity_matrix()[1]

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    print(user_similarity_df)

    str_user_id1 = str(user_id1).strip()
    str_user_id2 = str(user_id2).strip()

    if (
        str_user_id1 not in user_similarity_df.index
        or str_user_id2 not in user_similarity_df.index
    ):
        raise HTTPException(status_code=404, detail="User not found")

    similarity_score = user_similarity_df.loc[str_user_id1, str_user_id2]

    return {
        "user_id1": str_user_id1,
        "user_id2": str_user_id2,
        "similarity_score": similarity_score,
    }


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

    if str_user_id not in user_similarity_df.index:
        raise HTTPException(status_code=404, detail="User not found")

    similarity_score_users = user_similarity_df.loc[str_user_id, :]

    # ? Sort the similarity scores in descending order and get the top 5 users

    top_5_similar_users = similarity_score_users.sort_values(ascending=False).head(
        6
    )  # ? head(6) to exclude the user itself

    # ? Exclude the user itself from the top similar users
    top_5_similar_users = top_5_similar_users[
        top_5_similar_users.index != str_user_id
    ].head(5)

    # ? Return the top 5 similar users and their similarity scores
    result = {"user": str_user_id, "data": top_5_similar_users.to_dict()}

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
