from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson import json_util, ObjectId
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from deta import Deta
import math

app = FastAPI()

# Initialize Deta with your project key
DETA_PROJECT_KEY = "your_deta_project_key"
deta = Deta(DETA_PROJECT_KEY)
cache_base_personality = deta.Base("user_personality")

# Connect to MongoDB
client = MongoClient("your_mongodb_connection_string")
db = client["Memeingle"]
USERS = db["users"]
MEMES = db["memes"]

async def get_cached_data(key):
    data = cache_base_personality.get(key)
    return data.get("value") if data else None

async def set_cache_data(key, data, expire=10800):
    cache_base_personality.put({"key": key, "value": data}, key, expire_in=expire)

def extract_features(user_data, meme_data, subreddit_list):
    """Extract relevant features from user and meme data."""
    sentiment_scores = []
    upvotes = []
    subreddits = []

    for liked_meme in user_data["details"]["liked"]:
        meme_id = str(liked_meme["meme"])
        meme = meme_data.get(meme_id)

        if meme and "Sentiment" in meme and "UpVotes" in meme and "Subreddit" in meme:
            if not math.isnan(meme["Sentiment"]):
                sentiment_scores.append(meme["Sentiment"])
            if not math.isnan(meme["UpVotes"]):
                upvotes.append(meme["UpVotes"])
            subreddits.append(meme["Subreddit"])

    # Calculate averages and variance
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    avg_upvotes = np.mean(upvotes) if upvotes else 0
    sentiment_variance = np.var(sentiment_scores) if sentiment_scores else 0
    most_common_subreddit = Counter(subreddits).most_common(1)[0][0] if subreddits else None

    # Create subreddit vector
    subreddit_vector = [1 if most_common_subreddit == subreddit else 0 for subreddit in subreddit_list]

    # Create feature vector
    feature_vector = [
        avg_sentiment,
        avg_upvotes,
        *subreddit_vector,
        len(sentiment_scores),
        sentiment_variance,
    ]

    return feature_vector

@app.get("/predict-personality/{user_id}")
async def predict_personality(user_id: str):
    cached_data = await get_cached_data(user_id)
    if cached_data:
        return cached_data

    try:
        # Fetch all users and memes data
        user_data_list = list(USERS.find({}))
        meme_data_dict = {str(meme["_id"]): meme for meme in MEMES.find({})}
        subreddit_list = MEMES.distinct("Subreddit")

        if len(user_data_list) < 3:
            raise ValueError("Not enough users to perform clustering")

        # Find the target user data
        user_data = USERS.find_one({"_id": ObjectId(user_id)})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        # Extract features for the target user and all users
        X_user = extract_features(user_data, meme_data_dict, subreddit_list)
        X = [extract_features(user, meme_data_dict, subreddit_list) for user in user_data_list]

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        # Determine optimal number of clusters using silhouette scores
        silhouette_scores = []
        max_clusters = min(10, len(user_data_list) // 2)

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_imputed)
            silhouette_scores.append(silhouette_score(X_imputed, kmeans.labels_))

        optimal_clusters = np.argmax(silhouette_scores) + 2

        # Apply KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_imputed)

        # Use cluster labels as target variable for training the classifier
        y = [f"Cluster {label}" for label in cluster_labels]

        # Train a Decision Tree Classifier
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_imputed, y)

        # Predict personality for the target user
        X_user_imputed = imputer.transform([X_user])
        predicted_cluster_label = clf.predict(X_user_imputed)[0]

        # Personality mapping
        personality_mapping = {
            "Cluster 0": "Highly Positive and Engaged",
            "Cluster 1": "Moderate Sentiment and Engagement",
            "Cluster 2": "Low Sentiment, High Engagement",
            "Cluster 3": "Mixed Sentiment, Niche Interests",
            "Cluster 4": "High Sentiment, Low Engagement",
        }

        predicted_personality_description = personality_mapping.get(predicted_cluster_label, "Unknown Personality")

        # Cluster distribution across all users
        all_user_predictions = clf.predict(X_imputed)
        cluster_distribution = Counter(all_user_predictions)

        # Calculate percentages
        total_predictions = sum(cluster_distribution.values())
        cluster_distribution_percent = {
            cluster: round((count / total_predictions) * 100, 2)
            for cluster, count in cluster_distribution.items()
        }

        # Prepare the response
        response = {
            "user_id": str(user_id),
            "predicted_personality": predicted_personality_description,
            "cluster_distribution": {
                "Highly Positive and Engaged": cluster_distribution_percent.get("Cluster 0", 0),
                "Moderate Sentiment and Engagement": cluster_distribution_percent.get("Cluster 1", 0),
                "Low Sentiment, High Engagement": cluster_distribution_percent.get("Cluster 2", 0),
                "Mixed Sentiment, Niche Interests": cluster_distribution_percent.get("Cluster 3", 0),
                "High Sentiment, Low Engagement": cluster_distribution_percent.get("Cluster 4", 0),
            },
        }

        # Cache the response for future use
        await set_cache_data(user_id, response)
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
