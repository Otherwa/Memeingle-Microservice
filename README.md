# Memeingle API

## Description
The Memeingle API provides endpoints to manage memes, users, and recommendations in the Memeingle application. The API offers functionalities to list memes, list users, get personalized meme recommendations, and calculate similarity scores between users based on their liked memes.

## Endpoints

### Memes
- **GET /memes**
  - Description: Retrieve a list of memes from the MongoDB collection.
  - Response: JSON array of meme objects.

### Users
- **GET /users**
  - Description: Retrieve a list of users from the MongoDB collection.
  - Response: JSON array of user objects.

### Recommendations
- **GET /recommendations/{user_id}**
  - Description: Get the top 10 meme recommendations for a specific user based on collaborative filtering.
  - Parameters:
    - `user_id` (str): The ID of the user for whom recommendations are requested.
  - Response: JSON object containing the user ID and a list of recommended meme IDs.

### Similarity
- **GET /similarity/{user_id1}/{user_id2}**
  - Description: Get the similarity score between two users based on their liked memes.
  - Parameters:
    - `user_id1` (str): The ID of the first user.
    - `user_id2` (str): The ID of the second user.
  - Response: JSON object containing the user IDs and their similarity score.

- **GET /similar/{user_id}**
  - Description: Get the top 5 users most similar to the specified user based on their liked memes.
  - Parameters:
    - `user_id` (str): The ID of the user.
  - Response: JSON object containing the user ID and a dictionary of the top 5 similar users and their similarity scores.

## Data Structures
- **Meme**
  - Each meme object consists of various attributes such as ID, title, author, URL, etc.

- **User**
  - Each user object includes attributes like ID, email, password, liked memes, etc.

## Response Format
All API responses are in JSON format.

## Usage

### List Memes
Retrieve a list of memes from the MongoDB collection:
```http
GET /memes
```

### List Users
Retrieve a list of users from the MongoDB collection:
```http
GET /users
```

### Get Recommendations
Get the top 10 meme recommendations for a specific user:
```http
GET /recommendations/{user_id}
```
Replace `{user_id}` with the ID of the user for whom recommendations are requested.

### Get Similarity
Get the similarity score between two users:
```http
GET /similarity/{user_id1}/{user_id2}
```
Replace `{user_id1}` and `{user_id2}` with the IDs of the users to compare.

### Get Similar Users
Get the top 5 users most similar to the specified user:
```http
GET /similar/{user_id}
```
Replace `{user_id}` with the ID of the user.

## Setup and Running the API
1. **Install dependencies**:
   Ensure you have Python and pip installed. Install the required packages using:
   ```bash
   pip install fastapi pymongo bson pandas scikit-learn uvicorn
   ```

2. **MongoDB connection**:
   Update the MongoDB connection string in the script:
   ```python
   client = MongoClient("your_mongodb_connection_string")
   ```

3. **Run the API**:
   Start the FastAPI server using:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   The API will be available at `http://localhost:8000`.

## API Documentation
Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/api/v1/docs`
- OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`

## Notes
- Ensure MongoDB is running and accessible via the provided connection string.
- The API calculates recommendations and similarities using collaborative filtering based on user liked memes. Adjustments to the algorithms and data structures can be made based on specific application needs.

## License
This project is licensed under the MIT License.
