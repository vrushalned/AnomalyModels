import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib  # Safer alternative for model serialization
import httpx
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model  # For loading TensorFlow models
from typing import List
import shap

# Initialize FastAPI application
app = FastAPI()

# Data storage paths
MODEL_DIR = "models"
TENSORFLOW_MODEL_PATH = os.path.join(MODEL_DIR, "tensorflow_anomaly_model.h5")

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Helper function to preprocess commit data
def preprocess_data(data: pd.DataFrame):
    # Feature engineering: Example, could be based on Additions, Deletions, TotalChanges, etc.
    features = data[['Additions', 'Deletions', 'TotalChanges', 'SentimentScore']].copy()
    features = StandardScaler().fit_transform(features)
    return features

# Helper function for training anomaly detection model (Multiple approaches)
def train_anomaly_detection_model(data: pd.DataFrame, method: str, repo: str):
    features = preprocess_data(data)
    
    if method == 'isolation_forest':
        model = IsolationForest(contamination=0.1)
    elif method == 'one_class_svm':
        model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    elif method == 'autoencoder':
        model = train_autoencoder(features)
        return model
    else:
        raise HTTPException(status_code=400, detail="Invalid method")

    model.fit(features)
    
    # Save the model for this specific repository
    repo_model_path = os.path.join(MODEL_DIR, f"{repo}_anomaly_model.joblib")
    joblib.dump(model, repo_model_path)
    return model

# Helper function for training an autoencoder (TensorFlow)
def train_autoencoder(features: np.ndarray):
    # Defining a simple Autoencoder using TensorFlow
    model = Sequential([ 
        Dense(64, activation='relu', input_shape=(features.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(features.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features, features, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    model.save(TENSORFLOW_MODEL_PATH)  # Save model
    return model

# Helper function to calculate sentiment score for commit messages
def calculate_sentiment_score(commit_message: str):
    # Using TextBlob or VaderSentiment to analyze sentiment
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(commit_message)
    return sentiment['compound']

# Function to evaluate model
def evaluate_model(model, data: pd.DataFrame):
    features = preprocess_data(data)
    
    if isinstance(model, IsolationForest) or isinstance(model, OneClassSVM):
        predictions = model.predict(features)
        report = classification_report([1] * len(predictions), predictions)  # Assuming we have ground truth
        return report
    elif isinstance(model, Sequential):
        predictions = model.predict(features)
        # Anomaly detection based on reconstruction error
        error = np.mean(np.abs(features - predictions), axis=1)
        return error

# Endpoint to fetch commits, train anomaly detection model and return commits
@app.post("/fetch_and_train_model/") 
async def fetch_and_train_model(owner: str, repo: str, token: str, method: str = 'isolation_forest'):
    headers = {"Authorization": f"token {token}"}
    commit_list = []
    
    # Fetch all branches
    async with httpx.AsyncClient() as client:
        branches_response = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/branches", headers=headers
        )
        if branches_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to fetch branches from GitHub")
        
        branches = branches_response.json()
        for branch in branches:
            branch_name = branch["name"]
            page = 1
            
            # Fetch all commits for this branch
            while True:
                commits_response = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/commits",
                    headers=headers,
                    params={"sha": branch_name, "per_page": 100, "page": page},
                )
                if commits_response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Unable to fetch commits from GitHub")
                
                commits_data = commits_response.json()
                if not commits_data:  # Break the loop if no more data
                    break
                
                for commit in commits_data:
                    sha = commit['sha']
                    
                    # Fetch detailed commit information
                    commit_detail_response = await client.get(
                        f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}",
                        headers=headers,
                    )
                    if commit_detail_response.status_code != 200:
                        continue  # Skip if commit details can't be fetched
                        
                    commit_detail = commit_detail_response.json()
                    stats = commit_detail.get('stats', {})
                    files_changed = [
                        {"filename": f['filename'], "additions": f['additions'], "deletions": f['deletions']}
                        for f in commit_detail.get('files', [])
                    ]
                    
                    # Calculate sentiment score for commit message
                    sentiment_score = calculate_sentiment_score(commit['commit']['message'])

                    commit_list.append({
                        'RepositoryName': repo,
                        'Branch': branch_name,
                        'Sha': sha,
                        'AuthorName': commit['commit']['author']['name'],
                        'AuthorEmail': commit['commit']['author']['email'],
                        'CommitMessage': commit['commit']['message'],
                        'Additions': stats.get('additions', 0),
                        'Deletions': stats.get('deletions', 0),
                        'TotalChanges': stats.get('total', 0),
                        'FilesChanged': files_changed,
                        'SentimentScore': sentiment_score,
                        'CommitDate': commit['commit']['author']['date'],
                    })
                
                # Check for pagination
                if "next" not in commits_response.headers.get("Link", ""):
                    break
                page += 1
    
    commit_df = pd.DataFrame(commit_list)
    
    # Train the model based on the selected method and save it for the specific repository
    model = train_anomaly_detection_model(commit_df, method, repo)
    
    # Return the commits data
    return {"status": "Model trained successfully", "commits": commit_list}

# Function to calculate feature importance using SHAP
def calculate_feature_importance(model, data):
    # Using SHAP to explain the model’s predictions
    if isinstance(model, (IsolationForest, OneClassSVM)):
        explainer = shap.KernelExplainer(model.predict, data)
    else:
        explainer = shap.KernelExplainer(model.predict, data)
    
    shap_values = explainer.shap_values(data)
    
    # Create a summary of the feature importance
    feature_names = pd.DataFrame(data).columns  # Assuming the data is a pandas DataFrame
    if len(shap_values) > 1:
        feature_importance = {
            feature_names[i]: np.abs(shap_values[i]).mean()  # Get the mean of SHAP values for each feature
            for i in range(len(feature_names))
        }
    else:
        feature_importance = {
            feature_names[0]: np.abs(shap_values[0]).mean()  # Get the mean of SHAP values for each feature
        }

    return feature_importance, shap_values

# Endpoint to check anomalies for a repository's commits
'''
@app.post("/check_anomalies/") 
async def check_anomalies_with_shap(
    owner: str,
    repo: str,
    commits_data: List[dict],  # Input list of commits
    threshold: float = 0.5
):
    # Define repository-specific model path
    repo_model_path = os.path.join(MODEL_DIR, f"{repo}_anomaly_model.joblib")
    
    # Check if the model exists for the repository
    if not os.path.exists(repo_model_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Model not trained yet for repository {owner}/{repo}"
        )
    
    # Load the trained model for the specific repository
    model = joblib.load(repo_model_path)
    
    # Convert commits data to DataFrame
    commit_df = pd.DataFrame(commits_data)
    
    # Preprocess data
    features = preprocess_data(commit_df)
    
    # Detect anomalies
    if isinstance(model, (IsolationForest, OneClassSVM)):
        predictions = model.predict(features)
    else:
        predictions = (model.predict(features) > threshold).astype(int)  # Assuming autoencoder model prediction
    
    anomalies = commit_df[predictions == -1]  # Anomalous data points

    # Calculate SHAP feature importance
    feature_importance, shap_values = calculate_feature_importance(model, features)
    
    return {
        "anomalies": anomalies.to_dict(orient='records'),
        "feature_importance": feature_importance,
        "shap_values": shap_values
    }
'''

@app.post("/check_anomalies/")
async def check_anomalies(
    repo: str,
    commits_data: List[dict],  # List of commit data (provided by the user)
    threshold: float = 0.5
):
    # Check if the model exists
    repo_model_path = os.path.join(MODEL_DIR, f"{repo}_anomaly_model.joblib")
    
    # Check if the model exists for the repository
    if not os.path.exists(repo_model_path):
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Load the trained model
    model = joblib.load(repo_model_path)
    
    # Convert the input data to a DataFrame
    commit_df = pd.DataFrame(commits_data)
    
    # Preprocess data
    features = preprocess_data(commit_df)  # Implement this function to extract features
    
    # Detect anomalies
    if isinstance(model, IsolationForest) or isinstance(model, OneClassSVM):
        anomalies = model.predict(features)
    elif isinstance(model, Sequential):
        predictions = model.predict(features)
        error = np.mean(np.abs(features - predictions), axis=1)
        anomalies = (error > threshold).astype(int)
    
    # Attach anomaly labels to the input data
    commit_df["Anomaly"] = anomalies
    commit_df["Reason"] = None
    for idx, row in commit_df.iterrows():
        if row["Anomaly"] == -1:  # If it's an anomaly
            reasons = []
            
            # Example rule-based reasons
            if row["Additions"] > commit_df["Additions"].mean() + 2 * commit_df["Additions"].std():
                reasons.append("High number of additions")
            if row["Deletions"] > commit_df["Deletions"].mean() + 2 * commit_df["Deletions"].std():
                reasons.append("High number of deletions")
            if row["TotalChanges"] > commit_df["TotalChanges"].mean() + 2 * commit_df["TotalChanges"].std():
                reasons.append("High total changes")
            if pd.to_datetime(row["CommitDate"]).hour not in range(8, 20):
                reasons.append("Unusual commit time")
            
            commit_df.at[idx, "Reason"] = ", ".join(reasons) if reasons else "Unusual pattern"
    
    # Return results
    return {
        "anomalies": commit_df[commit_df["Anomaly"] == -1][["Sha", "Anomaly", "Reason"]].to_dict(orient="records"),
        "all_commits": commit_df.to_dict(orient="records")
    }

