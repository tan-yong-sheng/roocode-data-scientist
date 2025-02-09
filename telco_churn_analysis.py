import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score

def load_and_preprocess_data(file_path):
    """Load and preprocess the telco churn dataset."""
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    # Create total services feature
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[services].apply(lambda x: sum(x != 'No'), axis=1)
    
    return df

def prepare_modeling_data(df):
    """Prepare data for modeling."""
    df_model = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df_model[col] = le.fit_transform(df_model[col])
    
    # Drop customerID
    df_model = df_model.drop('customerID', axis=1)
    
    return df_model

def train_models(X, y):
    """Train and evaluate classification models."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    class_weights = {0: 1, 1: len(y[y==0])/len(y[y==1])}
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight=class_weights)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'report': classification_report(y_test, y_pred)
        }
    
    return results, X_test, y_test

def perform_clustering(df, features):
    """Perform customer segmentation using KMeans clustering."""
    # Prepare data for clustering
    X_cluster = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Find optimal number of clusters
    silhouette_scores = []
    K = range(2, 6)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Use optimal K
    best_k = K[np.argmax(silhouette_scores)]
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, silhouette_scores, best_k

def main():
    # 1. Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_and_preprocess_data('telco-customer-churn.csv')
    
    # 2. Prepare data for modeling
    print("\nPreparing data for modeling...")
    df_model = prepare_modeling_data(df)
    
    # 3. Train and evaluate models
    print("\nTraining and evaluating models...")
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    model_results, X_test, y_test = train_models(X, y)
    
    # 4. Perform customer segmentation
    print("\nPerforming customer segmentation...")
    cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services']
    df_clustered, silhouette_scores, best_k = perform_clustering(df_model, cluster_features)
    
    # 5. Print results
    print("\nMODEL PERFORMANCE")
    print("----------------")
    for name, results in model_results.items():
        print(f"\n{name} Results:")
        print(results['report'])
    
    # Get feature importance from Random Forest
    rf_model = model_results['Random Forest']['model']
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTOP 10 MOST IMPORTANT FEATURES")
    print("-----------------------------")
    print(importance_df.head(10))
    
    print("\nCLUSTER ANALYSIS")
    print("---------------")
    print(f"Optimal number of clusters: {best_k}")
    print("\nSilhouette Scores:")
    for k, score in zip(range(2, 6), silhouette_scores):
        print(f"K={k}: {score:.3f}")
    
    # Analyze clusters
    for cluster in range(best_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Size: {len(cluster_data)} customers")
        print(f"Average Tenure: {cluster_data['tenure'].mean():.1f} months")
        print(f"Average Monthly Charges: ${cluster_data['MonthlyCharges'].mean():.2f}")
        print(f"Average Total Services: {cluster_data['total_services'].mean():.1f}")
        print(f"Churn Rate: {(cluster_data['Churn'].mean() * 100):.1f}%")

if __name__ == "__main__":
    main()