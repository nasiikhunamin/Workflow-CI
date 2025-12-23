import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DAGSHUB_USERNAME = "nasikhunamin815"
DAGSHUB_REPO_NAME = "Workflow-CI"
DATA_PATH = 'mushroom_preprocessing/mushroom_clean.csv'

def main():

    try:
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    except:
        pass 

    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {DATA_PATH}")
        return

    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.autolog()

    with mlflow.start_run(): 
        print("Training Model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Run Selesai. Accuracy: {acc}")

if __name__ == "__main__":
    main()