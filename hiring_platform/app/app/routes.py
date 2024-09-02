from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

@app.route('/dashboard')
@login_required
def dashboard():
    data = pd.read_csv('data/jobs.csv')
    X = data[['skills', 'experience', 'education']]
    y = data['hired']

    # Feature Engineering: Convert categorical data to numeric
    X = pd.get_dummies(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    return f"Job Matching Accuracy: {accuracy:.2f}"
