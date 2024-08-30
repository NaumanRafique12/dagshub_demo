# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
import dagshub
dagshub.init(repo_owner='NaumanRafique12', repo_name='dagshub_demo', mlflow=True)

mlflow.set_tracking_uri("https://github.com/NaumanRafique12/dagshub_demo.git")
mlflow.set_experiment(experiment_name="Random_Forest_New")
n_estimators= 10
max_depth=2
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
with mlflow.start_run(run_name="all_artifacts_file_model_png"):
    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    mlflow.set_tag("Name","Noman")
    mlflow.set_tag("Algorithm","Random Forest")
    mlflow.set_tag("Params","n_estimators=100")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf_model,"random forest model")
    
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save the plot as an image file
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
   
    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
