"""
Machine Learning Model - Python ML
A simple classification model using scikit-learn.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class MLModel:
    """A simple Machine Learning classification model."""

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the ML model.

        Args:
            n_estimators: Number of trees in the random forest.
            random_state: Random seed for reproducibility.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the model on the provided data.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Features to predict.

        Returns:
            Predicted labels.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary containing accuracy and classification report.
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return {
            "accuracy": accuracy,
            "report": report
        }


def main():
    """Main function to demonstrate the ML model."""
    # Load the Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    print("Training the model...")
    model = MLModel(n_estimators=100, random_state=42)
    model.train(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    results = model.evaluate(X_test, y_test)

    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])

    # Make sample predictions
    print("\nSample predictions:")
    sample_data = X_test[:3]
    predictions = model.predict(sample_data)
    target_names = iris.target_names
    for i, pred in enumerate(predictions):
        print(f"  Sample {i + 1}: Predicted class = {target_names[pred]}")


if __name__ == "__main__":
    main()
