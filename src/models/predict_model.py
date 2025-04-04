# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):
    try:
        # Predict the loan eligibility on the testing set
        y_pred = model.predict(X_test_scaled)
    except ValueError as e:
        print(f"Error: There was an issue with making predictions: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while making predictions: {e}")
        exit(1)

    try:
        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
    except ValueError as e:
        print(f"Error: There was an issue with calculating accuracy: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while calculating accuracy: {e}")
        exit(1)

    try:
        # Calculate the confusion matrix
        confusion_mat = confusion_matrix(y_test, y_pred)
    except ValueError as e:
        print(f"Error: There was an issue with calculating the confusion matrix: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while calculating the confusion matrix: {e}")
        exit(1)

    return accuracy, confusion_mat
