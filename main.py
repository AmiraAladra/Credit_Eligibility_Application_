from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import  plot_feature_importance
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    try:
        # Load and preprocess the data
        data_path = "data/raw/credit.csv"
        df = load_and_preprocess_data(data_path)
    except FileNotFoundError:
        print(f"Error: The file at {data_path} was not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading and preprocessing the data: {e}")
        exit(1)

    try:
        # Create dummy variables and separate features and target
        X, y = create_dummy_vars(df)
    except KeyError as e:
        print(f"Error: Missing required column in the dataset: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred while creating dummy variables: {e}")
        exit(1)

    try:
        # Train the Random Forest model
        model, X_test_scaled, y_test = train_RFmodel(X, y)
    except ValueError as e:
        print(f"Error: Issue with model training: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        exit(1)

    try:
        # Evaluate the model
        plot_feature_importance(model, X)
        accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{confusion_mat}")
    except Exception as e:
        print(f"An error occurred while evaluating the model: {e}")
        exit(1)
