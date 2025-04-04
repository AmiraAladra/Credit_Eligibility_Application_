from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function to train the model
def train_RFmodel(X, y):
    try:
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    except ValueError as e:
        print(f"Error: There was an issue with splitting the data: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while splitting the data: {e}")
        exit(1)

    try:
        # Scale the data using MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        print(f"Error: There was an issue with scaling the data: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while scaling the data: {e}")
        exit(1)

    try:
        # Train the RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=2,
                                       max_depth=2,
                                       max_features=8).fit(X_train_scaled, y_train)
    except ValueError as e:
        print(f"Error: There was an issue with training the model: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while training the model: {e}")
        exit(1)

    try:
        # Save the trained model
        with open('models/RFmodel.pkl', 'wb') as f:
            pickle.dump(model, f)
    except FileNotFoundError:
        print("Error: The directory 'models' does not exist.")
        exit(1)
    except PermissionError:
        print("Error: Permission denied while trying to save the model.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving the model: {e}")
        exit(1)

    return model, X_test_scaled, y_test
