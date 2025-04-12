import pandas as pd

# create dummy features
def create_dummy_vars(df):
    try:
        # Create dummy variables for all 'object' type variables except 'Loan_Status'
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)
    except KeyError as e:
        print(f"Error: Missing expected column in the data for dummy variable creation: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while creating dummy variables: {e}")
        exit(1)

    try:
        # Store the processed dataset in data/processed
        df.to_csv('data/processed/Processed_Credit_Dataset.csv', index=None)
    except FileNotFoundError:
        print("Error: The directory 'data/processed' does not exist.")
        exit(1)
    except PermissionError:
        print("Error: Permission denied while trying to write the processed file.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving the processed data: {e}")
        exit(1)

    try:
        # Separate the input features and target variable
        X = df.drop('Loan_Approved', axis=1)
        y = df['Loan_Approved']
    except KeyError:
        print("Error: The 'Loan_Approved' column is missing in the dataset.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while separating features and target: {e}")
        exit(1)

    return X, y

