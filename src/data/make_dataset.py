import pandas as pd

def load_and_preprocess_data(data_path):
    try:
        # Import the data from 'credit.csv'
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file at {data_path} was not found.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {data_path} is empty.")
        exit(1)
    except pd.errors.ParserError:
        print(f"Error: The file at {data_path} could not be parsed.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        exit(1)

    try:
        # Impute all missing values in all the features
        df['Gender'].fillna('Male', inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Education'].fillna(df['Education'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['ApplicantIncome'].fillna(df['ApplicantIncome'].median(), inplace=True)
        df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median(), inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)
    except KeyError as e:
        print(f"Error: Missing expected column in the data: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while imputing missing values: {e}")
        exit(1)

    try:
        # Drop 'Loan_ID' variable from the data
        df = df.drop('Loan_ID', axis=1)
    except KeyError:
        print("Error: 'Loan_ID' column is not present in the data.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while dropping 'Loan_ID': {e}")
        exit(1)

    return df
