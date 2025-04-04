Credit Loan Eligibility Predictor

Project Overview
This project aims to predict the eligibility of loan applicants based on various personal and financial characteristics. It uses a Random Forest Classifier to analyze the data and provide predictions about whether a loan applicant is eligible for a loan. The model is trained using features such as gender, marital status, income, credit history, and more.

The project includes data preprocessing, feature engineering, model training, evaluation, and visualization of the results.

Features
Data Preprocessing: Missing values are handled through imputation, and categorical variables are transformed into dummy variables.

Model Training: A Random Forest model is trained on the processed data.

Model Evaluation: The model's accuracy and confusion matrix are evaluated on the test set.

Visualization: Feature importance is visualized using a bar chart.

Error Handling: Basic error handling is included to ensure robustness during data processing, model training, and evaluation.

Technologies Used
Python 3.x

Libraries:

pandas: Data manipulation and preprocessing

scikit-learn: Machine learning algorithms and utilities

matplotlib & seaborn: Visualization of results

pickle: Saving and loading the trained model

Version control: Git, GitHub

Project Structure

Copy
.
├── data/
│   ├── raw/                       # Raw dataset (credit.csv)
│   └── processed/                 # Processed dataset and output files (e.g., Processed_Credit_Dataset.csv)
├── models/                        # Trained models (e.g., RFmodel.pkl)
├── src/
│   ├── data/                      # Scripts for data loading and preprocessing (make_dataset.py)
│   ├── features/                  # Scripts for feature engineering (build_features.py)
│   ├── models/                    # Scripts for model training and prediction (train_model.py, predict_model.py)
│   ├── visualization/             # Scripts for visualizing results (visualize.py)
│   └── main.py                    # Main entry point for running the entire pipeline
├── requirements.txt               # List of required Python packages
└── README.md                      # Project description and setup instructions
How to Run the Project
1. Clone the Repository
First, clone this repository to your local machine:


Copy
git clone https://github.com/yourusername/credit-loan-eligibility.git
cd credit-loan-eligibility
2. Install Dependencies
Make sure you have all the necessary Python dependencies installed. You can install them via pip using the requirements.txt file.


Copy
pip install -r requirements.txt
3. Running the Code
Once the dependencies are installed, you can run the pipeline by executing the main.py script. This will load and preprocess the data, train the model, and evaluate its performance.


Copy
python src/main.py
4. Resulting Files
Model: The trained Random Forest model will be saved in the models folder as RFmodel.pkl.

Feature Importance Visualization: A bar chart showing the feature importances will be saved as feature_importance.png in the current directory.

Processed Data: The preprocessed data will be saved in data/processed/Processed_Credit_Dataset.csv.

Code Explanation
1. load_and_preprocess_data(data_path)
This function loads the raw dataset from a CSV file and handles missing values by imputing them with appropriate values (mode or median). The Loan_ID column is dropped.

2. create_dummy_vars(df)
This function creates dummy variables for categorical features (e.g., Gender, Married, etc.) using pandas.get_dummies. It then splits the dataset into features (X) and target variable (y) for training.

3. train_RFmodel(X, y)
This function trains a Random Forest Classifier using the preprocessed data. The dataset is split into training and test sets, and the data is scaled using MinMaxScaler. The trained model is saved using pickle.

4. evaluate_model(model, X_test_scaled, y_test)
This function evaluates the trained model by predicting loan eligibility on the test set, calculating the accuracy score, and generating a confusion matrix.

5. plot_feature_importance(model, X)
This function generates a bar chart showing the importance of each feature in predicting loan eligibility, based on the trained model's feature_importances_. The chart is saved as an image file.

Contributing
Feel free to fork this repository and submit pull requests for any improvements or bug fixes. If you find any issues or have suggestions for new features, please open an issue.