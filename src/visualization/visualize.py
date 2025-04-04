import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_importance(model, x):
    """
    Plot a bar chart showing the feature importances.
    
    Args:
        model (sklearn model): Trained model with feature importances.
        x (DataFrame): Input features (must have columns representing features).
    """
    try:
        # Check if the model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("The model does not have 'feature_importances_' attribute.")
        
        # Check if the model's feature importances match the number of input features
        if len(model.feature_importances_) != x.shape[1]:
            raise ValueError(f"Mismatch between number of features in model and input data. "
                             f"Model has {len(model.feature_importances_)} features, but input data has {x.shape[1]} features.")
        
        # Plotting the feature importance
        fig, ax = plt.subplots()
        ax = sns.barplot(x=model.feature_importances_, y=x.columns)
        plt.title("Feature importance chart")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        # Save the plot as a PNG file
        fig.savefig("feature_importance.png")
        print("Feature importance chart saved successfully.")
    
    except AttributeError as e:
        print(f"Error: {e}")
        exit(1)
    
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    except Exception as e:
        print(f"An unexpected error occurred while plotting feature importance: {e}")
        exit(1)
