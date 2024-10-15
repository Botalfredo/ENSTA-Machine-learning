import numpy as np

def predict_multi(house_features, mu, sigma, theta):
    """
    Predicts the price of a house with given features using the learned
    model parameters (theta), mean (mu), and standard deviation (sigma).
    """
    # Normalize the features of the house (same as training)
    house_features_normalized = (house_features - mu) / sigma

    # Add intercept term to the features
    house_features_normalized = np.concatenate([np.ones((house_features_normalized.shape[0], 1)), house_features_normalized], axis=1)

    # Make the prediction
    price = house_features_normalized.dot(theta)

    return price
