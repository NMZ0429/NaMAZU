from typing import List
import torch
from torch import Tensor
import math

__all__ = ["NBC"]


class NBC(torch.nn.Module):
    """Naive Bayes Classifier.
    
    Initializer takes optional training data for auto training.
    
    Attributes:
        offset (int): An integer to increment the conditional probabilities
                      in order to smooth probabilities to avoid that a posterior
                      probability be 0.
        is_categorical (List[bool]): A list of boolean for indicating if a feature is categorical or numerical.
        nb_features (int): An integer for the numbers of feature of the data.
        nb_class (int): An integer for the number of classes in the labels.
        class_probs (Tensor): A torch tensor for the proportion of each class.
        cond_probs (Tensor): A torch tensor for the conditional probability of
                             having a given value on a certain feature in the population 
                             of each class.
        pi: A torch tensor for the value of pi.
    """

    def __init__(
        self,
        offset: int = 1,
        training_data: Tensor = None,
        training_labels: Tensor = None,
        is_categorical: List[bool] = None,
    ):
        super().__init__()
        self.offset: int = offset
        self.pi = torch.tensor(math.pi)

        if training_data and training_labels and is_categorical:
            self.fit(training_data, training_labels, is_categorical)

    def fit(self, X: Tensor, y: Tensor, is_categorical: List[bool]):
        """Fits the model given data and labels as input
        
        Args:
            X (Tensor): A torch tensor containing a batch of data.
            y (Tensor): A torch tensor containing a batch of labels.
            is_categorical (Tensor) : A list of boolean for indicating if a feature is categorical or numerical.
        
        Returns:
            int: 0 if the model is fitted

        Raises:
            ValueError: If the number of features is different from the number of features in X.
        """
        # It is mandatory to pass a list describing if each feature is categorical or numerical
        if len(is_categorical) != X.shape[1]:
            raise ValueError(
                "The number of features in the data and the list of categorical features must be the same"
            )

        self.is_categorical = is_categorical
        size = X.shape[0]

        self.nb_features = X.shape[1]
        y_uvals = y.unique()
        self.nb_class = len(y_uvals)
        # Probability of each class in the training set
        self.class_probs = y.int().bincount().float() / size

        features_maxvals = torch.zeros((self.nb_features,), dtype=torch.int32)
        for j in range(self.nb_features):
            features_maxvals[j] = X[:, j].max()

        # All the posterior probabilites
        cond_probs = []
        for i in range(self.nb_class):
            cond_probs.append([])
            # Group samples by class
            idx = torch.where(y == y_uvals[i])[0]
            elts = X[idx]

            size_class = elts.shape[0]
            for j in range(self.nb_features):
                cond_probs[i].append([])
                if self.is_categorical[j]:
                    # If categorical
                    # For each features
                    for k in range(features_maxvals[j] + 1):
                        # Count the number of occurence of each value in this feature given the group class
                        # Divided by the number of samples in the class
                        p_x_k = (
                            torch.where(elts[:, j] == k)[0].shape[0] + self.offset
                        ) / size_class
                        # Append to posteriors probabilities
                        cond_probs[i][j].append(p_x_k)
                else:
                    # If numerical
                    features_class = elts[:, j]
                    # Compute mean and std
                    mean = features_class.mean()
                    std = (features_class - mean).pow(2).mean().sqrt()
                    # Store these value to use them for the gaussian likelihood
                    cond_probs[i][j] = [mean, std]
        self.cond_probs = cond_probs
        return 0

    def _gaussian_likelihood(self, X, mean, std):
        """Computes the gaussian likelihood
        
        Args:
            X: A torch tensor for the data.
            mean: A float for the mean of the gaussian.
            std: A flot for the standard deviation of the gaussian.
        """
        return (1 / (2 * self.pi * std.pow(2))) * torch.exp(
            -0.5 * ((X - mean) / std).pow(2)
        )

    def forward(self, X):
        """Predicts labels given an input
        
        Args:
            X: A torch tensor containing a batch of data.
        """
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        nb_samples = X.shape[0]
        pred_probs = torch.zeros((nb_samples, self.nb_class), dtype=torch.float32)
        for k in range(nb_samples):
            elt = X[k]
            for i in range(self.nb_class):
                # Set probability by the prior (class probability)
                pred_probs[k][i] = self.class_probs[i]
                prob_feature_per_class = self.cond_probs[i]
                for j in range(self.nb_features):
                    if self.is_categorical[j]:
                        # If categorical get the probability of drawing the value of the input on feature j
                        # inside class i
                        pred_probs[k][i] *= prob_feature_per_class[j][elt[j].int()]
                    else:
                        # If numerical, multiply by the gaussian likelihood with parameters
                        # mean and std of the class i on feature j
                        mean, std = prob_feature_per_class[j]
                        pred_probs[k][i] *= self._gaussian_likelihood(elt[j], mean, std)
        # Get to highest probability among all classes
        return pred_probs.argmax(dim=1)
