import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random



def get_data():
    with open('train.csv') as training_data:
        rows = training_data.readlines()[1:]
    
    ys = []
    xs = []
    for row in rows:
        raw_numbers = list(map(int, row.split(',')))
        ys.append(raw_numbers[0])
        xs.append(torch.tensor(raw_numbers[1:]).float() / 255.0 )
    return torch.stack(xs), torch.tensor(ys)


class SimpleDigitClassifier(nn.Module):
    """
    Simple single-layer neural network for digit classification.
    - Input: 784 features (28x28 flattened image)
    - Output: 10 classes (digits 0-9)
    - Architecture: Just one linear layer (10 perceptrons with 784 inputs each)
    """

    def __init__(self):
        super(SimpleDigitClassifier, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 784) - flattened images

        Returns:
            Probabilities over 10 classes (batch_size, 10)
        """
        # Linear transformation
        logits = self.linear(x)

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        return probs

    def predict(self, x):
        """
        Get the predicted class (digit with highest probability).

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        probs = self.forward(x)
        return torch.argmax(probs, dim=1)


def main():
    # Example usage
    model = SimpleDigitClassifier()

    # Create a random batch of flattened images (batch_size=5, 784 pixels)
    #sample_images = torch.randn(5, 784)
    xs, ys = get_data()
    sample_images = xs[:5]


    # Get probabilities
    probabilities = model(sample_images)
    print("Probabilities shape:", probabilities.shape)
    print("\nSample probabilities for first image:")
    print(probabilities[0])

    # Get predictions
    predictions = model.predict(sample_images)
    print("\nPredicted digits:", predictions)


if __name__ == "__main__":
    main()
