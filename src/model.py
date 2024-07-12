import torch
import torch.nn as nn

# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Fourth convolutional block
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Fifth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            # Sixth convolutional block
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Seventh convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            # Eighth convolutional block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Ninth convolutional block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the input tensor
            nn.Linear(7 * 7 * 512, 4096),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(4096),# First fully connected layer
            nn.ReLU(),
            
            nn.Linear(4096, 1024),
            nn.Dropout(p=dropout),
            
            nn.BatchNorm1d(1024),# Second fully connected layer
            nn.ReLU(),
            
            nn.Linear(1024, num_classes),  # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through feature extraction layers
        x = self.features(x)
        # Pass through classifier layers
        x = self.classifier(x)
        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)

def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    out = model(images)
    assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"
