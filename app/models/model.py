from ..config import config

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import io


class AttractivenessClassifier:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._model: models.ResNet = None

    def load(self, path: str):
        model = models.resnet50(weights="IMAGENET1K_V2")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),  # 512 features in the last layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization to prevent overfitting
            nn.Linear(512, 1),  # Final output: 1 value (the attractiveness score)
        )
        self._model = model
        self._model.to(self._device)
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        print(f"Model loaded from {path}")
        print(f"Model device: {self._device}")

    def predict(self, image: bytes) -> float:
        self._model.eval()
        try:
            # Convert bytes to PIL Image
            image: Image.Image = Image.open(io.BytesIO(image))

            # Apply transformations
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)

            # Make prediction
            with torch.no_grad():
                output = self._model(image_tensor)

            image.close()

            return output.item()

        except Exception as e:
            print(f"Error processing image: {e}")
            raise ValueError(f"Could not process image: {e}")

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs=10,
        lr=0.001,
    ):
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        best_loss = float("inf")

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self._model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    self._model.eval()  # Set model to evaluate mode
                    dataloader = test_loader

                running_loss = 0.0
                running_mae = 0.0  # Track Mean Absolute Error as well

                # Iterate over data
                for inputs, labels in dataloader:
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device).view(
                        -1, 1
                    )  # Ensure correct shape [batch_size, 1]

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self._model(inputs)
                        loss = criterion(outputs, labels)
                        mae = nn.L1Loss()(outputs, labels)  # Calculate MAE

                        # Backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_mae += mae.item() * inputs.size(0)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_mae = running_mae / len(dataloader.dataset)

                # Print statistics
                print(
                    f"{phase.capitalize()} Loss: {epoch_loss:.4f} MAE: {epoch_mae:.4f}"
                )

                # Save the best model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(self._model.state_dict(), config.best_model_path)
                    print(f"New best model saved with loss: {best_loss:.4f}")

            # Step the scheduler
            scheduler.step(epoch_loss)
            print()

        print(f"Training complete. Best val loss: {best_loss:.4f}")

        # Load the best model and do a final evaluation
        self._model.load_state_dict(torch.load(config.best_model_path))
        self._model.eval()

        self.evaluate(test_loader)

    def evaluate(self, test_loader: DataLoader):
        if not self._model:
            raise ValueError("Can't evaluate model, it's not loaded")
        self._model.eval()

        total_mae = 0.0
        total_rmse = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device).view(-1, 1)

                outputs = self._model(inputs)
                mae = nn.L1Loss()(outputs, labels)
                rmse = nn.MSELoss()(outputs, labels)

                total_mae += mae.item() * inputs.size(0)
                total_rmse += rmse.item() * inputs.size(0)

        mae = total_mae / len(test_loader.dataset)
        rmse = total_rmse / len(test_loader.dataset) ** 0.5

        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")

        return mae, rmse


model = AttractivenessClassifier()
