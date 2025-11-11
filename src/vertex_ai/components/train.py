from kfp.dsl import ClassificationMetrics
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.10",
    packages_to_install=["torch", "pandas", "scikit-learn", "joblib"],
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    train_metrics: Output[Metrics],
    val_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    batch_size: int = 1024,
    y_column: str = "binds",
) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Create synthetic dataset ---
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(1000, 10)
    y = (X.sum(dim=1) > 0).long()  # simple binary classification

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- 2. Define small NN ---
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 2)  # binary classification

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- 3. Training loop ---
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    # --- 4. Save model ---
    model_path = Path(model_output.path) / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
