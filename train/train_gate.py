import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model_objects.gate import SimpleGateCNN

# ----------------------------------------
# 1. Data Loading (Simpler now!)
# ----------------------------------------
def get_gate_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 0):
    """
    Loads data from 'data_dir/train' and 'data_dir/val'
    """
    # --- TRAINING TRANSFORMS ---
    train_tfms = transforms.Compose([
        # 1. Resize to Model Input Size
        transforms.Resize((128, 128)),

        # 2. Data Augmentation
        transforms.RandomHorizontalFlip(p=0.5),  # Flip left/right
        transforms.RandomRotation(10),  # Rotate slightly

        # 4. Convert to Tensor
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- VALIDATION TRANSFORMS ---
    # No random changes, just resize and normalize
    val_tfms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Load Datasets
    print(f"Loading Training Data from: {train_dir}")
    train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)

    print(f"Loading Validation Data from: {val_dir}")
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tfms)

    # Class Mapping Check
    print(f"Classes found: {train_ds.class_to_idx}")

    # Create Loaders
    # num_workers=0 is safer on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, train_ds.class_to_idx


# ----------------------------------------
# 2. Train / Eval Functions
# ----------------------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ----------------------------------------
# 3. Main Training Manager
# ----------------------------------------
def train_gate_manager(model, train_loader, valid_loader, device, class_weights, epochs=10):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Standard learning rate

    # Decays LR if validation accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    best_state = None

    print("\n--- Starting Training ---")

    for epoch in range(epochs):
        # 1. Train
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)

        # 2. Evaluate
        va_loss, va_acc = evaluate(model, valid_loader, device, criterion)

        # Step Scheduler
        scheduler.step(va_acc)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.1%} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.1%}")

        # 3. Save Best
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> New Best Model! ({best_acc:.1%})")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def plot_confusion_matrix(model, loader, device, classes):
    print("\nGenerating Confusion Matrix...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# ----------------------------------------
# 4. Main Execution
# ----------------------------------------
def main():
    # --- CONFIG ---
    data_dir = r"C:\Users\yoavt\PycharmProjects\final_projact\data\gate_dataset"
    output_dir = r"C:\Users\yoavt\PycharmProjects\final_projact\models"

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 1. Get Data
    # Batch size 32 is standard and stable
    train_loader, valid_loader, class_to_idx = get_gate_dataloaders(data_dir, batch_size=32)

    # Get class names in correct order (0, 1, 2, 3)
    # We swap keys and values to get [name_of_0, name_of_1, ...]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # 2. Build Model
    model = SimpleGateCNN().to(device)
    # --- ADD THIS BLOCK ---
    # Alphabetical order based on folder names:
    # 0: low_light, 1: low_res, 2: motion_blur, 3: normal
    # We give a 1.5x penalty to mistakes made on low_light and motion_blur
    weights = torch.tensor([1.5, 1.0, 1.5, 1.0], dtype=torch.float32).to(device)

    # 3. Train (Pass the weights)
    model = train_gate_manager(
        model,
        train_loader,
        valid_loader,
        device,
        class_weights=weights,  # Pass weights here
        epochs=13  # Increased slightly since GAP needs a bit more time to converge
    )

    # 4. Save Results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = os.path.join(output_dir, "gate_model_best_3.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n Saved best model to: {model_path}")

    # 5. Show Errors
    plot_confusion_matrix(model, valid_loader, device, class_names)


if __name__ == "__main__":
    main()