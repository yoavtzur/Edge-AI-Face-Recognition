import os
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ----------------------------------------
# Data
# ----------------------------------------
def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    Expects:
      data/train/<class_name>/*.jpg
      data/valid/<class_name>/*.jpg
    """

    # Mild augmentations (better for face-ID style classification)
    train_tfms = transforms.Compose([
        #ResNet expects 224x224 inputs
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.20, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        # values to match statistics of data the model was originally trained on
        transforms.Normalize(mean=ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
                             std=ResNet18_Weights.IMAGENET1K_V1.transforms().std),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
                             std=ResNet18_Weights.IMAGENET1K_V1.transforms().std),
    ])
    # assigns labels to each folder
    train_ds = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_tfms)
    valid_ds = datasets.ImageFolder(root=os.path.join(data_dir, "valid"), transform=valid_tfms)

    # Ensure identical mapping
    if train_ds.class_to_idx != valid_ds.class_to_idx:
        raise ValueError(
            f"Train and valid class_to_idx differ:\n"
            f"train: {train_ds.class_to_idx}\n"
            f"valid: {valid_ds.class_to_idx}"
        )

    # Prints counts per class ( useful to spot imbalance)
    train_counts = Counter(train_ds.targets) #counts number of pictures
    valid_counts = Counter(valid_ds.targets)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    print("Train counts:", {idx_to_class[i]: train_counts[i] for i in sorted(train_counts)})
    print("Valid counts:", {idx_to_class[i]: valid_counts[i] for i in sorted(valid_counts)})

    # Weighted sampler for imbalanced datasets
    # weight per sample = 1 / count(class)
    class_counts = torch.tensor([train_counts[i] for i in range(len(train_ds.classes))], dtype=torch.float)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[torch.tensor(train_ds.targets, dtype=torch.long)]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Definning the way in which the model gets the data.
    train_loader = DataLoader(   #
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # use sampler instead of shuffle
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, train_ds.class_to_idx


# ----------------------------------------
# Model
# ----------------------------------------
def build_model(num_classes: int):
    # Use official pretrained weights
    model = models.resnet18(weights=None)
    state_dict = torch.load(r"C:\Users\yoavt\PycharmProjects\final_projact\models\resnet18-f37072fd.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable_params(model: nn.Module, phase: int):
    """
    phase 1: train only fc (linear probing)
    phase 2: fine-tune layer4 + fc
    """
    # Phase 1: Freeze almost everything
    for p in model.parameters():
        p.requires_grad = False

    # Always train fc ( layer output)
    for p in model.fc.parameters():
        p.requires_grad = True

    # Phase 2: Unfreeze the last block(layer4)
    if phase >= 2:
        for p in model.layer4.parameters():
            p.requires_grad = True


def get_optimizer(model: nn.Module, phase: int):
    if phase == 1:
        lr = 1e-3   #learning rate for training
        params = model.fc.parameters()
    else:
        lr = 3e-5  # small learning rate for fine-tuning
        params = [p for p in model.parameters() if p.requires_grad]


    # weight decay is a technique used to prevent overfitting making the model
    # less likely to create too strong "opinions".
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    return optimizer


# -------------------------------------
# Train / Eval
# -------------------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # --- GPU OPTIMIZATION ---
        # non_blocking=True: Tells CPU "Send this data to GPU and move to next line immediately."
        # Because we used pin_memory=True in DataLoader, the transfer happens in background
        # while the GPU processes the PREVIOUS batch.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)         # Reset gradients from previous step
        outputs = model(images)                       # Forward pass: Make a guess
        loss = criterion(outputs, labels)             # Calculate error
        loss.backward()                               # Calculate corrections
        optimizer.step()                              #  Apply corrections

        total_loss += loss.item() * images.size(0)
        # preds holds the output with maximum probability
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# Disable gradient calculation to save memory/speed during validation
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()# Set model to evaluation mode (turns off Dropout/BatchNorm updates)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # non_blocking=True: Tells CPU "Send this data to GPU and move to next line immediately."
        # Because we used pin_memory=True in DataLoader, the transfer happens in background
        # while the GPU processes the PREVIOUS batch.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images) #take a guess
        loss = criterion(outputs, labels) # how wrong the model is compared  to the output

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader,valid_loader,device,phase1_epochs: int = 5,phase2_epochs: int = 10,):
    #  plain CE (sampler already handles imbalance well)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None # Variable to hold the 'brain' (weights) of the best performing model

    # ==========================================
    # PHASE 1: "The Warm-Up" (Train Head Only)
    # ==========================================
    # Goal: Train the random classification layer to catch up with the pre-trained body.
    set_trainable_params(model, phase=1)
    # Optimizer: In Phase 1, we  use a higher Learning Rate (e.g., 1e-3)
    # because the head starts from scratch and needs to learn fast.
    optimizer = get_optimizer(model, phase=1)
    # If the validation loss stops dropping (plateaus) for 2 epochs,
    # cut the learning rate in half (factor=0.5) to find a more precise minimum.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    for epoch in range(phase1_epochs):
        #Train on known data
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        #Test on unseen data
        va_loss, va_acc = evaluate(model, valid_loader, device, criterion)
        #Adjust Learning Rate based on validation loss
        scheduler.step(va_loss)

        print(f"[Phase 1] Epoch {epoch+1}/{phase1_epochs} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.2%} | "
             f"Valid: loss={va_loss:.4f} acc={va_acc:.2%}")
        # 4. Save the "Best So Far"
        # We copy the weights to CPU so we don't clog up GPU memory.
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # ==========================================
    # ---- Phase 2: fine-tune layer4 + FC
    # ==========================================
    #Allow the last block (layer4) to adapt specifically to faces
    set_trainable_params(model, phase=2)
    #  We re-initialize the optimizer, we use a VERY small Learning Rate
    optimizer = get_optimizer(model, phase=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    for epoch in range(phase2_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = evaluate(model, valid_loader, device, criterion)
        scheduler.step(va_loss)

        print(f"[Phase 2] Epoch {epoch+1}/{phase2_epochs} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.2%} | "
              f"Valid: loss={va_loss:.4f} acc={va_acc:.2%}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best ,We overwrite the current model with the best saved weights.
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best valid accuracy: {best_acc:.2%}")
    return model


# ------------------------------------
# Main
# ------------------------------------
def main():
    data_dir = r"C:\Users\yoavt\PycharmProjects\final_projact\data\resnet dataset"
    output_dir = r"C:\Users\yoavt\PycharmProjects\final_projact\models"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, valid_loader, class_to_idx = get_dataloaders(data_dir, batch_size=32, num_workers=2)
    print("Class to index mapping:", class_to_idx)

    model = build_model(num_classes=len(class_to_idx))
    model.to(device)

    model = train_model(
        model,
        train_loader,
        valid_loader,
        device,
        phase1_epochs=16,
        phase2_epochs=9,  # Note: Lower learning rate here!
    )

    # Save weights + mapping
    model_path = os.path.join(output_dir, "resnet18_3.pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model weights to:", model_path)

    mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    print("Saved class mapping to:", mapping_path)


if __name__ == "__main__":
    main()

