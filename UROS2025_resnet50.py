#pip install torch torchvision torchaudio scikit-learn matplotlib pillow seaborn numpy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import random

# === Dataset Path ===
dataset_path = "e:/UROS2025/Cancer Datasets/ALL+"

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === ImageNet Normalization ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# === Transforms ===
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
}

# === Load Dataset ===
full_dataset = datasets.ImageFolder(root=dataset_path, transform=None)


class_names = full_dataset.classes
class_to_idx = full_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# === Split Dataset ===
train_idx, temp_idx = train_test_split(range(len(full_dataset)), test_size=0.3, stratify=full_dataset.targets, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[full_dataset.targets[i] for i in temp_idx], random_state=42)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

# === Apply transform manually ===
def apply_transform_to_subset(subset, transform):
    return [(transform(subset[i][0]), subset[i][1]) for i in range(len(subset))]

# Wrapping in Dataset object for DataLoader compatibility
def to_dataset(data):
    return torch.utils.data.TensorDataset(
        torch.stack([x[0] for x in data]),
        torch.tensor([x[1] for x in data])
    )


train_dataset = to_dataset(apply_transform_to_subset(train_dataset, data_transforms['train']))
val_dataset = to_dataset(apply_transform_to_subset(val_dataset, data_transforms['validation']))
test_dataset = to_dataset(apply_transform_to_subset(test_dataset, data_transforms['test']))

# === Data Loaders ===
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),
    'validation': torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
}

# === Label Smoothing Cross-Entropy Function ===
def label_smoothing_cross_entropy(logits, targets, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    return (confidence * nll_loss + smoothing * smooth_loss).mean()

# === Model ===
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(128, len(class_names))
)
model = model.to(device)

# === Optimizer and Scheduler ===
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# === Training Loop ===
def train_model(model, num_epochs=15, patience=3):
    best_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}\n{"-"*20}')
        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = label_smoothing_cross_entropy(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_counter = 0
                    torch.save(model.state_dict(), 'models/pytorch/resnet_multi_ALL.h5')
                    print(f"New best model saved with loss: {best_loss:.4f}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(torch.load('models/pytorch/resnet_multi_ALL.h5'))
                        return model, train_losses, val_losses, train_accs, val_accs

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    model.load_state_dict(torch.load('models/pytorch/resnet_multi_ALL.h5'))
    return model, train_losses, val_losses, train_accs, val_accs

# === Train or Load Model ===
os.makedirs('models/pytorch', exist_ok=True)
model_path = 'models/pytorch/resnet_multi_ALL.h5'

if os.path.exists(model_path):
    print("Found existing trained model.")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
else:
    print("Training new model...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(model, num_epochs=15, patience=3)

    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.tight_layout()
    plt.show()

# === Evaluation ===
print("\n" + "="*50)
print("Evaluating on Test Set")
print("="*50)

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Classification Report
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.show()

# Binary Metrics (if applicable)
if cm.shape == (2,2):
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if TP + FN else 0
    specificity = TN / (TN + FP) if TN + FP else 0
    precision = TP / (TP + FP) if TP + FP else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"\nDetailed Metrics:")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, np.array(all_probs)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# === Display Random Predictions (3 per class) ===
print(f"\n{'='*50}")
print("Displaying 3 Random Sample Predictions per Class")
print(f"{'='*50}")

selected_paths = []
true_labels = []

# Collect 3 random images per class
for class_name, class_idx in class_to_idx.items():
    class_dir = os.path.join(dataset_path, class_name)
    class_images = [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    chosen = random.sample(class_images, min(3, len(class_images)))
    selected_paths.extend(chosen)
    true_labels.extend([class_idx] * len(chosen))

# Load and preprocess selected images
imgs = [Image.open(p).convert("RGB") for p in selected_paths]
inputs = torch.stack([data_transforms['test'](img).to(device) for img in imgs])

# Run inference
with torch.no_grad():
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

# Plot results in a grid
n_classes = len(class_names)
fig, axs = plt.subplots(n_classes, 3, figsize=(15, 5 * n_classes))
axs = axs.flatten()

for i, (img, pred, true) in enumerate(zip(imgs, preds, true_labels)):
    axs[i].imshow(img)
    axs[i].axis('off')
    axs[i].set_title(
        f"True: {idx_to_class[true]}\nPred: {idx_to_class[pred]}\n"
        f"Conf: {np.max(probs[i])*100:.1f}%"
    )

plt.tight_layout()
plt.suptitle("Sample Predictions", y=1.02, fontsize=16)
plt.show()

