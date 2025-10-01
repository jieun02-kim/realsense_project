import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder("dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=32)

# MobileNetV2 로드
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2클래스 (gown/normal)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프 (간단 버전)
for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")

torch.save(model.state_dict(), "gown_classifier.pth")