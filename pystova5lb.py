#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:36:34 2025

@author: arinapustova
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# Проверка структуры данных
for split in ['train', 'test']:
    for cls in ['butterfly', 'dragonfly', 'fly']:
        path = f'./data/{split}/{cls}'
        if os.path.exists(path):
            print(f"Папка {path}: {len(os.listdir(path))} файлов")
        else:
            print(f"Папка {path} не существует!")

# Устройство
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Преобразования
data_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка данных
try:
    train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                                    transform=data_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/test',
                                                   transform=data_transforms)
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit()

# Проверка классов
class_names = train_dataset.classes
print("Классы:", class_names)

# DataLoader
batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

# Проверка загрузки батча
try:
    inputs, classes = next(iter(train_loader))
    print("Размер батча:", inputs.shape, "Метки:", classes)
except Exception as e:
    print(f"Ошибка при загрузке батча: {e}")
    exit()

# Загрузка AlexNet
net = torchvision.models.alexnet(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

# Замена классификатора
num_classes = 3
new_classifier = net.classifier[:-1]
new_classifier.add_module('fc', nn.Linear(4096, num_classes))
net.classifier = new_classifier
net = net.to(device)

# Функция потерь и оптимизатор
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.classifier.parameters(), lr=0.01)

# Обучение
num_epochs = 5
save_loss = []
t = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = lossFn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_loss.append(loss.item())
        if i % 10 == 0:
            print(f'Эпоха {epoch+1}/{num_epochs}, Шаг {i}, Ошибка: {loss.item():.4f}')

print(f"Время обучения: {time.time() - t:.2f} секунд")

# График потерь
plt.figure()
plt.plot(save_loss)
plt.title('Функция потерь во время обучения')
plt.xlabel('Шаг')
plt.ylabel('Ошибка')
plt.show()

# Оценка
correct_predictions = 0
num_test_samples = len(test_dataset)
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

accuracy = 100 * correct_predictions / num_test_samples
print(f'Точность модели на тестовом наборе: {accuracy:.2f}%')

# Визуализация
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)