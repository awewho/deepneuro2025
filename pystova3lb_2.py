#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:13:34 2025

@author: arinapustova
"""

import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

# Считываем данные
df = pd.read_csv('data.csv')

# Подготовка данных
answers = df.iloc[:, 4].values
answers = np.where(answers == "Iris-setosa", 1, -1)
signs = df.iloc[:, :4].values

# Преобразование в тензоры
X = torch.FloatTensor(signs)
y = torch.FloatTensor(answers).view(-1, 1)  # Изменяем форму для совместимости

# Визуализация первых двух признаков
plt.figure(figsize=(10, 6))
plt.scatter(signs[answers==1, 0], signs[answers==1, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(signs[answers==-1, 0], signs[answers==-1, 1], color='blue', marker='x', label='Iris-versicolor')
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.title('Визуализация данных ирисов')
plt.legend()
plt.show()

# Создание модели
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.linear = nn.Linear(4, 1)  # 4 входа, 1 выход
        
    def forward(self, x):
        return self.linear(x)

model = IrisClassifier()

# Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
losses = []
epochs = 100
print_freq = epochs // 10

print("Начинаем обучение модели...")
for epoch in range(epochs):
    # Прямой проход
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % print_freq == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# График обучения
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.title('График обучения')
plt.show()

# Определение порога классификации
with torch.no_grad():
    predictions = model(X)
    max_pred = torch.max(predictions).item()
    min_pred = torch.min(predictions).item()
    threshold = (max_pred + min_pred) / 2
    print(f"\nПорог классификации: {threshold:.4f}")

# Функция для интерактивного предсказания
def predict_flower():
    print("\nВведите параметры цветка для классификации (0 для выхода):")
    while True:
        try:
            p1 = float(input("Длина чашелистика (см): "))
            if p1 == 0: break
            
            p2 = float(input("Ширина чашелистика (см): "))
            p3 = float(input("Длина лепестка (см): "))
            p4 = float(input("Ширина лепестка (см): "))
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor([p1, p2, p3, p4])
                prediction = model(input_tensor).item()
                
            print("\nРезультат классификации:")
            if prediction >= threshold:
                print("Iris-setosa (вероятность: {:.2f}%)".format((prediction - min_pred)/(max_pred - min_pred)*100))
            else:
                print("Iris-versicolor (вероятность: {:.2f}%)".format((max_pred - prediction)/(max_pred - min_pred)*100))
            print("="*40)
            
        except ValueError:
            print("Ошибка ввода. Пожалуйста, введите числовые значения.")
        except Exception as e:
            print(f"Произошла ошибка: {e}")

# Запуск интерактивного режима
predict_flower()
print("\nРабота программы завершена.")