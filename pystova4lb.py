#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:36:34 2025

@author: arinapustova
"""

import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('dataset_simple.csv')
X = torch.FloatTensor(df.iloc[:, [0]].values)  # Возраст как признак
y = torch.FloatTensor(df.iloc[:, [1]].values)  # Доход как целевая переменная

# Визуализация данных
plt.figure(figsize=(10, 6))
plt.scatter(X.numpy(), y.numpy(), alpha=0.6)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Зависимость дохода от возраста')
plt.grid(True)
plt.show()

# Нейронная сеть для регрессии
class IncomePredictor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(IncomePredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры сети
input_size = 1    # Один входной признак - возраст
hidden_size = 10  # Увеличил количество нейронов для лучшей аппроксимации
output_size = 1   # Один выход - прогнозируемый доход

# Создание модели
model = IncomePredictor(input_size, hidden_size, output_size)

# Функция потерь и оптимизатор
criterion = nn.MSELoss()  # Среднеквадратичная ошибка для регрессии
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam работает лучше для этой задачи

# Обучение модели
epochs = 500
losses = []

print("Начинаем обучение модели...")
for epoch in range(epochs):
    # Прямой проход
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# График процесса обучения
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Эпоха')
plt.ylabel('Ошибка (MSE)')
plt.title('График обучения')
plt.grid(True)
plt.show()

# Предсказания на обучающих данных
with torch.no_grad():
    predicted = model(X)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(X.numpy(), y.numpy(), label='Реальные данные', alpha=0.6)
plt.scatter(X.numpy(), predicted.numpy(), color='red', label='Предсказания', alpha=0.6)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Предсказание дохода по возрасту')
plt.legend()
plt.grid(True)
plt.show()

# Оценка модели
mae = torch.mean(torch.abs(predicted - y))
print(f"\nСредняя абсолютная ошибка (MAE): {mae.item():.2f}")

# Интерактивный режим предсказания
def predict_income():
    print("\nВведите возраст для предсказания дохода (0 для выхода):")
    while True:
        try:
            age = float(input("Возраст: "))
            if age == 0:
                break
                
            with torch.no_grad():
                income = model(torch.FloatTensor([[age]]))
                print(f"Предсказанный доход: {income.item():.2f} рублей\n")
                
        except ValueError:
            print("Ошибка ввода. Пожалуйста, введите числовое значение.")

# Запуск интерактивного режима
predict_income()
print("\nРабота программы завершена.")