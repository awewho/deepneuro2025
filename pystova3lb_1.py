#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 11:39:21 2025

@author: arinapustova
"""

import torch

# 1. Создаем тензор x целочисленного типа, хранящий случайное значение
x = torch.randint(1, 10, (1,), dtype=torch.int32)
print("Исходный тензор x:", x)

# 2. Преобразуем тензор к типу float32
x = x.to(dtype=torch.float32)
print("Тензор x после преобразования к float32:", x)

# 3. Проводим операции с тензором x
# Мой вариант в  ЭИОС - 10
n = 3  

# Возведение в степень n
x_pow = x ** n
print("x в степени", n, ":", x_pow)

# Умножение на случайное значение в диапазоне от 1 до 10
random_value = torch.randint(1, 10, (1,)).float()
x_mul = x_pow * random_value
print("x после умножения на случайное значение:", x_mul)

# Взятие экспоненты от полученного числа
x_exp = torch.exp(x_mul)
print("Экспонента от x:", x_exp)

# 4. Вычисление и вывод на экран значения производной
# Устанавливаем requires_grad=True для x, чтобы отслеживать градиенты
x.requires_grad_(True)

# Выполняем те же операции с тензором x, чтобы вычислить градиент
x_pow = x ** n
x_mul = x_pow * random_value
x_exp = torch.exp(x_mul)

# Вычисляем градиент
x_exp.backward()

# Выводим значение производной
print("Производная по x:", x.grad)