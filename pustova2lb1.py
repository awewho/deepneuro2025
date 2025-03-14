#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:00:01 2025

@author: arinapustova
"""
from random import randint

list_of_randint = []

min = int(input("Введите нижнюю границу: "))
max = int(input("Введите верхнюю границу: "))
n = int(input("Введите количество элементов: "))

for i in range(n):
    list_of_randint.append(randint(min, max))
    
print(list_of_randint)

sum = 0
for i in list_of_randint:
    sum += i * (i%2 == 0)

print("Сумма элементов списка: "+ str(sum))

