import numpy as np
 
# --- Функція активації (ступінчаста) ---
def activation(v):
    return 1 if v >= 0 else 0
 
# --- Нейрон OR(x1, x2) ---
# Ваги: W1=1, W2=1, W0=-0.5
# Розподільча пряма: x1 + x2 = 0.5
def neuron_OR(x1, x2):
    W1, W2, W0 = 1, 1, -0.5
    v = W1*x1 + W2*x2 + W0
    return activation(v)
 
# --- Нейрон AND(x1, x2) ---
# Ваги: W1=1, W2=1, W0=-1.5
# Розподільча пряма: x1 + x2 = 1.5
def neuron_AND(x1, x2):
    W1, W2, W0 = 1, 1, -1.5
    v = W1*x1 + W2*x2 + W0
    return activation(v)
 
# --- Нейрон XOR через OR і AND ---
# XOR = OR AND NOT(AND)
# Третій нейрон: y1=OR, y2=AND
# Ваги: Wy1=1, Wy2=-1, W0=-0.5
# Розподільча пряма: y1 - y2 = 0.5
def neuron_XOR(x1, x2):
    y1 = neuron_OR(x1, x2)   # вихід OR-нейрона
    y2 = neuron_AND(x1, x2)  # вихід AND-нейрона
    Wy1, Wy2, W0 = 1, -1, -0.5
    v = Wy1*y1 + Wy2*y2 + W0
    return activation(v)
 
# --- Тестування на всіх комбінаціях ---
print('Тестування нейронів OR, AND, XOR:')
print(f'{'x1':>4} {'x2':>4} | {'OR':>4} {'AND':>5} | {'XOR':>4}')
print('-' * 35)
inputs = [(0,0), (0,1), (1,0), (1,1)]
for x1, x2 in inputs:
    or_out  = neuron_OR(x1, x2)
    and_out = neuron_AND(x1, x2)
    xor_out = neuron_XOR(x1, x2)
    print(f'{x1:>4} {x2:>4} | {or_out:>4} {and_out:>5} | {xor_out:>4}')
