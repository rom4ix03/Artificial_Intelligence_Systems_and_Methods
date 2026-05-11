import numpy as np
import matplotlib.pyplot as plt

# Вузлові точки
x_nodes = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y_nodes = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# 1. Заповнення матриці Вандермонда (поліном 4-го ступеня)
X_matrix = np.vander(x_nodes, 5, increasing=True)

# 2. Отримання коефіцієнтів інтерполяційного полінома
coeffs = np.linalg.solve(X_matrix, y_nodes)
print('Коефіцієнти a0..a4:', coeffs)

# 3. Визначення функції полінома
def polynomial(x, c):
    return sum(c[i] * x**i for i in range(len(c)))

# 5. Значення у проміжних точках
print(f'P(0.2) = {polynomial(0.2, coeffs):.6f}')
print(f'P(0.5) = {polynomial(0.5, coeffs):.6f}')

# 4. Побудова графіка
x_plot = np.linspace(0.05, 0.75, 500)
y_plot = [polynomial(xi, coeffs) for xi in x_plot]
plt.scatter(x_nodes, y_nodes, color='blue', s=80,
            label='Вузли інтерполяції')
plt.plot(x_plot, y_plot, color='red', linewidth=2,
         label='Поліном 4-го ступеня P(x)')
plt.xlabel('x'); plt.ylabel('y')
plt.title('Інтерполяція. Поліном 4-го ступеня')
plt.legend(); plt.grid(True); plt.show()
