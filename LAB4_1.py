import numpy as np
import matplotlib.pyplot as plt

def solve_linear_regression(x_data, y_data):
    """Обчислення коефіцієнтів лінійної регресії методом МНК."""
    # Формування матриці плану (Design Matrix)
    # Додаємо стовпець одиниць для вільного члена (beta0)
    A_matrix = np.vstack([np.ones(len(x_data)), x_data]).T
    
    # Розв'язання системи нормальних рівнянь через псевдоінверсію або lstsq
    # Це еквівалентно методу np.linalg.solve(A.T @ A, A.T @ B)
    coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, y_data, rcond=None)
    
    return coeffs, residuals

# Вхідний набір даних (Варіант 3)
x_vals = np.array([7, 12, 17, 22, 27, 32], dtype=float)
y_vals = np.array([8, 7, 6, 5, 4, 3], dtype=float)

# Отримання параметрів моделі
(b0, b1), res_sum = solve_linear_regression(x_vals, y_vals)

# Розрахунок коефіцієнта детермінації R^2
y_mean_dist = np.sum((y_vals - np.mean(y_vals))**2)
r_squared = 1 - (res_sum[0] / y_mean_dist)

# Виведення результатів аналізу
print(f"{'Параметр':<15} | {'Значення':<10}")
print("-" * 30)
print(f"{'beta0':<15} | {b0:.4f}")
print(f"{'beta1':<15} | {b1:.4f}")
print(f"{'Сума кв. пох.':<15} | {res_sum[0]:.6f}")
print(f"{'R2':<15} | {r_squared:.6f}")
print(f"\nМодель: y = {b0:.4f} + ({b1:.4f}) * x")

# Візуалізація результатів
plt.figure(figsize=(10, 6))

# Побудова лінії регресії
x_range = np.array([x_vals.min() - 2, x_vals.max() + 2])
y_range = b0 + b1 * x_range

plt.plot(x_range, y_range, 'r-', alpha=0.7, lw=2, label=f'Апроксимація: y = {b0:.2f} {b1:.2f}x')
plt.scatter(x_vals, y_vals, c='darkblue', edgecolors='white', s=100, label='Вихідні дані', zorder=3)

# Оформлення графіка
plt.title('Лінійна регресія (МНК) — Аналіз варіанту №3', fontsize=14)
plt.xlabel('Аргумент X', fontsize=12)
plt.ylabel('Функція Y', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True)

plt.tight_layout()
plt.show()
