import numpy as np
import matplotlib.pyplot as plt

class QuadraticRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coeffs = None
        self.r2 = None
        self.ss_res = None

    def fit(self):
        # Формування матриці Вандермонда для другого ступеня (1, x, x^2)
        # Це автоматизує обчислення всіх Σx, Σx2, Σx3, Σx4
        A_matrix = np.vander(self.x, 3, increasing=True)
        
        # Розв'язання системи нормальних рівнянь: (A.T @ A) @ beta = A.T @ y
        # Це ідентично вашому ручному наповненню матриці 3x3
        self.coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, self.y, rcond=None)
        
        # Розрахунок метрик якості
        y_pred = self.predict(self.x)
        self.ss_res = np.sum((self.y - y_pred)**2)
        ss_tot = np.sum((self.y - np.mean(self.y))**2)
        self.r2 = 1 - (self.ss_res / ss_tot)
        
        return self.coeffs

    def predict(self, x_val):
        b0, b1, b2 = self.coeffs
        return b0 + b1 * x_val + b2 * x_val**2

# --- Підготовка даних (Варіант 3) ---
x_data = np.array([7, 12, 17, 22, 27, 32], dtype=float)
y_data = np.array([8, 7, 6, 5, 4, 3], dtype=float)

# Створення та навчання моделі
model = QuadraticRegression(x_data, y_data)
beta = model.fit()

# --- Виведення звіту ---
print("=" * 45)
print(f"{'Параметр':<20} | {'Значення':<15}")
print("-" * 45)
for i, b in enumerate(beta):
    print(f"beta_{i:<14} | {b:.8f}")
print("-" * 45)
print(f"{'R-squared':<20} | {model.r2:.10f}")
print(f"{'Sum of Sq. Error':<20} | {model.ss_res:.10f}")
print("=" * 45)

# --- Візуалізація ---
plt.figure(figsize=(10, 6), facecolor='#f0f0f0')

# Генерація точок для гладкої кривої
x_smooth = np.linspace(x_data.min() - 3, x_data.max() + 3, 500)
y_smooth = model.predict(x_smooth)

# Побудова графіків
plt.plot(x_smooth, y_smooth, color='darkgreen', label='Параболічна модель (МНК)', lw=2.5, zorder=2)
plt.scatter(x_data, y_data, color='crimson', s=120, edgecolors='black', label='Експеримент (Вар. 3)', zorder=3)

# Текстовий блок з рівнянням на графіку
eq_text = f'$y = {beta[0]:.2f} + ({beta[1]:.2f})x + ({beta[2]:.6f})x^2$'
plt.gca().text(0.05, 0.1, eq_text, transform=plt.gca().transAxes, 
               fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Налаштування осей та сітки
plt.title('Апроксимація даних поліномом 2-го ступеня', fontsize=14, fontweight='bold')
plt.xlabel('Параметр X', fontsize=12)
plt.ylabel('Результат Y', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', shadow=True)

plt.tight_layout()
plt.show()
