import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, matthews_corrcoef, 
                             classification_report, confusion_matrix)

# Налаштування бекенду для збереження файлів без відображення вікна
import matplotlib
matplotlib.use('Agg')

# 1. Підготовка даних
iris_data = load_iris()
features, targets = iris_data.data, iris_data.target

# Розподіл на тренувальну та тестову вибірки (70/30)
x_train, x_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.3, random_state=0
)

# 2. Налаштування та навчання Ridge-класифікатора
# Використовуємо стохастичний градієнтний спуск (sag) та точність 1e-2
ridge_model = RidgeClassifier(tol=1e-2, solver="sag")
ridge_model.fit(x_train, y_train)

# Отримання прогнозів
y_predictions = ridge_model.predict(x_test)

# 3. Розрахунок та виведення метрик
performance_metrics = {
    'Accuracy': accuracy_score(y_test, y_predictions),
    'Precision': precision_score(y_test, y_predictions, average='weighted'),
    'Recall': recall_score(y_test, y_predictions, average='weighted'),
    'F1 Score': f1_score(y_test, y_predictions, average='weighted'),
    'Cohen Kappa': cohen_kappa_score(y_test, y_predictions),
    'Matthews Corrcoef': matthews_corrcoef(y_test, y_predictions)
}

print("=== Результати RidgeClassifier ===")
for label, value in performance_metrics.items():
    print(f"{label:18}: {np.round(value, 4)}")

print('\nЗвіт за класами:')
print(classification_report(y_test, y_predictions, target_names=iris_data.target_names))

# 4. Візуалізація матриці помилок
conf_mat = confusion_matrix(y_test, y_predictions)

plt.figure(figsize=(7, 6))
sns.heatmap(
    conf_mat, 
    annot=True, 
    fmt='d', 
    cmap='Greens', # Змінив колір на зелений для відмінності
    cbar=False,
    xticklabels=iris_data.target_names, 
    yticklabels=iris_data.target_names,
    square=True
)

plt.title('Матриця помилок (Ridge Classifier)')
plt.xlabel('Передбачені мітки')
plt.ylabel('Справжні мітки')

# Шлях збереження (залиште свій або використовуйте відносний)
output_path = r'C:\Users\olexi\OneDrive\Рабочий стол\Системи і методи штучного інтелекту\ЛР2\Confusion_Ridge.jpg'
plt.tight_layout()
plt.savefig(output_path, dpi=120)
plt.close()

print(f"\nГрафік успішно збережено за шляхом: {output_path}")
