import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Шлях до текстової бази даних
DATA_FILE = r'C:\Users\olexi\OneDrive\Рабочий стол\Системи і методи штучного інтелекту\ЛР2\income_data.txt'

def get_balanced_subset(path, limit_per_label=3000):
    """Завантажує збалансовану підвибірку даних без пропущених значень."""
    collected_data = []
    class_stats = {'<=50K': 0, '>50K': 0}
    
    with open(path, 'r') as stream:
        for row in stream:
            if all(v >= limit_per_label for v in class_stats.values()):
                break
            if '?' in row:
                continue
                
            tokens = row.strip().split(', ')
            if len(tokens) < 15:
                continue
                
            label = tokens[-1]
            if label in class_stats and class_stats[label] < limit_per_label:
                collected_data.append(tokens)
                class_stats[label] += 1
                
    return np.array(collected_data), class_stats

# 1. Підготовка даних
raw_matrix, stats = get_balanced_subset(DATA_FILE)
print(f"Завантажено: {stats['<=50K']} (клас <=50K) та {stats['>50K']} (клас >50K)")

# 2. Перетворення категорій у числа
encoders_list = []
encoded_data = np.empty(raw_matrix.shape)

for col_idx in range(raw_matrix.shape[1]):
    val_sample = raw_matrix[0, col_idx]
    
    if val_sample.isdigit():
        encoded_data[:, col_idx] = raw_matrix[:, col_idx]
    else:
        l_enc = preprocessing.LabelEncoder()
        encoded_data[:, col_idx] = l_enc.fit_transform(raw_matrix[:, col_idx])
        encoders_list.append(l_enc)

# Формування матриці ознак та вектора відповідей
X_vals = encoded_data[:, :-1].astype(int)
y_vals = encoded_data[:, -1].astype(int)

# Розподіл на навчання та тест (80/20)
train_x, test_x, train_y, test_y = train_test_split(
    X_vals, y_vals, test_size=0.2, random_state=5
)

# 3. Конфігурації нелінійних моделей SVM
svm_variants = [
    {'title': 'Polynomial (degree 3)', 'model': SVC(kernel='poly', degree=3, random_state=0)},
    {'title': 'Gaussian RBF',        'model': SVC(kernel='rbf', gamma='scale', random_state=0)},
    {'title': 'Sigmoid',             'model': SVC(kernel='sigmoid', gamma='scale', random_state=0)}
]

print("-" * 60)
print(f"{'Ядро моделі':<25} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
print("-" * 60)

# 4. Процес навчання та валідації
for variant in svm_variants:
    core_name = variant['title']
    clf = variant['model']
    
    # Навчання та прогноз
    clf.fit(train_x, train_y)
    y_predictions = clf.predict(test_x)
    
    # Розрахунок метрик
    a = accuracy_score(test_y, y_predictions)
    p = precision_score(test_y, y_predictions, average='weighted')
    r = recall_score(test_y, y_predictions, average='weighted')
    f = f1_score(test_y, y_predictions, average='weighted')
    
    # Виведення рядка результатів
    print(f"{core_name:<25} | {a*100:>6.2f}% | {p*100:>6.2f}% | {r*100:>6.2f}% | {f*100:>6.2f}%")
