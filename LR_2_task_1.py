import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Шлях до датасету
DATA_PATH = r'C:\Users\olexi\OneDrive\Рабочий стол\Системи і методи штучного інтелекту\ЛР2\income_data.txt'

def load_and_filter_data(path, limit_per_class=25000):
    features, labels = [], []
    c1, c2 = 0, 0
    
    with open(path, 'r') as file:
        for row in file:
            if c1 >= limit_per_class and c2 >= limit_per_class:
                break
            if '?' in row:
                continue
                
            parts = row.strip().split(', ')
            if len(parts) < 15:
                continue
                
            target = parts[-1]
            if target == '<=50K' and c1 < limit_per_class:
                features.append(parts)
                c1 += 1
            elif target == '>50K' and c2 < limit_per_class:
                features.append(parts)
                c2 += 1
                
    return np.array(features), c1, c2

# Завантажуємо дані
raw_data, n_low, n_high = load_and_filter_data(DATA_PATH)
print(f"Знайдено записів: <=50K — {n_low}, >50K — {n_high}")

# Трансформація текстових даних у цифрові
encoders = []
data_encoded = np.empty(raw_data.shape)

for col_idx in range(raw_data.shape[1]):
    column_sample = raw_data[0, col_idx]
    
    if column_sample.isdigit():
        data_encoded[:, col_idx] = raw_data[:, col_idx]
    else:
        le = preprocessing.LabelEncoder()
        data_encoded[:, col_idx] = le.fit_transform(raw_data[:, col_idx])
        encoders.append(le)

# Відокремлюємо ознаки та цільову змінну
features_matrix = data_encoded[:, :-1].astype(int)
target_vector = data_encoded[:, -1].astype(int)

# Спліт вибірки
x_train, x_test, y_train, y_test = train_test_split(
    features_matrix, target_vector, test_size=0.2, random_state=5
)

# Створення та навчання моделі SVM
svm_model = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=3000))
svm_model.fit(x_train, y_train)

# Отримання прогнозів
predictions = svm_model.predict(x_test)

# Виведення результатів
print("\n--- Аналіз продуктивності моделі ---")
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, average='weighted'),
    "Recall": recall_score(y_test, predictions, average='weighted'),
    "F1 Score": f1_score(y_test, predictions, average='weighted')
}

for name, value in metrics.items():
    print(f"{name:10}: {value*100:.2f}%")

print("\nЗвіт класифікації:")
print(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

# Валідація
f1_cv = cross_val_score(svm_model, features_matrix, target_vector, scoring='f1_weighted', cv=3)
print(f"\nСередній F1 за крос-валідацією: {np.mean(f1_cv)*100:.2f}%")

# Тест на конкретному прикладі
sample_person = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
                 'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
                 '0', '0', '40', 'United-States']

encoded_sample = []
enc_ptr = 0

for val in sample_person:
    if val.isdigit():
        encoded_sample.append(int(val))
    else:
        encoded_sample.append(encoders[enc_ptr].transform([val])[0])
        enc_ptr += 1

# Прогноз для прикладу
final_pred = svm_model.predict([encoded_sample])
human_readable_result = encoders[-1].inverse_transform(final_pred)[0]
print(f"\nПрогноз для тестового запису: {human_readable_result}")
