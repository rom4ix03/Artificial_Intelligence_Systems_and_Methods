import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def prepare_iris_data():
    """Завантаження та підготовка ознак і міток."""
    raw_iris = load_iris()
    cols = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    
    # Створення DataFrame
    df = pd.DataFrame(raw_iris.data, columns=cols)
    # Мапінг числових цілей у назви сортів
    df['target_class'] = [raw_iris.target_names[i] for i in raw_iris.target]
    
    return df, cols

# 1. Завантаження та спліт даних
iris_df, feature_cols = prepare_iris_data()
features = iris_df[feature_cols].values
labels = iris_df['target_class'].values

x_train, x_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.20, random_state=1
)

# 2. Конфігурація пулу моделей для порівняння
algorithms = {
    'LogReg': LogisticRegression(solver='lbfgs', max_iter=200),
    'LDA':    LinearDiscriminantAnalysis(),
    'KNN':    KNeighborsClassifier(),
    'CART':   DecisionTreeClassifier(),
    'NB':     GaussianNB(),
    'SVM':    SVC(gamma='auto')
}

print("--- Результати крос-валідації (10-fold) ---")
best_results = []

model_names = []

# Оцінка моделей через StratifiedKFold
for alias, algo in algorithms.items():
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_acc = cross_val_score(algo, x_train, y_train, cv=skf, scoring='accuracy')
    
    best_results.append(cv_acc)
    model_names.append(alias)
    
    print(f"{alias:6}: Середня точність {cv_acc.mean()*100:.2f}% (відхилення ±{cv_acc.std()*100:.2f}%)")

# 3. Фінальне навчання обраного класифікатора (SVM)
final_classifier = SVC(gamma='auto')
final_classifier.fit(x_train, y_train)

# Валідація на відкладеній вибірці
val_predictions = final_classifier.predict(x_val)

print("\n" + "="*30)
print("ФІНАЛЬНА ОЦІНКА НА ТЕСТОВИХ ДАНИХ")
print("="*30)
print(f"Accuracy Score: {accuracy_score(y_val, val_predictions):.4f}")
print("\nМатриця помилок:")
print(confusion_matrix(y_val, val_predictions))
print("\nДетальний звіт:")
print(classification_report(y_val, val_predictions))

# 4. Прогноз для невідомого екземпляра
unknown_flower = np.array([[5.0, 2.9, 1.0, 0.2]])
outcome = final_classifier.predict(unknown_flower)

print("\n--- Прогноз для нового об'єкта ---")
print(f"Параметри: {unknown_flower[0]}")
print(f"Результат класифікації: {outcome[0]}")
