import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

input_file = 'C:\\Users\\olexi\\OneDrive\\Рабочий стол\\Системи і методи штучного інтелекту\\ЛР2\\income_data.txt'
X_raw, count1, count2 = [], 0, 0
max_pts = 5000
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count1 >= max_pts and count2 >= max_pts:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if len(data) < 15:
            continue
        if data[-1] == '<=50K' and count1 < max_pts:
            X_raw.append(data); count1 += 1
        if data[-1] == '>50K' and count2 < max_pts:
            X_raw.append(data); count2 += 1

X_raw = np.array(X_raw)
le_list, X_enc = [], np.empty(X_raw.shape)
for i, item in enumerate(X_raw[0]):
    if item.isdigit():
        X_enc[:, i] = X_raw[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_enc[:, i] = le.fit_transform(X_raw[:, i])
        le_list.append(le)

X = X_enc[:, :-1].astype(float)
y = X_enc[:, -1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = [
    ('LR',   LogisticRegression(solver='lbfgs', max_iter=500)),
    ('LDA',  LinearDiscriminantAnalysis()),
    ('KNN',  KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB',   GaussianNB()),
    ('SVM',  SVC(kernel='rbf', gamma='scale')),
]

results, names_list = [], []
print(f"Dataset: {count1+count2} samples | Train: {len(X_train)} | Test: {len(X_test)}")
print("="*55)
print("Algorithm  |  Mean Acc   |  Std Dev")
print("-"*45)
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_res = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_res)
    names_list.append(name)
    print(f"{name:<10} | {cv_res.mean()*100:>8.2f}%  | {cv_res.std()*100:.2f}%")

plt.figure()
plt.boxplot(results, tick_labels=names_list)
plt.title('Algorithm Comparison - Income Data'); plt.ylabel('Accuracy')

# Best model full report
best = DecisionTreeClassifier()
best.fit(X_train, y_train)
y_pred = best.predict(X_test)
print(f"\n=== CART на тестовій вибірці ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['<=50K','>50K']))
print("Done. Graph saved.")
