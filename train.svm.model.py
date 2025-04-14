# import pandas as pd
# import numpy as np
# from collections import Counter
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import VotingClassifier, BaggingClassifier
# from sklearn.multiclass import OneVsRestClassifier
#
# # Načítaj dáta
# df = pd.read_csv('features_s_labelmi.csv')
#
# # Vstupné dáta (features) a cieľový label
# X = df.drop(columns=['filename', 'label'])
# y = df['label']
#
# # Zakóduj labely na čísla (napr. disco -> 0, blues -> 1, ...)
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Rozdeľ na tréningovú a testovaciu množinu
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
#
# # Štandardizácia dát (pretože SVM je citlivý na škálu vlastností)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Analyzuj distribúciu tried
# class_counts = Counter(y_train)
# print("Distribúcia tried v trénovacích dátach:")
# for label, count in sorted(class_counts.items()):
#     print(f"  {label_encoder.classes_[label]}: {count}")
#
# # Vyskúšaj rôzne metódy feature selection
# print("\n🔍 Skúšam rôzne metódy feature selection...")
# feature_selectors = {
#     'f_classif': SelectKBest(f_classif, k=50),
#     'mutual_info': SelectKBest(mutual_info_classif, k=50)
# }
#
# best_score = 0
# best_selector = None
# best_X_train = None
# best_X_test = None
#
# for name, selector in feature_selectors.items():
#     X_train_selected = selector.fit_transform(X_train_scaled, y_train)
#     X_test_selected = selector.transform(X_test_scaled)
#
#     # Rýchly test s jednoduchým SVM
#     svm = SVC(kernel='linear', C=1, class_weight='balanced')
#     scores = cross_val_score(svm, X_train_selected, y_train, cv=3, scoring='accuracy')
#     avg_score = np.mean(scores)
#     print(f"  {name}: Priemerné skóre = {avg_score:.4f}")
#
#     if avg_score > best_score:
#         best_score = avg_score
#         best_selector = selector
#         best_X_train = X_train_selected
#         best_X_test = X_test_selected
#
# print(f"🔍 Najlepšia metóda feature selection: {best_score:.4f}")
#
# # Aplikuj PCA pre redukciu dimenzionality
# print("\n🔍 Skúšam rôzne konfigurácie PCA...")
# pca_configs = [0.9, 0.95, 0.99]
# best_pca_score = 0
# best_pca = None
# best_X_train_pca = None
# best_X_test_pca = None
#
# for n_components in pca_configs:
#     pca = PCA(n_components=n_components)
#     X_train_pca = pca.fit_transform(best_X_train)
#     X_test_pca = pca.transform(best_X_test)
#
#     # Rýchly test s jednoduchým SVM
#     svm = SVC(kernel='linear', C=1, class_weight='balanced')
#     scores = cross_val_score(svm, X_train_pca, y_train, cv=3, scoring='accuracy')
#     avg_score = np.mean(scores)
#     print(f"  PCA(n_components={n_components}): Priemerné skóre = {avg_score:.4f}, Počet príznakov: {X_train_pca.shape[1]}")
#
#     if avg_score > best_pca_score:
#         best_pca_score = avg_score
#         best_pca = pca
#         best_X_train_pca = X_train_pca
#         best_X_test_pca = X_test_pca
#
# print(f"🔍 Najlepšia konfigurácia PCA: {best_pca_score:.4f} s {best_X_train_pca.shape[1]} príznakmi")
#
# # Použij najlepšie transformované dáta
# X_train_pca = best_X_train_pca
# X_test_pca = best_X_test_pca
#
# print(f"Pôvodný počet príznakov: {X_train_scaled.shape[1]}")
# print(f"Počet príznakov po SelectKBest: {X_train_selected.shape[1]}")
# print(f"Počet príznakov po PCA: {X_train_pca.shape[1]}")
#
# # Rozšírený grid search pre SVM - optimalizovaný pre hudobnú klasifikáciu
# print("\n🔍 Optimalizujem hyperparametre SVM...")
#
# # Najprv zistíme, ktorý kernel je najsľubnejší
# kernel_test = {
#     'kernel': ['linear', 'rbf', 'poly'],
#     'C': [1],
#     'class_weight': ['balanced']
# }
#
# kernel_search = GridSearchCV(
#     SVC(probability=True, random_state=42),
#     kernel_test,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=-1
# )
# kernel_search.fit(X_train_pca, y_train)
# best_kernel = kernel_search.best_params_['kernel']
# print(f"🔍 Najsľubnejší kernel: {best_kernel}")
#
# # Teraz detailnejšie prehľadáme hyperparametre pre najlepší kernel
# if best_kernel == 'linear':
#     param_grid_svm = {
#         'kernel': ['linear'],
#         'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
#         'class_weight': ['balanced', None],
#         'decision_function_shape': ['ovo', 'ovr']
#     }
# elif best_kernel == 'rbf':
#     param_grid_svm = {
#         'kernel': ['rbf'],
#         'C': [0.1, 0.5, 1, 5, 10, 50, 100],
#         'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
#         'class_weight': ['balanced', None],
#         'decision_function_shape': ['ovo', 'ovr']
#     }
# elif best_kernel == 'poly':
#     param_grid_svm = {
#         'kernel': ['poly'],
#         'C': [0.1, 1, 10, 100],
#         'gamma': ['scale', 'auto', 0.01, 0.1, 1],
#         'degree': [2, 3, 4, 5],
#         'coef0': [0, 0.1, 1],
#         'class_weight': ['balanced', None],
#         'decision_function_shape': ['ovo', 'ovr']
#     }
# else:
#     param_grid_svm = {
#         'kernel': ['sigmoid'],
#         'C': [0.1, 1, 10, 100],
#         'gamma': ['scale', 'auto', 0.01, 0.1, 1],
#         'class_weight': ['balanced', None],
#         'decision_function_shape': ['ovo', 'ovr']
#     }
#
# # Použijeme GridSearchCV na optimalizáciu hyperparametrov s krížovou validáciou
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search_svm = GridSearchCV(
#     SVC(probability=True, random_state=42),
#     param_grid_svm,
#     cv=cv,
#     scoring='accuracy',
#     n_jobs=-1,  # použij všetky dostupné jadrá
#     verbose=1
# )
# grid_search_svm.fit(X_train_pca, y_train)
#
# # Najlepšie parametre
# print("🎯 Najlepšie parametre pre SVM:", grid_search_svm.best_params_)
# print("🎯 Najlepšie skóre počas grid search:", grid_search_svm.best_score_)
#
# # Vytvor ensemble model s najlepším SVM
# best_svm = grid_search_svm.best_estimator_
#
# # Vytvor bagging ensemble s najlepším SVM
# # V novších verziách scikit-learn sa parameter 'base_estimator' zmenil na 'estimator'
# bagging_svm = BaggingClassifier(
#     estimator=best_svm,
#     n_estimators=10,
#     random_state=42,
#     n_jobs=-1
# )
# bagging_svm.fit(X_train_pca, y_train)
#
# # Predikcia a hodnotenie pre základný SVM
# y_pred_svm = best_svm.predict(X_test_pca)
# print("🎯 SVM - Presnosť:", accuracy_score(y_test, y_pred_svm))
# print("\n📊 SVM - Report:\n", classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
#
# # Predikcia a hodnotenie pre bagging SVM
# y_pred_bagging = bagging_svm.predict(X_test_pca)
# print("🎯 Bagging SVM - Presnosť:", accuracy_score(y_test, y_pred_bagging))
# print("\n📊 Bagging SVM - Report:\n", classification_report(y_test, y_pred_bagging, target_names=label_encoder.classes_))
#
# # Vytvor a zobraz confusion matrix
# cm = confusion_matrix(y_test, y_pred_bagging)
# print("\n📊 Confusion Matrix:")
# for i, row in enumerate(cm):
#     print(f"{label_encoder.classes_[i]}: {row}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('features_s_labelmi.csv')

# Separate features and target
X = df.drop(columns=['filename', 'label'])
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply the best feature selection method (f_classif with k=50)
selector = SelectKBest(f_classif, k=50)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Apply PCA with n_components=0.99 (resulted in 37 features)
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train_selected)
X_test_pca = pca.transform(X_test_selected)

# Create the SVM model with the best parameters
best_svm = SVC(
    kernel='rbf',
    C=5,
    gamma='auto',
    class_weight=None,
    decision_function_shape='ovo',
    probability=True,
    random_state=42
)

# Train the model
best_svm.fit(X_train_pca, y_train)

# Evaluate the model
y_pred = best_svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 SVM Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))