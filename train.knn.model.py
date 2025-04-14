# import pandas as pd
# import numpy as np
# from collections import Counter
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import BaggingClassifier
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
# # Štandardizácia dát (pretože kNN je citlivý na škálu vlastností)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
#
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
#     # Rýchly test s jednoduchým kNN
#     knn = KNeighborsClassifier(n_neighbors=5)
#     scores = cross_val_score(knn, X_train_selected, y_train, cv=3, scoring='accuracy')
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
#     # Rýchly test s jednoduchým kNN
#     knn = KNeighborsClassifier(n_neighbors=5)
#     scores = cross_val_score(knn, X_train_pca, y_train, cv=3, scoring='accuracy')
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
# print(f"Počet príznakov po SelectKBest: {best_X_train.shape[1]}")
# print(f"Počet príznakov po PCA: {X_train_pca.shape[1]}")
#
# # Rozšírený grid search pre kNN - optimalizovaný pre hudobnú klasifikáciu
# print("\n🔍 Optimalizujem hyperparametre kNN...")
#
# # Definuj rozšírené parametre pre kNN
# param_grid_knn = {
#     'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],  # rozšírený rozsah susedov
#     'weights': ['uniform', 'distance'],  # rôzne váhy
#     'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],  # rozšírené metriky vzdialenosti
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # rôzne algoritmy vyhľadávania
#     'p': [1, 2, 3]  # parameter pre Minkowski metriku
# }
#
# # Použijeme GridSearchCV na optimalizáciu hyperparametrov s krížovou validáciou
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search_knn = GridSearchCV(
#     KNeighborsClassifier(),
#     param_grid_knn,
#     cv=cv,
#     scoring='accuracy',
#     n_jobs=-1,  # použij všetky dostupné jadrá
#     verbose=1
# )
# grid_search_knn.fit(X_train_pca, y_train)
#
# # Najlepšie parametre
# print("🎯 Najlepšie parametre pre kNN:", grid_search_knn.best_params_)
# print("🎯 Najlepšie skóre počas grid search:", grid_search_knn.best_score_)
#
# # Vytvor ensemble model s najlepším kNN
# best_knn = grid_search_knn.best_estimator_
#
# # Vytvor bagging ensemble s najlepším kNN
# bagging_knn = BaggingClassifier(
#     estimator=best_knn,
#     n_estimators=10,
#     random_state=42,
#     n_jobs=-1
# )
# bagging_knn.fit(X_train_pca, y_train)
#
# # Predikcia a hodnotenie pre základný kNN
# y_pred_knn = best_knn.predict(X_test_pca)
# print("🎯 kNN - Presnosť:", accuracy_score(y_test, y_pred_knn))
# print("\n📊 kNN - Report:\n", classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_))
#
# # Predikcia a hodnotenie pre bagging kNN
# y_pred_bagging = bagging_knn.predict(X_test_pca)
# print("🎯 Bagging kNN - Presnosť:", accuracy_score(y_test, y_pred_bagging))
# print("\n📊 Bagging kNN - Report:\n", classification_report(y_test, y_pred_bagging, target_names=label_encoder.classes_))
#
# # Vytvor a zobraz confusion matrix
# cm = confusion_matrix(y_test, y_pred_bagging)
# print("\n📊 Confusion Matrix:")
# for i, row in enumerate(cm):
#     print(f"{label_encoder.classes_[i]}: {row}")
#
# # Porovnaj s pôvodným kNN modelom
# print("\n🔍 Porovnanie s pôvodným kNN modelom:")
# original_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
# original_knn.fit(X_train_scaled, y_train)
# y_pred_original = original_knn.predict(X_test_scaled)
# print("🎯 Pôvodný kNN - Presnosť:", accuracy_score(y_test, y_pred_original))
# print("\n📊 Pôvodný kNN - Report:\n", classification_report(y_test, y_pred_original, target_names=label_encoder.classes_))
#
# # Porovnaj s najlepším modelom
# print("\n🏆 Najlepší model:")
# if accuracy_score(y_test, y_pred_knn) > accuracy_score(y_test, y_pred_bagging):
#     print("🥇 Základný kNN s optimalizovanými hyperparametrami")
#     best_overall = best_knn
#     best_pred = y_pred_knn
# else:
#     print("🥇 Bagging kNN")
#     best_overall = bagging_knn
#     best_pred = y_pred_bagging
#
# print(f"🎯 Presnosť: {accuracy_score(y_test, best_pred):.4f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
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

# Create the KNN model with the best parameters
best_knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='manhattan',
    algorithm='auto',
    p=1
)

# Train the model
best_knn.fit(X_train_pca, y_train)

# Evaluate the model
y_pred = best_knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 KNN Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))