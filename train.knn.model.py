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
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Try to import matplotlib for visualization, but continue if not available
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Note: matplotlib is not installed. Visualization will be skipped.")

# Create a function to test a feature set
def test_feature_set(feature_file, feature_name):
    print(f"\n🔍 Testing feature set: {feature_name}")

    # Load data
    # Try to detect the separator automatically
    with open(feature_file, 'r') as f:
        first_line = f.readline().strip()
        if ';' in first_line:
            separator = ';'
        else:
            separator = ','

    df = pd.read_csv(feature_file, sep=separator)

    # Check if 'filename' and 'label' columns exist
    if 'filename' in df.columns and 'label' in df.columns:
        # Separate features and target
        X = df.drop(columns=['filename', 'label'])
        y = df['label']
    else:
        # Assume the last column is the label and the second-to-last is the filename
        print(f"Warning: 'filename' or 'label' columns not found in {feature_file}.")
        print("Assuming the last column is 'label' and the second-to-last is 'filename'.")

        # Get the last two column names
        cols = df.columns.tolist()
        filename_col = cols[-2]
        label_col = cols[-1]

        # Separate features and target
        X = df.drop(columns=[filename_col, label_col])
        y = df[label_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    k = min(50, X.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Apply PCA with n_components=0.99 (preserves 99% of variance)
    pca = PCA(n_components=0.99)
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    print(f"Number of features after PCA: {X_train_pca.shape[1]} (from {X_train_selected.shape[1]})")

    # Create the KNN model with the best parameters
    best_knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )

    # Train and evaluate model without PCA
    best_knn.fit(X_train_selected, y_train)
    y_pred = best_knn.predict(X_test_selected)
    accuracy_no_pca = accuracy_score(y_test, y_pred)
    print(f"🎯 KNN Accuracy (without PCA): {accuracy_no_pca:.4f}")

    # Train and evaluate model with PCA
    best_knn_pca = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )
    best_knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = best_knn_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    print(f"🎯 KNN Accuracy (with PCA): {accuracy_pca:.4f}")

    print("\n📊 Classification Report (with PCA):")
    print(classification_report(y_test, y_pred_pca, target_names=label_encoder.classes_))

    return {
        'feature_name': feature_name,
        'accuracy_no_pca': accuracy_no_pca,
        'accuracy_pca': accuracy_pca,
        'num_features': X.shape[1],
        'num_selected_features': k,
        'num_pca_features': X_train_pca.shape[1]
    }

# Get all CSV files from the features directory
feature_files = [f for f in os.listdir('features') if f.endswith('.csv')]
results = []

# Test each feature set
for feature_file in feature_files:
    feature_name = feature_file.replace('_features.csv', '')
    result = test_feature_set(os.path.join('features', feature_file), feature_name)
    results.append(result)

# Sort results by PCA accuracy
results.sort(key=lambda x: x['accuracy_pca'], reverse=True)

# Print results table
print("\n🏆 Feature Sets Ranked by Accuracy (with PCA):")
print("=" * 120)
print(f"{'Feature Set':<20} | {'Accuracy (no PCA)':<15} | {'Accuracy (with PCA)':<15} | {'Original Features':<15} | {'Selected Features':<15} | {'PCA Features':<15}")
print("-" * 120)
for result in results:
    print(f"{result['feature_name']:<20} | {result['accuracy_no_pca']:.4f} | {result['accuracy_pca']:.4f} | {result['num_features']:<15} | {result['num_selected_features']:<15} | {result['num_pca_features']:<15}")

# Plot results if matplotlib is available
if matplotlib_available:
    plt.figure(figsize=(12, 8))

    # Set width of bars
    barWidth = 0.35

    # Set positions of the bars on X axis
    r1 = np.arange(len(results))
    r2 = [x + barWidth for x in r1]

    # Create grouped bars
    plt.bar(r1, [r['accuracy_no_pca'] for r in results], width=barWidth, label='Without PCA')
    plt.bar(r2, [r['accuracy_pca'] for r in results], width=barWidth, label='With PCA')

    # Add labels and title
    plt.xlabel('Feature Set')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: With vs Without PCA')
    plt.xticks([r + barWidth/2 for r in range(len(results))], [r['feature_name'] for r in results], rotation=45)
    plt.legend()

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ensure everything fits
    plt.tight_layout()

    # Save the figure
    plt.savefig('feature_accuracy_comparison.png')
    plt.close()
    print("\nResults visualization saved as 'feature_accuracy_comparison.png'")
