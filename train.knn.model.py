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


    # Select top 50 features (or all if less than 50)
    k_50 = min(50, X.shape[1])
    selector_50 = SelectKBest(f_classif, k=k_50)
    X_train_selected_50 = selector_50.fit_transform(X_train_scaled, y_train)
    X_test_selected_50 = selector_50.transform(X_test_scaled)

    # Use all features (k=1000 or all available)
    k_all = min(1000, X.shape[1])  # Effectively all features
    selector_all = SelectKBest(f_classif, k=k_all)
    X_train_selected_all = selector_all.fit_transform(X_train_scaled, y_train)
    X_test_selected_all = selector_all.transform(X_test_scaled)

    # Apply PCA with n_components=0.99 (preserves 99% of variance) for k=50
    pca_50 = PCA(n_components=0.99)
    X_train_pca_50 = pca_50.fit_transform(X_train_selected_50)
    X_test_pca_50 = pca_50.transform(X_test_selected_50)

    # Apply PCA with n_components=0.99 (preserves 99% of variance) for all features
    pca_all = PCA(n_components=0.99)
    X_train_pca_all = pca_all.fit_transform(X_train_selected_all)
    X_test_pca_all = pca_all.transform(X_test_selected_all)

    print(f"Number of features after PCA (k=50): {X_train_pca_50.shape[1]} (from {X_train_selected_50.shape[1]})")
    print(f"Number of features after PCA (all features): {X_train_pca_all.shape[1]} (from {X_train_selected_all.shape[1]})")

    # Create the KNN model with the best parameters
    best_knn = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )

    # Train and evaluate model with k=50 features without PCA
    best_knn_50 = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )
    best_knn_50.fit(X_train_selected_50, y_train)
    y_pred_50 = best_knn_50.predict(X_test_selected_50)
    accuracy_no_pca_50 = accuracy_score(y_test, y_pred_50)
    print(f"🎯 KNN Accuracy (k=50, without PCA): {accuracy_no_pca_50:.4f}")

    # Train and evaluate model with k=50 features with PCA
    best_knn_pca_50 = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )
    best_knn_pca_50.fit(X_train_pca_50, y_train)
    y_pred_pca_50 = best_knn_pca_50.predict(X_test_pca_50)
    accuracy_pca_50 = accuracy_score(y_test, y_pred_pca_50)
    print(f"🎯 KNN Accuracy (k=50, with PCA): {accuracy_pca_50:.4f}")

    # Train and evaluate model with all features without PCA
    best_knn_all = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )
    best_knn_all.fit(X_train_selected_all, y_train)
    y_pred_all = best_knn_all.predict(X_test_selected_all)
    accuracy_no_pca_all = accuracy_score(y_test, y_pred_all)
    print(f"🎯 KNN Accuracy (all features, without PCA): {accuracy_no_pca_all:.4f}")

    # Train and evaluate model with all features with PCA
    best_knn_pca_all = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        algorithm='auto',
        p=1
    )
    best_knn_pca_all.fit(X_train_pca_all, y_train)
    y_pred_pca_all = best_knn_pca_all.predict(X_test_pca_all)
    accuracy_pca_all = accuracy_score(y_test, y_pred_pca_all)
    print(f"🎯 KNN Accuracy (all features, with PCA): {accuracy_pca_all:.4f}")

    print("\n📊 Classification Report (k=50, with PCA):")
    print(classification_report(y_test, y_pred_pca_50, target_names=label_encoder.classes_))

    print("\n📊 Classification Report (all features, with PCA):")
    print(classification_report(y_test, y_pred_pca_all, target_names=label_encoder.classes_))

    return {
        'feature_name': feature_name,
        'accuracy_no_pca_50': accuracy_no_pca_50,
        'accuracy_pca_50': accuracy_pca_50,
        'accuracy_no_pca_all': accuracy_no_pca_all,
        'accuracy_pca_all': accuracy_pca_all,
        'num_features': X.shape[1],
        'num_selected_features_50': k_50,
        'num_selected_features_all': k_all,
        'num_pca_features_50': X_train_pca_50.shape[1],
        'num_pca_features_all': X_train_pca_all.shape[1]
    }

# Get all CSV files from the features directory
feature_files = [f for f in os.listdir('features') if f.endswith('.csv')]
results = []

# Test each feature set
for feature_file in feature_files:
    feature_name = feature_file.replace('_features.csv', '')
    result = test_feature_set(os.path.join('features', feature_file), feature_name)
    results.append(result)

# Sort results by PCA accuracy with k=50
results.sort(key=lambda x: x['accuracy_pca_50'], reverse=True)

# Print results table
print("\n🏆 Feature Sets Ranked by Accuracy:")
print("=" * 180)
print(f"{'Feature Set':<20} | {'Accuracy (k=50, no PCA)':<20} | {'Accuracy (k=50, PCA)':<20} | {'Accuracy (all, no PCA)':<20} | {'Accuracy (all, PCA)':<20} | {'Original Features':<15} | {'Selected Features (k=50)':<20} | {'Selected Features (all)':<20} | {'PCA Features (k=50)':<20} | {'PCA Features (all)':<20}")
print("-" * 180)
for result in results:
    print(f"{result['feature_name']:<20} | {result['accuracy_no_pca_50']:.4f} | {result['accuracy_pca_50']:.4f} | {result['accuracy_no_pca_all']:.4f} | {result['accuracy_pca_all']:.4f} | {result['num_features']:<15} | {result['num_selected_features_50']:<20} | {result['num_selected_features_all']:<20} | {result['num_pca_features_50']:<20} | {result['num_pca_features_all']:<20}")

# Plot results if matplotlib is available
if matplotlib_available:
    plt.figure(figsize=(16, 10))

    # Set width of bars
    barWidth = 0.2

    # Set positions of the bars on X axis
    r1 = np.arange(len(results))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Create grouped bars
    plt.bar(r1, [r['accuracy_no_pca_50'] for r in results], width=barWidth, label='k=50, Without PCA')
    plt.bar(r2, [r['accuracy_pca_50'] for r in results], width=barWidth, label='k=50, With PCA')
    plt.bar(r3, [r['accuracy_no_pca_all'] for r in results], width=barWidth, label='All Features, Without PCA')
    plt.bar(r4, [r['accuracy_pca_all'] for r in results], width=barWidth, label='All Features, With PCA')

    # Add labels and title
    plt.xlabel('Feature Set')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: k=50 vs All Features, With vs Without PCA')
    plt.xticks([r + 1.5 * barWidth for r in range(len(results))], [r['feature_name'] for r in results], rotation=45)
    plt.legend()

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ensure everything fits
    plt.tight_layout()

    # Save the figure
    plt.savefig('feature_accuracy_comparison.png')
    plt.close()
    print("\nResults visualization saved as 'feature_accuracy_comparison.png'")
