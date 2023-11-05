import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Use the training set, validation set, and test set from Assignment 3 (Hierarchical Clustering) for this Assignment.
faces_data = fetch_olivetti_faces(shuffle=True, random_state=32)
X, y = faces_data.data, faces_data.target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=32)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=32)

# Normalize
X_train = X_train/255

# Use PCA preserving 99% of the variance to reduce the dataset’s dimensionality
pca = PCA(n_components=0.99, random_state=32)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

# Define an autoencoder with k-fold cross-validation to fine-tune hyperparameters
input_size = X_train_pca.shape[1]
hidden_values = [64, 128, 256]
central_units = [32, 64]
best_model = None
best_score = float('inf')
best_hyperparameters = {}

learning_rates = [0.001, 0.01, 0.1]

kfold = KFold(n_splits=5, shuffle=True, random_state=32)

for units in hidden_values:
    for units_2 in central_units:
        for learning_rate in learning_rates:
            for train_idx, val_idx in kfold.split(X_train_pca):
                X_train_fold, X_val_fold = X_train_pca[train_idx], X_train_pca[val_idx]

                input_img = Input(shape=(input_size,))
                hidden1 = Dense(units, activation='relu', kernel_regularizer='l1')(input_img)
                central_code = Dense(units_2, activation='relu', kernel_regularizer='l1')(hidden1)
                hidden3 = Dense(units, activation='relu', kernel_regularizer='l1')(central_code)
                output_img = Dense(input_size, activation='relu')(hidden3)

                autoencoder = Model(input_img, output_img)
                autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

                autoencoder.fit(X_train_fold, X_train_fold, epochs=500, batch_size=32, verbose=0)

                val_loss = autoencoder.evaluate(X_val_fold, X_val_fold, verbose=1)
                print("model hidden and central units", units, units_2, "learning rate", learning_rate)

                if val_loss < best_score:
                    best_score = val_loss
                    best_model = autoencoder
                    best_hyperparameters = {
                        'hidden_values': units,
                        'central_units': units_2,
                        'learning_rate': learning_rate
                    }

print("Best hyperparameters:", best_hyperparameters)

# Run the best model with the test set and display the original and reconstructed images
decoded_imgs = best_model.predict(X_test_pca)
decoded_imgs_original_space = pca.inverse_transform(decoded_imgs)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original Images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    ax.set_title("Original")

    # Reconstructed Images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_original_space[i].reshape(64, 64), cmap='gray')
    ax.set_title("Reconstructed")

plt.show()
