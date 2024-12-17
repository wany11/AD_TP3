# Importer les bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import keras
import sklearn.manifold
from keras import layers
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse

# 1. Chargement et préparation des données Fashion-MNIST
# Charger les données
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalisation des images
#Cette normalisation va permettre d'améliorer la convergence de l’entraînement du modèle.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Vectorisation des images
#Cette étape va simplifier leur manipulation dans les couches dense du modèle.
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))


print("x_train : ", x_train.shape)
print("x_test : ", x_test.shape)

# 2. Création d'un autoencodeur simple
# Taille de l'espace latent
taille_code = 32

# Définir l'encodeur
input_img = keras.Input(shape=(784,))
encoded = layers.Dense(taille_code, activation='relu')(input_img)

# Définir le décodeur
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Créer le modèle autoencodeur
autoencoder = keras.Model(input_img, decoded)

# Créer le modèle encodeur
encoder = keras.Model(input_img, encoded)

# Créer le modèle décodeur
encoded_input = keras.Input(shape=(taille_code,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

print("Autoencoder summary : ")
autoencoder.summary()

# Compiler le modèle
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entraînement
nepochs = 20
history = autoencoder.fit(
    x_train, x_train,
    epochs=nepochs,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Tracer les courbes de loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(nepochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Losses')
plt.legend()
plt.show()

# 3. Analyse des résultats
# Encodage et décodage des images de test
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Visualisation des résultats
n = 5  # Nombre d'images à afficher
plt.figure(figsize=(20, 4))
for i in range(n):
    # Images originales
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Images décodées
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Calcul des MSE et SSIM
mse_values = [mse(x_test[i], decoded_imgs[i]) for i in range(len(x_test))]
ssim_values = [ssim(x_test[i].reshape(28, 28), decoded_imgs[i].reshape(28, 28), data_range=1.0) for i in
               range(len(x_test))]
# Boîte à moustache pour les SSIM
plt.boxplot(ssim_values)
plt.title('Distribution des SSIM')
plt.show()

# 4. Analyse des paramètres : nombre d'époques et taille de l'espace latent
# Répéter l'apprentissage avec différentes valeurs de `taille_code` et `nepochs`

# 5. Autoencodeur profond
# Définir un autoencodeur avec plusieurs couches
input_img = keras.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

# Décodeur
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# Créer et compiler le modèle
autoencoder_deep = keras.Model(input_img, decoded)
autoencoder_deep.compile(optimizer='adam', loss='binary_crossentropy')

# Entraînement
nepochs = 50
history_deep = autoencoder_deep.fit(
    x_train, x_train,
    epochs=nepochs,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Comparaison avec le modèle simple
# Tracer les courbes de loss
loss_deep = history_deep.history['loss']
val_loss_deep = history_deep.history['val_loss']
plt.figure()
plt.plot(range(nepochs), loss_deep, 'bo', label='Training loss (Deep)')
plt.plot(range(nepochs), val_loss_deep, 'b', label='Validation loss (Deep)')
plt.title('Losses (Deep Autoencoder)')
plt.legend()
plt.show()

# Reconstruction des images
decoded_imgs = autoencoder.predict(x_test)

# Visualisation des images originales et reconstruites
n = 10  # Nombre d'images à afficher
plt.figure(figsize=(20, 4))
for i in range(n):
    # Images originales
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    ax.axis('off')

    # Images reconstruites
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstruit")
    ax.axis('off')

plt.show()

mse_values = []
ssim_values = []

for i in range(len(x_test)):
    original = x_test[i].reshape(28, 28)
    reconstructed = decoded_imgs[i].reshape(28, 28)

    mse_values.append(mse(original, reconstructed))
    ssim_values.append(ssim(original, reconstructed))

# Affichage des résultats
print("MSE moyen :", np.mean(mse_values))
print("SSIM moyen :", np.mean(ssim_values))

# Projection des images encodées dans l'espace latent
encoded_imgs = encoder.predict(x_test)

# Réduction en 2D avec t-SNE
tsne = sklearn.manifold.TSNE(n_components=2, random_state=42)
reduced_latent = tsne.fit_transform(encoded_imgs)

# Visualisation
plt.figure(figsize=(8, 8))
plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], c=y_test, cmap='tab10', s=10)
plt.colorbar()
plt.title("Visualisation de l'espace latent")
plt.show()
