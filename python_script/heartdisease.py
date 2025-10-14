#%%
# Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Configuration de l'environnement
tf.config.run_functions_eagerly(True)  # Exécution immédiate pour faciliter le débogage
tf.random.set_seed(42)  # Reproductibilité des résultats TensorFlow
np.random.set_seed(42)  # Reproductibilité des résultats NumPy

#%% 
# Préparation des données

df = pd.read_csv('heart.csv')  # Chargement du dataset Heart Disease

# Séparation features/cible
X = df.drop('target', axis=1).values  # 13 caractéristiques cliniques
y = df['target'].values  # Variable binaire (0=sain, 1=maladie)

# Division train/test avec stratification pour garder la proportion des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalisation pour améliorer la convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit et transform sur train
X_test = scaler.transform(X_test)  # Transform uniquement sur test

print(f"\nDataset préparé : Train={len(X_train)}, Test={len(X_test)}")

#%% 
# Définition de l'architecture du réseau

def create_model():
    """
    Crée un réseau de neurones feedforward pour la classification binaire.
    
    Architecture :
    - Input layer : 13 features
    - Hidden layer 1 : 16 neurones avec activation ReLU
    - Hidden layer 2 : 8 neurones avec activation ReLU
    - Output layer : 1 neurone avec activation sigmoid (probabilité entre 0 et 1)
    
    Returns:
        keras.Sequential : Modèle non compilé
    """
    return keras.Sequential([
        keras.layers.Input(shape=(13,)),  # Couche d'entrée
        keras.layers.Dense(16, activation='relu'),  # Première couche cachée
        keras.layers.Dense(8, activation='relu'),  # Deuxième couche cachée
        keras.layers.Dense(1, activation='sigmoid')  # Couche de sortie
    ])

#%% 
# Entraînement avec différents optimiseurs

# Learning rates standards selon Ruder (2016)
optimizers = {
    'Adam': keras.optimizers.Adam(learning_rate=0.001),  # Méthode adaptative
    'SGD': keras.optimizers.SGD(learning_rate=0.01),  # Descente de gradient classique
    'Adagrad': keras.optimizers.Adagrad(learning_rate=0.01),  # Adaptation par paramètre
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001)  # Moyenne mobile des gradients
}

results = {}  # Stockage des résultats

# Boucle d'entraînement pour chaque optimiseur
for name, optimizer in optimizers.items():
    print(f"\n--- {name} ---")
    
    model = create_model()  # Nouveau modèle pour chaque optimiseur
    
    # Compilation avec fonction de perte et métrique
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Loss pour classification binaire
        metrics=['accuracy']  # Métrique de performance
    )
    
    # Early stopping pour éviter le surapprentissage
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Surveille la perte de validation
        patience=10,  # Arrêt après 10 epochs sans amélioration
        restore_best_weights=True,  # Restaure les meilleurs poids
        verbose=0
    )
    
    # Entraînement du modèle
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Maximum d'epochs
        batch_size=32,  # 32 exemples par batch
        validation_split=0.2,  # 20% pour validation
        callbacks=[early_stop],  # Active l'early stopping
        verbose=0
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  # Évaluation finale
    
    # Sauvegarde des résultats
    results[name] = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'history': history
    }
    
    print(f"  Epochs trained: {len(history.history['loss'])}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

#%% 
# Configuration des couleurs pour les graphiques

colors = {
    'Adam': '#2E8B57',  
    'SGD': '#8A2BE2',  
    'Adagrad': '#32CD32',  
    'RMSprop': '#DDA0DD'  
}

#%% 
# Figure 1 : comparaison des erreurs d'entraînement et de test

fig, axes = plt.subplots(1, 2, figsize=(14, 6))  

# Subplot 1 : évolution de l'erreur d'entraînement
axes[0].set_title('Erreur d\'entraînement - Heart Disease', 
                  fontsize=14, fontweight='bold')

for name, res in results.items():
    train_error = [1 - acc for acc in res['history'].history['accuracy']]  # Erreur = 1 - accuracy
    epochs = range(1, len(train_error) + 1)
    axes[0].plot(epochs, train_error, label=name, 
                color=colors[name], linewidth=2.5)  # Tracé de la courbe

axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Erreur', fontsize=12)
axes[0].legend(fontsize=11)  
axes[0].grid(True, alpha=0.3)  

# Subplot 2 : erreur de test (barplot)
axes[1].set_title('Erreur de test - Heart Disease', 
                  fontsize=14, fontweight='bold')

test_errors = [(1 - res['test_accuracy']) * 100 for res in results.values()]  # Erreur en %
optimizer_names = list(results.keys())

bars = axes[1].bar(optimizer_names, test_errors,
                   color=[colors[n] for n in optimizer_names], alpha=0.7)  
axes[1].set_ylabel('Erreur (%)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

# Affichage des valeurs sur les barres
for bar, error in zip(bars, test_errors):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{error:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)

plt.tight_layout()  # Ajustement automatique de l'espacement
plt.savefig('heart_errors.png', dpi=300, bbox_inches='tight')  
plt.show()

#%% 
# Figure 2 : comparaison des pertes d'entraînement et de test

fig, axes = plt.subplots(1, 2, figsize=(14, 6))  

# Subplot 1 : évolution de la perte d'entraînement
axes[0].set_title('Perte d\'entraînement - Heart Disease', 
                  fontsize=14, fontweight='bold')

for name, res in results.items():
    train_loss = res['history'].history['loss']  # Extraction de la loss
    epochs = range(1, len(train_loss) + 1)
    axes[0].plot(epochs, train_loss, label=name, 
                color=colors[name], linewidth=2.5)  # Tracé de la courbe

axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Perte', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2 : perte de test (barplot)
axes[1].set_title('Perte de test - Heart Disease', 
                  fontsize=14, fontweight='bold')

test_losses = [res['test_loss'] for res in results.values()]  # Extraction des pertes de test

bars = axes[1].bar(optimizer_names, test_losses,
                  color=[colors[n] for n in optimizer_names], alpha=0.7)  
axes[1].set_ylabel('Perte', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

# Affichage des valeurs sur les barres
for bar, loss in zip(bars, test_losses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)

plt.tight_layout()  # Ajustement automatique
plt.savefig('heart_losses.png', dpi=300, bbox_inches='tight')  
plt.show()

#%% 
# Résumé des résultats sous forme de tableau

print(f"\n{'Optimiseur':<12} {'Test Loss':<12} {'Test Accuracy':<15} {'Erreur (%)'}")
print("-"*80)

for name, res in results.items():
    error = (1 - res['test_accuracy']) * 100  # Calcul de l'erreur en %
    print(f"{name:<12} {res['test_loss']:<12.4f} {res['test_accuracy']:<15.4f} {error:.2f}")

# Identification du meilleur optimiseur (celui avec la meilleure accuracy)
best = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
print(f"\n Meilleur optimiseur : {best} ({results[best]['test_accuracy']*100:.2f}%)")