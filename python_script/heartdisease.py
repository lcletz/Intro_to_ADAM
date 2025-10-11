#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Configuration
tf.config.run_functions_eagerly(True)
tf.random.set_seed(42)
np.random.seed(42)

#%% 
# Préparation des données

df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nDataset préparé : Train={len(X_train)}, Test={len(X_test)}")

#%% 
# Architecture

def create_model():
    return keras.Sequential([
        keras.layers.Input(shape=(13,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

#%% 
# Entraînement avec early stopping 

optimizers = {
    'Adam': keras.optimizers.Adam(learning_rate=0.001),
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'Adagrad': keras.optimizers.Adagrad(learning_rate=0.01),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001)
}

results = {}

for name, optimizer in optimizers.items():
    print(f"\n--- {name} ---")
    
    model = create_model()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    results[name] = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'history': history
    }
    
    print(f"  Epochs trained: {len(history.history['loss'])}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

#%% 
# Génération des figures

# Couleurs 
colors = {
    'Adam': '#2E8B57',
    'SGD': '#8A2BE2', 
    'Adagrad': '#32CD32',
    'RMSprop': '#DDA0DD'
}

#%% 
# Fig 1 : Erreurs d'entraînement et de test 

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1 : Erreur d'entraînement
axes[0].set_title('Erreur d\'entraînement - Heart Disease', 
                  fontsize=14, fontweight='bold')
for name, res in results.items():
    train_error = [1 - acc for acc in res['history'].history['accuracy']]
    epochs = range(1, len(train_error) + 1)
    axes[0].plot(epochs, train_error, label=name, 
                color=colors[name], linewidth=2.5)
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Erreur', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2 : Erreur de test
axes[1].set_title('Erreur de test - Heart Disease', 
                  fontsize=14, fontweight='bold')
test_errors = [(1 - res['test_accuracy']) * 100 for res in results.values()]
optimizer_names = list(results.keys())

bars = axes[1].bar(optimizer_names, test_errors,
                   color=[colors[n] for n in optimizer_names], alpha=0.7)
axes[1].set_ylabel('Erreur (%)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

for bar, error in zip(bars, test_errors):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{error:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('heart_errors.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 
# Fig 2 : Pertes d'entraînement et de test 

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1 : Perte d'entraînement
axes[0].set_title('Perte d\'entraînement - Heart Disease', 
                  fontsize=14, fontweight='bold')
for name, res in results.items():
    train_loss = res['history'].history['loss']
    epochs = range(1, len(train_loss) + 1)
    axes[0].plot(epochs, train_loss, label=name, 
                color=colors[name], linewidth=2.5)
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Perte', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2 : Perte de test
axes[1].set_title('Perte de test - Heart Disease', 
                  fontsize=14, fontweight='bold')
test_losses = [res['test_loss'] for res in results.values()]

bars = axes[1].bar(optimizer_names, test_losses,
                  color=[colors[n] for n in optimizer_names], alpha=0.7)
axes[1].set_ylabel('Perte', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

for bar, loss in zip(bars, test_losses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('heart_losses.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 
# Résumé des résultats


print(f"\n{'Optimiseur':<12} {'Test Loss':<12} {'Test Accuracy':<15} {'Erreur (%)'}")
print("-"*80)

for name, res in results.items():
    error = (1 - res['test_accuracy']) * 100
    print(f"{name:<12} {res['test_loss']:<12.4f} {res['test_accuracy']:<15.4f} {error:.2f}")

best = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
print(f"\n Meilleur optimiseur : {best} ({results[best]['test_accuracy']*100:.2f}%)")
