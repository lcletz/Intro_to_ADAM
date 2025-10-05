#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras


tf.random.set_seed(42)
plt.ion()

#%%
df = pd.read_csv('creditcard.csv')

#sample_size = 10000    
sample_size = 100000
df_sample = df.sample(n=sample_size, random_state=42)

X = df_sample[['V14', 'Amount']].values 
# 'V14' est le résultat de PCA anonymisant les données d'utilisation de la carte, 
# c'est fortement corrélé à la fraude et
# 'Amount' est le montant de la transaction

y = df_sample['Class'].values
print(f"Classes : {np.bincount(y)}")
# 0 = transaction normale, 1 = fraude

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
# pour éviter de répéter le code de création du modèle :
def create_model():
    """Crée une nouvelle instance du modèle"""
    return keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

#%%
optimizers = {
    'Adam': keras.optimizers.Adam(learning_rate=0.01),
    'SGD': keras.optimizers.SGD(learning_rate=0.1),
    'Adagrad': keras.optimizers.Adagrad(learning_rate=0.1),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.01)
}

results = {}

#%%
print("--- ADAM ---")
model_adam = create_model()
model_adam.compile(optimizer=optimizers['Adam'], loss='binary_crossentropy', metrics=['accuracy'])
history_adam = model_adam.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
test_loss_adam, test_accuracy_adam = model_adam.evaluate(X_test, y_test, verbose=0)
results['Adam'] = {
    'test_loss': test_loss_adam, 
    'test_accuracy': test_accuracy_adam, 
    'history': history_adam
}

#%%
print("--- SGD ---")
model_sgd = create_model()
model_sgd.compile(optimizer=optimizers['SGD'], loss='binary_crossentropy', metrics=['accuracy'])
history_sgd = model_sgd.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
test_loss_sgd, test_accuracy_sgd = model_sgd.evaluate(X_test, y_test, verbose=0)
results['SGD'] = {
    'test_loss': test_loss_sgd, 
    'test_accuracy': test_accuracy_sgd, 
    'history': history_sgd
}

#%%
print("--- ADAGRAD ---")
model_adagrad = create_model()
model_adagrad.compile(optimizer=optimizers['Adagrad'], loss='binary_crossentropy', metrics=['accuracy'])
history_adagrad = model_adagrad.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
test_loss_adagrad, test_accuracy_adagrad = model_adagrad.evaluate(X_test, y_test, verbose=0)
results['Adagrad'] = {
    'test_loss': test_loss_adagrad, 
    'test_accuracy': test_accuracy_adagrad, 
    'history': history_adagrad
}

#%%
print("--- RMSPROP ---")
model_rmsprop = create_model()
model_rmsprop.compile(optimizer=optimizers['RMSprop'], loss='binary_crossentropy', metrics=['accuracy'])
history_rmsprop = model_rmsprop.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
test_loss_rmsprop, test_accuracy_rmsprop = model_rmsprop.evaluate(X_test, y_test, verbose=0)
results['RMSprop'] = {
    'test_loss': test_loss_rmsprop, 
    'test_accuracy': test_accuracy_rmsprop, 
    'history': history_rmsprop
}

#%%
for name, res in results.items():
    print(f"{name:10}: Test Loss={res['test_loss']:.4f}, Test Accuracy={res['test_accuracy']:.4f}")

best_acc = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
best_loss = min(results.keys(), key=lambda x: results[x]['test_loss'])
print(f"\nMeilleure accuracy: {best_acc}") # Adam et SGD ont la même accuracy quand n=10000
print(f"Meilleure loss: {best_loss}") # SGD et Adagrad ont la même loss quand n=100000

#%%
colors = {
    'Adam': '#2E8B57',
    'SGD': '#8A2BE2', 
    'Adagrad': '#32CD32',
    'RMSprop': '#DDA0DD' 
}

#%%
plt.figure(figsize=(10, 6))
plt.title('Erreur d\'entraînement', fontsize=14, fontweight='bold')
for name, res in results.items():
    training_error = [1 - acc for acc in res['history'].history['accuracy']]
    epochs = range(1, len(training_error) + 1)
    plt.plot(epochs, training_error, label=name, color=colors[name], linewidth=2.5)
plt.xlabel('Epochs')
plt.ylabel('Erreur')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
plt.figure(figsize=(8, 6))
plt.title('Erreur de test', fontsize=14, fontweight='bold')

test_errors = []
optimizer_names = []
for name, res in results.items():
    test_error = 1 - res['test_accuracy']
    test_errors.append(test_error)
    optimizer_names.append(name)

bars = plt.bar(optimizer_names, test_errors, color=[colors[name] for name in optimizer_names], alpha=0.7)
plt.ylabel('Erreur')
plt.grid(True, alpha=0.3, axis='y')

for bar, error in zip(bars, test_errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
            f'{error:.4f}', ha='center', va='bottom', fontweight='bold')
plt.show()

#%%
plt.figure(figsize=(10, 6))
plt.title('Perte d\'entraînement', fontsize=14, fontweight='bold')
for name, res in results.items():
    training_loss = res['history'].history['loss']
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, label=name, color=colors[name], linewidth=2.5)
plt.xlabel('Epochs')
plt.ylabel('Perte')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
plt.figure(figsize=(8, 6))
plt.title('Perte de test', fontsize=14, fontweight='bold')

test_losses = []
for name, res in results.items():
    test_losses.append(res['test_loss'])

bars = plt.bar(optimizer_names, test_losses, color=[colors[name] for name in optimizer_names], alpha=0.7)
plt.ylabel('Perte')
plt.grid(True, alpha=0.3, axis='y')

for bar, loss in zip(bars, test_losses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00001,
            f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
plt.show()

# %%
