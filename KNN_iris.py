#⁠Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#⁠Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

#Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#⁠Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Créer et entraîner le modèle K-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Prédire et évaluer la précision
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#Exemple de caractéristiques pour une nouvelle fleur
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Remplace ces valeurs par celles de la nouvelle fleur

#Normaliser les caractéristiques de la nouvelle fleur
new_flower_scaled = scaler.transform(new_flower)

#Prédire la classe de la nouvelle fleur
predicted_class = knn.predict(new_flower_scaled)

#Obtenir le nom de la classe
class_names = iris.target_names
print(f"La fleur prédite est de la classe : {class_names[predicted_class[0]]}")

