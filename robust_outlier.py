import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Modelos que se van usar
from sklearn.linear_model import (
    RANSACRegressor, 
    HuberRegressor
)
from sklearn.svm import SVR

# Modulos de preprocesamiento y de acondicionamiento
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    path = './data/Felicidad/'
    # Importamos el dataset
    dataset = pd.read_csv(path + 'felicidad_corrupta.csv')
    print(dataset.head())

    # Establecer variables que van a ser usadas para predecir
    X = dataset[dataset.columns.drop(['country', 'score', 'rank'])]
    # Target
    y = dataset[['score']]
    # Separacion en datos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=42
    )

    # Guardar los estimadores
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(), # Es un meta estimador por lo que se puede
        # trabajar con  diferentes estimadores que podemos definir
        'HUBER': HuberRegressor(epsilon=1.35) # Si disminuye tendremos menos atipicos
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print("=" * 32)
        print(name)
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title('Predicted VS Real ' + name + ' ' + str(mean_squared_error(y_test, predictions)))
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.show()
