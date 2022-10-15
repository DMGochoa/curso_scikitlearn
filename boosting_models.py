import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Se hace la carga de los datos
    path = './data/Heart/'
    df_heart = pd.read_csv(path + 'heart.csv')
    print(df_heart['target'].describe())

    X = df_heart.drop(['target'], axis=1)
    y = df_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # En este caso toca decir cuantos arboles va tener el modelo
    values = list()
    for i in np.arange(10,151,10):
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)
        values.append([i, accuracy_score(boost_pred, y_test)])
    values = np.array(values)

    plt.figure()
    plt.plot(values[:,0], values[:,1])
    plt.title('Validation', fontsize=22)
    plt.xlabel('Quantity of trees', fontsize=18)
    plt.ylabel('Accuaracy Score', fontsize=18)
    plt.grid()
    plt.show()