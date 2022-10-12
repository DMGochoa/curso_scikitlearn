import pandas as pd
import sklearn 
import matplotlib.pyplot as plt
# Se Importan los modulos para hacer el Kernel PCA.
from sklearn.decomposition import KernelPCA
# Se hace la importacion de un modulo que nos sirva para comparar.
from sklearn.linear_model import LogisticRegression
# Para preparar los datos antes del entrenamiento
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Se hace la carga de los datos
    path = './data/Heart/'
    df_heart = pd.read_csv(path + 'heart.csv')
    #print(df_heart.head())

    # Se reparte los datos de salida y los de entrada
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    # Escalador de los datos
    #print(df_features)
    scale_X = StandardScaler()
    scale_X.fit(df_features)

    # Aplicar el escalado
    df_features = scale_X.transform(df_features)

    # Se hace la separacion de los conjuntos de validacion de de entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    #print(X_train.shape[0]/df_features.shape[0])
    kpca = KernelPCA(n_components=4, kernel='rbf')
    kpca.fit(X_train)

    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(df_train, y_train)

    print("Score KPCA: ", logistic.score(df_test, y_test))
