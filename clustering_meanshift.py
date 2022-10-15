import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift


if __name__ == "__main__":
    path = './data/Candy/'
    dataset = pd.read_csv(path + 'candy.csv')

    print(dataset.head())

    X = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(X) # Automaticamente calculo el ancho de banda
    print(meanshift.labels_)

    # Se va realizar PCA para realizar una grafica en 2D
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_data = pca.transform(X)

    # Es una forma de ver los grupos generados en un estacio
    # 2-D
    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=meanshift.labels_)
    plt.show()