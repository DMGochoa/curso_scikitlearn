import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


if __name__ == "__main__":
    sns.set()
    path = './data/Candy/'
    dataset = pd.read_csv(path + 'candy.csv')

    #print(dataset.head())

    X = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X) # Automaticamente calculo el ancho de banda
    print(kmeans.labels_)

    # Se va realizar PCA para realizar una grafica en 2D
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_data = pca.transform(X)

    # Es una forma de ver los grupos generados en un estacio
    # 2-D
    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_)
    plt.show()

    # Agregamos las labels al df
    dataset['groups'] = kmeans.labels_

    sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent','groups']], hue = 'groups')