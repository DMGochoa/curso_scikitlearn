import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

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

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('='*64)
    print(accuracy_score(knn_pred, y_test))
    
    clasificadores = {
        'KNC':KNeighborsClassifier(),
        'LSVC':LinearSVC(),
        'SVC':SVC(),
        'SGDC':SGDClassifier(),
        'DTC':DecisionTreeClassifier()
    }

    for name, estimator in clasificadores.items():
        bag_class = BaggingClassifier(
            base_estimator=estimator, 
            n_estimators=50).fit(X_train, y_train)
        bag_pred = bag_class.predict(X_test)
        print('='*64)
        print(name)
        print(accuracy_score(bag_pred, y_test))
        
    

    
