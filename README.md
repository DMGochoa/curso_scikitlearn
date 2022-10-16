# Este es un proyecto del curso de Scikit-Learn

Datasets que usaremos en el curso:

- World Happiness Report: Es un dataset que desde el 2012 recolecta variables sobre diferentes países y las relaciona con el nivel de felicidad de sus habitantes. Nota: Este data set lo vamos a utilizar para temas de regresiones
- The Ultimate Halloween Candy Power Ranking: Es un estudio online de 269 mil votos de más de 8371 IPs deferentes. Para 85 tipos de dulces diferentes se evaluaron tanto características del dulce como la opinión y satisfacción para generar comparaciones. Nota: Este dataset lo vamos a utilizar para temas de clustering
- Heart disease prediction: Es un subconjunto de variables de un estudio que realizado en 1988 en diferentes regiones del planeta para predecir el riesgo a sufrir una enfermedad relacionada con el corazón. Nota: Este data set lo vamos a utilizar para temas de clasificación.

## **¿Cómo afectan nuestros features a los modelos de Machine Learning?**

¿Qué son los features? Son los atributos de nuestro modelo que usamos para realizar una interferencia o predicción. Son las variables de entrada.

**Más features simpre es mejor?**

En general esto no es cierto, tenemos variables que pueden ser valiozas y otras irrelevantes, dependiendo de cuales usemos ocurrira lo siguiente:

- Puede agregar ruido al resultado.
- Dependiendo del modelo puede aumentar significativamente el costo computacional.
- Si se introducen muchas caracteristicas y estas contienen valores faltantes, se agregan sesgos muy significativos y perderiamos la capacidad de prediccion.

## **Una de las formas de saber que nuestros features han sido bien seleccionados es con el sesgo y la varianza.**

La mala seleccion de caracteristicas nos podria llevar as lo siguiente:

<img src="/Images/Markdown/var_bias.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

Nuestros modelos de **Machine Learning** pueden caer en dos tipos de escenarios. El primero es el **Underfitting** que significa que nuestro modelo no esta captando las caracteristicas y la variable de salida, por lo que se debe de investigar variables con mas significado, combinaciones o transformaciones para poder llegar a nuestra variable de salida. El segundo caso es el de **Overfitting** en donde le modelo termina es aprendiendo exactamente las variables de salida dadas las de entrada.  Si tenemos overfiting lo mejor es intentar seleccionar los features de una manera mas critica descartando aquellos que no aporten información o combinando algunos quedándonos con la información que verdaderamente importa.

Algunas Herramientas:

- Tecnicas de seleccion de caracteristicas y extraccion de caractereisticas (PCA).
- Regularizacion
- Balanceo: Oversampling y Undersampling

## Introduccion a PCA (Analisis de Componentes Principales)

Es normal encontrarse que en nuestros modelos de ML tengamos muchas caracteristicas y que las realciones no sean muy sencillas. Podriamos usar [**PCA**](https://www.youtube.com/watch?v=AniiwysJ-2Y&t=1117s) cuando nuestros conjuntos de datos tienen una alta cantidad de caracteristicasm hay una alta correlacion entre ellas, tenemos overfitting, y tambien un  alto coste computacional.

Se trata de reducir la dimensionalidad en caracteristicas atificiales, pero tratando de mantener la informacion.

Procedimiento:

1. Calculamos la matriz de covarianza para expresar las relaciones entre nuestras caracteristicas.
2. Hallamos los vectores propios y valores propios de esta matriz, para medir la fuerza y variabilidad de estas realaciones.
3. Ordenamos y escogemos los vectores propios con mayor variabilidad, esto es, que aportan mas informacion.

## **Otra alternativa a PCA y IPCA**

Otra herramienta que podemos utilizar es la transformacion por kernel. Un Kernel es una función matemática que toma mediciones que se comportan de manera no lineal y las proyecta en un espacio dimensional más grande en donde sen linealmente separables.

<img src="/Images/Markdown/kernel_transf.png"
     alt="Transformacion de kernel"
     style="float: left; margin-right: 10px;" />

Es util para casos en donde las caracteristicas no son linealmente separables

Lineales: $k(x,y) = x \times y$
Polinamiales: $k(x,y) = (x * y)^p$
Gaussianos (RBF): $k(x,y) = e^{\frac{||x-y||^2}{2 \sigma^2}}$

Lo complejo es detectar cuando y que kernel utilizar.

## Regularizacion

La regularizacion consiste en disminuir la complejidad del modelo a traves de una penalizacion aplicada a sus variables mas irrelevantes. Para entendes bien la **Regularizacion** debemos entender primero que es la funcion de entrenamiento. La funcion de entrenamiento es una funcion que nos indica que tan alejado estamos de la realidad.

En la literatura hay tres tipos principales, y estos son:

- **L1 Lasso**: Reducir la complejidad a traves de la eliminacion de caracteristicas que no aportan demasiado al modelo.
- **L2 Ridge**: Reducir la complejidad disminuyendo el impacto de ciertas caracteristicas a nuestro modelo.
- **ElasticNet**: Es una combinacion de las dos anteriores.

No hay campeon definitivo para todos los problemas, es por eso que toca realizar pruebas o con la experiencia tomar diciones de cual usar. Si hay pocas caracteristicas que se relacionen directamente con la variable a predecir: **Probar Lasso**. Si hay varias caracteristicas relacionadas con la variable a predecir, **probar Ridge**.

## Problema de los Valores Atípicos

Primero podemos definir que un **dato atípico** son todas aquellas mediciones que se encuentren por fuera del comportamiento general de una muestra de datos. Estos pueden indicar variabilidad, errores de medicion, o novedades.

- Pueden ser problematicos o no.
     1. Pueden generar sesgos importantes en los modelos de Machine Learning.
     2. Pueden contienener informacion relevante sobre la naturaleza de los datos.
     3. Deteccion temprana de fallos.

Para identificarlos se puede hacer por:

- **Z-score**: Mide la distancia (en desviaciones estandar) de un punto dado a la media.
- Tecnicas de clustering como DBSCAN.
- O con el metodo de Dixon si $q < Q_1 - 1.5 \times IQR$ o $q > Q_3 + 1.5 \times IQR$. Es de resaltar que el factor 1.5 puede variar a 3.0 dependiendo de si es una cola pesada y picada.

Con scikit podemos usar metodos conocidos como regresiones robustas. Algunos de estos modelos son:

- **RANSAC (Random Sample Consensus)**: Usamos una muestra aleatoria sobre el conjunto de datos que tenemos, buscando la muestra que mas datos "buenos" logre incluir. El modelo asume que los "valores malos" no tienen patrones especificos.
- **Huber Regresor**: No ignora los valores atipicos, disminuye su influencia en el modelo. Los datos son tratados como atipicos si el error absoluto de nuestra perdida esta por encima de un umbral llamado epsilon. Se ha demostrado que un valor de epsilon que logra un 95% de eficiencia estadistica es $1.35$.

## Metodos de ensamble

1. Combinar diferentes metodos de ML con diferentes configuraciones y aplicar un metodo para lograr un consenso.
2. La diversidad es una buena opcion.
3. Los metodos de ensamble se han destacado por ganar muchas competencias de ML.

**Bagging**: es como si nuestros algoritmos de ML fueran diferentes expertos y lo que hacemos es tener la opinion de cada uno de los expertos, esto lo hace de manera paralela. (Bootstrap AGGragation)

**Boosting**: en este caso le pedimos a un experto su criterio sobre un problema. Medimos su posible error, y luego usando ese error calculado le pedimos a otro experto su juicio sobre el mismo problema.

## Clustering

Estrategia de machine learning de aprendizaje no supervizado. Los algoritmos de clustering son las estrategias que podemos usar para agrupar los datos de tal manera que todos los datos pertenecientes a un grupo sean lo mas similares que sea posible entre si, y lo mas diferentes a los de otros grupos.

Algunos casos en los que se aplica clustering:

- No conocemos con anterioridad las etiquetas de nuestros datos.
- Queremos descubrir patrones ocultos a simple vista.
- Queremos identificar datos atipicos.

Algunos algoritmos son:

- k-means
- Spectral Clustering
- Meanshift
- Clustering Jerarquico
- DBScan

## Validacion Cruzada

La ultima palabra la tienen los datos. Necesitamos mentalidad de testeo y todos los modelos son malos, solamente que algunos resultan utiles.

Tipos de validacion:

- Dividir datos de entrenamiento y de prueba (Hold-Out)
  - Se requiere un protoripo rapido
  - No se tiene mucho conocimiento
  - No se cuenta con un equipo con abundante poder de computo
- Usar validacion cruzada (K-folds)
  - Recomendable en la mayoria de los casos.
  - Se cuenta con un equipo suficiente para desarrollar ML
  - Se requiere la integracion con tecnicas de optimizacion parametrica
  - Se tiene mas timepo para las pruebas
- Validacion Cruzada (LOOCV)
  - Se tiene gran poder de computo.
  - Se cuenta con pocos datos como para dividir por trining/Test
  - Se busca hacer la validacion muy exaustiva

## Enfoques de Optimizacion Parametrica

1. Optimizacion Manual
   1. Escoger el modelo que queremos ajustar.
   2. Buscar en la documentacion que parametros hay.
   3. Identificar los posibles ajustes.
   4. Probar combinaciones iterativamente.
2. Optimizacion por grilla de parametros [**GridSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
   1. Definir una o varias metricas que queramos optimizar.
   2. Identificar los posibles valores que pueden tener los parametros.
   3. Usar Cross Validation.
   4. Esperar a ver si hay algun resultado.
3. Optimizacion por busqueda Aleatoria [**RandomizedSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
   1. Definir una o varias metricas que queramos optimizar.
   2. Identificar los rangos de valores que pueden tomar ciertos parametros.
   3. Usar Cross Validation.
   4. Esperar a ver si hay algun resultado.
