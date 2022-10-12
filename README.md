# Este es un proyecto del curso de Scikit-Learn de Platzi

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
