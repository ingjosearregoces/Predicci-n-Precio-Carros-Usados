# Predicci-n-Precio-Carros-Usados
Proyecto que logra predecir bajo ciertos atributos y caracteristicas el precio de carros usados.
PREDICCIÓN DE COSTOS RESPECTO A COCHES USADOS

Por:
Leonardo Jose Viana De Avila
Leonardoj_vianad@javeriana.edu.co
Jose Carlos Arregocés Castro
josearregocesc@javeriana.edu.co

Introducción:
Mediante el presente trabajo se busca predecir de manera certera el precio de un vehículo usado teniendo en cuenta ciertas características relevantes que influyen en el valor final de venta. A partir de un dataset (anexo[1]) que contiene información de interés sobre un vehículo como, por ejemplo, recorrido neto, modelo, poder del motor, etc, se pretende estimar el precio final para venta del carro usado.  Los métodos utilizados fueron estimación lineal con la cual se logró un R2 del 78.8% y Random Forest con un resultado del 86.7%. Fue necesario usar técnicas de depuración de datos como por ejemplo eliminar las columnas que no eran necesarias para nuestro análisis como por ejemplo fecha de matrícula o año de registro y por otra parte implementar el cambio de tipo de dato como pasar de String a número para poder depurar correctamente los datos y filtrarlos de manera exacta. Todo lo anterior fue desarrollado a partir del lenguaje de programación Python el cual fue el tópico principal de la asignatura a lo largo de todo el semestre. 
Todo el desarrollo del codigo y su respectiva explicacion se encuenctra en el video solicitado en la rubrica (anexo[2]).
Por otra parte se anexa tambien la URL de donde se puede extraer el dataset que se uso en la practica (anexo [3]).
Finalmente se incluye la URL del repositorio en la pagina web GitHub, en donde se puede encontrar el presente documento y su respectivo codigo en lenguaje Python (anexo[4]).
A analizar:
Marca, modelo, millaje, fecha de registro, combustible, color, tipo de coche, precio, fecha de venta, año de venta


Desarrollo:
1.	Se importaron las librerías correspondientes (pandas, matplotlib, seaborn y sklearn)
2.	Se almacena el dataframe en el programa y se proceden a descartar datos que se consideren poco útiles para el análisis en primera instancia (como millaje menor a una milla) puesto a que a criterio del grupo no aporta ninguna información relevante.
3.	Teniendo en cuenta lo anterior se mira la distribución de cada característica individual y se eliminan los datos que se consideren que aporten menos y se modifica todo el dataframe de tal manera que este solo contenga los datos útiles para el análisis concerniente.
4.	Se halla la correlación entre las variables y se mira cuáles son las más útiles para analizar en caso de que exista la posibilidad de descartar alguna para simplificar el cálculo.
5.	Luego de tener los datos más importantes ordenados de forma adecuada se procede a crear el modelo determinando el conjunto de prueba y el conjunto de entrenamiento en conjunto con el R2 para saber que tan preciso es (78.8% para nuestro caso).
6.	Se grafico la estimación del modelo para verificar que concordase con los datos provistos para este dataset.
7.	A demás de esto se hizo un modelamiento por Random Forest y se calculó el nuevo R2(para este segundo caso 86.7%). 
TAMAÑO DATASET:  4842 datos













Resultados:

 
Ilustración 1 Distribución del precio
En la ilustracion 1 se pueden observar los resultados obtenidos a partir del dataframe y los datos obtenidos. Se puede observar que el precio tomado en cuenta para nuestro analisis parte de un valor un poco mas alto de 0 hasta los 75000 a criterio del grupo.
 
Ilustración 2 Distribución de millaje
En la ilustracion 2 se pueden observar los resultados obtenidos a partir del dataframe y los datos obtenidos. Se puede observar que el precio tomado en cuenta para nuestro analisis se pueden tomar valores de 0 hasta los 40000 a criterio del grupo puesto a que más arriba o abajo no se encuentra información significativamente relevante para el análisis .
 

Ilustración 3 Distribución de poder de motor
En la ilustración 3 al igual que en la 1 y 2, se puede observar una distribución de predicción de datos según el poder del motor. En donde los valores de interés hacia la práctica oscilan entre 80 y 300 aproximadamente.
 
Ilustración 4 Distribución de años de uso
Uno de los datos más importantes que se pueden tener en cuenta en el proceso de predicciones para los fines del presente informe son los años de uso. A través de los años de uso se puede estimar el valor de un vehículo usado ya que de esto dependen las condiciones en las que el carro se pueda encontrar. En la ilustración 4 se puede observar la distribución de interés de los años de uso de los vehículos que se encuentran en un rango entre 1 año y 18 años aproximadamente.
 
Ilustración 5. Matriz de influencia sobre precio 
En a ilustración 5 se puede observar la influencia de ciertos atributos que componen la predicción del precio a futuro en un vehículo usado. Como se puede observar el millaje y el poder del motor son los datos más relevantes a la hora de valorar un carro respecto al tiempo. Con base en esto se pueden calcular algunas predicciones que logran dar una proyección certera de cuánto puede costar el carro a medida que estos parámetros varían.
 
Ilustración 6. Regresión lineal estimación Lineal
En la regresión de estimación lineal se pueden ver los datos un poco más dispersos en comparación con la ilustración 9 que pertenece a Random Forest, corroborando que el segundo es un mejor método para alcanzar una aproximación más cercana al 100%. Con el método actual se logró una precisión del 78.8%.
 
Ilustración 7 Regresión lineal Random Forest
En la regresión de Random Forest se pudo obtener un mejor resultado en la regresión lineal como se puede observar en la ilustración 9. Se visualizan datos más cercanos a la línea a medida que la distribución avanza esto es un buen indicativo de la practica ya que se logró una precisión del 86.7%
Conclusiones:
1.	Como principal análisis de los resultados podemos evidenciar que los modelos de regresión lineal a pesar de tomar menos tiempo de ejecución a la hora de implementarlo tienden a ser menos precisos como otros métodos de modelamiento tales como random forest.
2.	Podemos deducir de las gráficas realizadas mediante esta entrega que se pueden evitar tomar medidas directamente con la vista para excluir datos de las distribuciones normal y en su lugar agruparlos respecto a la desviación estándar puesto a que esto nos permite excluir los datos limpiamente de manera menos objetiva.
3.	Determinar las variables con las que se piensa evaluar el modelo es parte fundamental del desarrollo de un buen análisis puesto a que añadir datos que no poseen dependencia puede generar incoherencias en dicho modelo.

Anexos:
[1] https://www.kaggle.com/datasets/danielkyrka/bmw-pricing-challenge
[2] https://www.youtube.com/watch?v=tP9VR3BA0Ls&feature=youtu.be
[3] https://www.kaggle.com/datasets/danielkyrka/bmw-pricing-challenge
[4] https://github.com/ingjosearregoces/Predicci-n-Precio-Carros-Usados
![image](https://user-images.githubusercontent.com/105678386/171969764-322db6ea-cfe3-4c7a-8e63-288ac7f0de71.png)
