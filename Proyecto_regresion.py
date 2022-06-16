# Proyecto Final por: Jose Arregoces

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

datos_originales = pd.read_csv('bmw_pricing_challenge.csv')

Filtro_engine_power = datos_originales['engine_power'] > 1 # filtro usado para eliminar las muestras con "engine_power" = 0
datos = datos_originales [Filtro_engine_power]

Filtro_engine_power = datos['mileage'] > 1 # filtro usado para eliminar las muestras con "mileage" < 0
datos = datos [Filtro_engine_power]

# se buscan datos NULL y se suman por categorías
datos.isnull().sum()


#distribucion de precio
plt.figure(figsize=(12,6))
sns.kdeplot(data=datos, x='price')
plt.title('Distribution of Price', size=20)

Filtro_price = datos_originales['price'] > 75000


#distribucion de kilometraje
plt.figure(figsize=(12,6))
sns.kdeplot(data=datos, x='mileage')
plt.title('Distribution of mileage', size=20)


#distribucion de engine power
plt.figure(figsize=(12,6))
sns.kdeplot(data=datos, x='engine_power')
plt.title('Distribution of engine power', size=20)




# analizando la distribucion del precio podemos ver que hay unos otlayers en el precio, se puede ver que arriba de 7.5000 hay muy pocos valores y el precio máximo llega hasta los 175.000
# por lo tanto estas muestras se pueden conciderar como outlayers y deben ser eliminadas
# por la parte inferior no se perciben aoutlayers 
superior = 75000
Filtro_outlayer_superior_precio = datos['price'] < superior # filtro usado para eliminar outlayer superiores
datos = datos [Filtro_outlayer_superior_precio]

# analizando la distribucion del kilometraje, se puede observar que arriba 40.000 hay muy pocos datos y los datos llegan hasta 100.000, por lo tanto a los valores arriba de 50.000 se les puede considerar outlayers
superior = 400000
Filtro_outlayer_superior_mileage = datos['mileage'] < superior # filtro usado para eliminar outlayer superiores
datos = datos [Filtro_outlayer_superior_mileage]


# analizando la distribucion de "engine power", se puede observar que arriba 300 y abajo de 80  hay muy pocos datos, por lo tanto se les puede considerar outlayers
superior = 300
inferior = 80 
Filtro_outlayer_superior_engine_power = datos['engine_power'] < superior # filtro usado para eliminar outlayer superiores
datos = datos [Filtro_outlayer_superior_engine_power]

Filtro_outlayer_inferior_engine_power = datos['engine_power'] > inferior # filtro usado para eliminar outlayer superiores
datos = datos [Filtro_outlayer_inferior_engine_power]
##############################




#correlacion entre las variables 
correlacion = datos.corr()
# graficar correlación 
plt.figure(figsize=(8,8))
sns.heatmap(round(correlacion, 2),cmap='viridis', annot=True)

############################################
# calculando los años de uso 
datos['registration_date']= pd.to_datetime(datos['registration_date'])
datos['sold_at']= pd.to_datetime(datos['sold_at']) # se convierten a un tipo datatime para poder exraer el año 
# se extraen los años de venta y registro, para poder calcular los años de uso de cada vehículo
datos['sold_year']= datos['sold_at'].dt.year
datos['registration_year']= datos['registration_date'].dt.year
datos['years_used']= datos['sold_year'] - datos['registration_year']
############################################

plt.figure(figsize=(12,6))
sns.kdeplot(data=datos, x='years_used')
plt.title('Distribution of years_used', size=20)

# analizando la distribucion de los años de uso, se puede observar que arriba 15 hay muy pocos datos y los datos llegan hasta 25, por lo tanto a los valores arriba de 15 se les puede considerar outlayers
superior = 15
Filtro_outlayer_superior_years_used = datos['years_used'] < superior # filtro usado para eliminar outlayer superiores
datos = datos [Filtro_outlayer_superior_years_used]
###################################################
#eliminando caracteristica innecesarias 
del datos['registration_date']
del datos['sold_at']
del datos['sold_year']
del datos['registration_year']
del datos['maker_key']
#del datos['model_key']
correlacion = datos.corr()
# graficar correlación 
plt.figure(figsize=(8,8))
sns.heatmap(round(correlacion, 2),cmap='viridis', annot=True)
# limpieza de datos 
todos = datos['engine_power'] >= 1

###################################################
marcas = datos['model_key'].value_counts()
F_todos = datos['model_key'][todos]

for i in F_todos.index:
    datos['model_key'][i] = datos['model_key'][i].replace(" Gran Turismo","1")
    datos['model_key'][i] = datos['model_key'][i].replace(" Gran Coupé","2")
    datos['model_key'][i] = datos['model_key'][i].replace(" Gran Tourer","3")
    datos['model_key'][i] = datos['model_key'][i].replace(" Active Tourer","4")
    datos['model_key'][i] = datos['model_key'][i].replace("ActiveHybrid 5","1111")
    datos['model_key'][i] = datos['model_key'][i].replace("M","99")
    datos['model_key'][i] = datos['model_key'][i].replace(" M","88")
    datos['model_key'][i] = datos['model_key'][i].replace("X","9")
    datos['model_key'][i] = datos['model_key'][i].replace("Z","11")
    datos['model_key'][i] = datos['model_key'][i].replace("i","111")
    datos['model_key'][i] = datos['model_key'][i].replace(" ","999")
    
marcas = datos['model_key'].value_counts()
datos['modelo'] = pd.to_numeric(datos['model_key'])
del datos['model_key']
# se calcula la nueva correlacion
correlacion = datos.corr()
# graficar correlación 
plt.figure(figsize=(8,8))
sns.heatmap(round(correlacion, 2),cmap='viridis', annot=True)


###################################################

print (datos['fuel'].value_counts())
# dado que esta caracteristica está muy concetrada en un solo valor no sirve para analizar la variación del precio 
del datos['fuel']
# como todos los carros son de la misma marca, se elimina esta columna
# se pasan los datos booleanos a 0 o 1
datos['feature_1']=datos['feature_1'].astype('int')
datos['feature_2']=datos['feature_2'].astype('int')
datos['feature_3']=datos['feature_3'].astype('int')
datos['feature_4']=datos['feature_4'].astype('int')
datos['feature_5']=datos['feature_5'].astype('int')
datos['feature_6']=datos['feature_6'].astype('int')
datos['feature_7']=datos['feature_7'].astype('int')
datos['feature_8']=datos['feature_8'].astype('int')


#################################3
# car_type top 3

car_type= datos['car_type'].value_counts()
top_3_car_type= car_type[:3]
datos['car_type']= datos['car_type'].apply(lambda x:x if x in top_3_car_type else 'other')
################################
# color 
paint= datos['paint_color'].value_counts()
top_6_paints= paint[:6]
datos['paint_color']= datos['paint_color'].apply(lambda x:x if x in top_6_paints else 'other')
#################################
# se convierten las variables categprocas a numericas
datos= pd.get_dummies(datos,columns=['paint_color', 'car_type'],drop_first=True)
datos.info()



##################### modelo ###########################
X = datos.drop('price', axis=1) # todos menos el precio
Y = datos.price 
# separar datos 
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, train_size=0.85, random_state=5)


LReg=LinearRegression()
LReg.fit(X_train,Y_train)
Y_pred=LReg.predict(X_test)
R2=r2_score(Y_test , Y_pred)
print ('R2:',R2.round(5))


X_test_scaled2 = StandardScaler().fit_transform(X_test)
X_train_scaled2 = StandardScaler().fit_transform(X_train)
LReg2=LinearRegression()
LReg2.fit(X_train_scaled2,Y_train)
Y_pred2=LReg2.predict(X_test_scaled2)
R22 = r2_score(Y_test , Y_pred2)
print('R2:',R22.round(5))



####################################
#grafica de prediccion vs valores reales

plt.figure(figsize=(8,8))
plt.scatter(Y_test, Y_pred)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
lims = [0, 70000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

######################################################################3
#Random Forest
from sklearn.ensemble import RandomForestRegressor
n_estimators=30
model = RandomForestRegressor(random_state=42,max_depth=7,max_features=9,n_estimators=n_estimators)
model.fit(X_train, Y_train)

model_score = model.score(X_train,Y_train)
y_predicted = model.predict(X_test)
R2_DT = r2_score(Y_test, y_predicted)
print('R2:', R2_DT.round(3))


plt.figure(figsize=(8,8))

plt.scatter(Y_test, y_predicted)
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
lims = [0, 70000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


##################################################################
#PCA
X_scaled2 = StandardScaler().fit_transform(X)
matriz_cov2 = np.cov(X_scaled2.T)

eig_vals, eig_vecs = np.linalg.eig(matriz_cov2)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x: x[0], reverse=True)
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])
    
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(23, 21))

    plt.bar(range(21), var_exp, alpha=0.5, align='center',
    label='Varianza individual explicada', color='g')
    plt.step(range(21), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='best')
    plt.tight_layout()
#######################################################################

