# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:48:00 2020

@author: ASUS
"""

#modulos necesarios para la clasificación y poryeccion de los resultados en un mapa
from mpl_toolkits.basemap import Basemap
import cartopy.crs as crs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Import modules for ML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from warnings import warn
import sklearn.metrics as sklm
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans, vq
import seaborn as sns

#importación del dataset
df=pd.read_excel('dataset.xlsx',sheet_name='Pineda',index_col=0)

#inpección de las características del dataset
print(df.columns) #nombre de las columnas
print(df.head()) #primeras filas del dataset
print(df.describe()) #estadísticas descriptivas
print(df.isnull().sum()) #número de valores nulos en el dataset
print(dF.dtypes) #tipo de datos en el dataset

#listas de nombres de columnas para facilidad en el procesamiento
#columnas numéricas
NUMERIC_COLUMNS=['Año','#_epistolas','Lat', 'Lon','Freq_Remitente','freq_ciudad_origen']
#columna de etiquetas para el aprendizaje
LABELS=['Relacion biblioteca']
NUM_LAB=NUMERIC_COLUMNS+LABELS
#columnas textuales
TEXT_COLUMNS = [c for c in df.columns if c not in NUM_LAB]
TEXT_NUM_COLUMNS=TEXT_COLUMNS+NUMERIC_COLUMNS
print(df[TEXT_COLUMNS].head())

#tratar los valores nulos en la columna textual
df['texto']=df['texto'].fillna('')


#limpieza de texto en la columna a través de una expresión regular: eliminar caracteres extraños como paréntesis, puntos, llaves, comillas, etc.
caracteres_reemplazar='/|"|\[...]|\[. . .]|\(|;|\)|\'|\[…]|\.|\,|\¿|…|\?'
df['texto']=df['texto'].str.replace(caracteres_reemplazar, '',regex=True)

#definición de la función para generar el vocabulario y vectores de palabras en las columnas textuales

def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Deshacerse de las columnas numéricas
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop,axis=1)
    
    #Reemplazar nulos con espacios en blanco
    text_data.fillna('',inplace=True)
    
    # Juntar todas las columnas textuales en una sola fila con un espacio entre sí
    return text_data.apply(lambda x: " ".join(x), axis=1)

stop_words_spanish=stopwords.words('spanish')
# Creación de patron para tokenizar (token por palabra): TOKENS_BASIC
TOKENS_BASIC = '\\S+(?=\\s+)'

# Abrir instancia del vectorizador: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC,stop_words=stop_words_spanish,ngram_range=(1,2))

# Generación de los vectores por cada token con la función definida arriba
text_vector = combine_text_columns(departamento_punto)
# Aplicar el vectorizador a los tokens
vec_basic.fit(text_vector)
# mostrar el nombre de tokens y los primeros 500 tokens del vocabulario
msg = "Hay {} tokens en texto si separamos según el patron Tokens_Basic"
print(msg.format(len(vec_basic.get_feature_names())))
# exámen del vocabulario
print(vec_basic.get_feature_names()[:500])


# obtener la información textual: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Obtener la información numérica: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: departamento_punto[NUMERIC_COLUMNS], validate=False)


# aplicar y transformar la info textual: just_text_data
just_text_data = get_text_data.fit_transform(departamento_punto)

# aplicar y transformar la info numérica: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(departamento_punto)

# Mostrar los primeros registros para observar los resultados
print('info textual')
print(just_text_data.head())
print(just_text_data.shape)
print('\nInfo numérica')
print(just_numeric_data.head())
print(just_numeric_data.shape)


#algoritmo randomforest dio mejores resultados que otros

# Separar el dataset de columnas de texto y numéricas en particiones de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(df[TEXT_NUM_COLUMNS],
                                                    pd.get_dummies(dF[LABELS]), 
                                                    test_size=0.3,random_state=456)


# Iniciar el objeto pipeline para ensamblar de forma práctica tanto los procesos de las columnas textuales como numéricas: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(RandomForestClassifier()))
    ])
# aplicar el pipeline a los datos de entrenamiento
pl.fit(X_train,y_train)
#predecir sobre los datos de testeo sin etiquetas
y_pred = pl.predict(X_test)
# Calcular la exactitud de las predicciones comparando con las etiquetas 
accuracy = pl.score(X_test,y_test)
print(classification_report(y_test, y_pred))
print("\nExacitud: ", accuracy)

#calcular las probabilidades (similar al método predict, pero este no clasifica en valores booleanos y si en valores de probabilidad)
prediccion=pl.predict_proba(departamento_punto)
print(prediccion)
# guardar las predicciones en un dataset: prediction_df
prediccion_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
                             index=departamento_punto.index,
                             data=prediccion)
print(prediccion_df)
#guardar a un archivo csv
prediccion_df.to_csv('coleccionismo.csv',index=False)

#generar una columna en el dataset original con las predicciones
df['coleccionista_si']=prediccion_df['Relacion biblioteca_1']

#mapeo de la correspondencia según la clasificación lograda de corresponsales
# Definir la caja de límites geográficos
boundingbox = [-83.7,-5.15,-63.49,13.78] 

# Definir el objeto Basemap con las coordenadas de la lista boundingbox y proyección mercator
m = Basemap(width=12000000,height=9000000,resolution='h',
            llcrnrlon = boundingbox[0],
            llcrnrlat = boundingbox[1],
            urcrnrlon = boundingbox[2],
            urcrnrlat = boundingbox[3],
            projection='merc')

# Dibijar los continentes en gris,
# Fronteras de países en negro, fronteras departamentales en gris y oceano en azul
m.fillcontinents(color='gray', zorder= 0)
m.drawcountries(color='black')
m.drawstates(color='gray')
m.drawlsmask( ocean_color='aquamarine')

#definir las variables geográficas longitud y latitud
lon = [x for x in departamento_punto['Lon']]
lat = [x for x in departamento_punto['Lat']]
#definir los valores de clasificación según si es o no colaborador para representar por color
color_colec= [x for x in departamento_punto['coleccionista_si']]

# Dibujar los puntos y mostrar la visualización
scatter_1=m.scatter(lon, lat,c=color_colec, cmap = 'brg',s=7, latlon = True, alpha = 0.8)
plt.axis('off')
plt.title('Mapa de Colaboradores (NPL)')
#Especificar la leyenda para mostrar los colores únicos
Legend=plt.legend(*scatter_1.legend_elements(),title="Clases",title_fontsize=12.5,fontsize=10.5,frameon=True,framealpha=0.7,shadow=True,loc='lower right',facecolor='beige')
#Definir el texto de los elementos de la leyenda
Legend.get_texts()[0].set_text('No colaborador')
Legend.get_texts()[1].set_text('Colaborador')
#guardar la imagen en un directorio
plt.savefig('Mapa_colaboradores_coleccionistas.png', dpi = 450) 
#mostrar todo el resultado          
plt.show()
