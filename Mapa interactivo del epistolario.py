# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:06:46 2020

@author: ASUS
"""


#importar los módulos necesario
import pandas as pd #necesario para la gestión de los datos
import numpy as np
from geopy.geocoders import Nominatim #necesario para usar el servicio de georeferenciación
from geopy.extra.rate_limiter import RateLimiter #necesario para evitar errores en el tiempo de respuesta del servidor del servicio de georeferenciación
import geopandas as gpd #necesario para la gestión de los datos espacial, es la versión espacial del módulo pandas
from shapely.geometry import Point #para generar las geometrías de punto que se usaran en el mapa del epistolario
import folium #para generar la visualización interactiva del mapa

#importar el marco de datos que contiene una columna con los nombres de las ubicaciones de orígen de cada carta
df=pd.read_excel('Pineda.xlsx')

# Inicializar el objeto Nominatim que usa openstreetmap
geolocator = Nominatim(user_agent="UbicacionesPineda")
#para evitar el timeout error del servidor se usa una demora de 1 segundo en cada llamada al servicio
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
#extrae valores unicos en la columna que contiene los nombres de las ubicaciones de cada carta para hacer mas eficiente la busqueda
ubicaciones = df['Ciudad_origen'].unique()
#elimina la entrada None type 
index = np.argwhere(ubicaciones==None)
ubicaciones = np.delete(ubicaciones, index)
#geocodifica los valores del array de valores unicos Ubicaciones y lo guarda en un diccionario: d
if d:   
    d = dict(zip(ubicaciones, pd.Series(ubicaciones).apply(geolocator.geocode)))
else:
    None

#inserta en el dataset original los resultados según las coincidencias entre el diccionart
df['location']  = df['Ciudad_origen'].map(d)

#%%
#Las coordenadas deben ser separadas de los demás datos retornados por el servidor
#primero en una columna para latitud y longitud
df['coordinates']=df['location'].str[1]
df['coordinates'] = df['coordinates'].astype(str)
df['coordinates']=df['coordinates'].str.replace('(', ' ')
df['coordinates']=df['coordinates'].str.replace(')', ' ')

#luego latitud y longitud independientemente
df['Latitud'], df['Longitud'] = df['coordinates'].str.split(',', 1).str
#tratar los nulos
df['Latitud']=df['Latitud'].replace('nan',0)
df['Latitud']=df['Latitud'].replace('None',0)
df['Longitud']=df['Longitud'].replace('None',0)
df['Longitud']=df['Longitud'].fillna(0)
df['Latitud']=df['Latitud'].astype(float)
df['Longitud']=df['Longitud'].astype(float)
#inpeccionar los primeros 50 resultados
print(df.head(50))

#agrupar por ubicación única todas las cartas y contar cuantas cartas por lugar. Mostrar el resultado de la agregación
ubicaciones_counts = df.groupby(['Ciudad_origen']).size()
print(ubicaciones_counts)
counts_df = ubicaciones_counts.to_frame()
counts_df.reset_index(inplace=True)
counts_df.columns = ['Ciudad_origen', '#_epistolas']
print(counts_df.head(10))
# Combinar el marco de datos con conteo con el marco de datos original según el nombre de cada lugar
df_and_counts = pd.merge(df,counts_df, on = 'Ciudad_origen')
print(df_and_counts.head(10))

#generar las geometrías tomando la longitud y la latitud
df_and_counts['geometry'] = df.apply(lambda x: Point(float(x.Longitud), float(x.Latitud)), axis=1)
#Generar un dataframe para guardar las geometrias, se indica el sistema de coordendas
df_geo = gpd.GeoDataFrame(df_and_counts,crs =3116, geometry = df_and_counts.geometry)
#mostrar el sistema de referencia, columnas y tipo de dato de cada columna
print(df_geo.crs)
print(df_geo.columns)
print(type(df_geo))

#coordenadas de Colombia como punto de partida para la visualización interactiva
colombia=[4.598889, -74.080833]
# Construcción del mapa con la ubicación de partida
downtown_map = folium.Map(location = colombia)
# Creación de los popups en un bucle creado para hacer los marcadores en el mapa con numero de cartas por ubicación y el nombre de la ubicación
for row in df_geo.iterrows():
    row_values = row[1] 
    location = [row_values['Latitud'], row_values['Longitud']]
    popup = ('Ubicación: ' + str(row_values[2]) + 
             ';  ' + 'Número epístolas: ' + str(row_values[-2]))
    marker = folium.Marker(location = location, popup = popup)
    marker.add_to(downtown_map)

# Guardar el mapa a un archivo html
downtown_map.save("Mapa_correspondencia_Pineda.html")

