# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:15:28 2020

@author: ASUS
"""
#librerias a usar para la generación de visualizaciones de redes
import holoviews as hv
import networkx as nx
from holoviews import opts
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from operator import itemgetter
from bokeh.io import output_file, show
#especificar la extensión de holoviews para lograr la visualización interactiva
hv.extension('bokeh')
#cargar el dataset
df=pd.read_excel('Pineda.xlsx')

#convertir a formato temporal los valores de años en el dataset
time_format = '%Y'
df['Año']=pd.to_datetime(df['Año'], format=time_format)

# Inicializar el grafo: G
G = nx.Graph()

# añadir los nodos tanto como remitentes como destinatarios
G.add_nodes_from(df['Remitente'], bipartite='Remitente')
G.add_nodes_from(df['Destinatario'],bipartite='Destinatario')
#mostrar el número total de nodos
print('Número de nodos en la red de Anselmo Pineda: ',len(G.nodes()))
#añadir los vínculos entre los nodos incluyendo la información de años como metadato de cada vínculo
for r, d in df.iterrows():
   G.add_edge(d['Remitente'], d['Destinatario'],date=d['Año'])
   
#generar un subgrafo para dividir la red a partir de un marco temporal
graph_sub = nx.Graph()
#agregar al subgrafo los vínculos entre nodos según el marco temporal definido como condición
graph_sub.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if d['date'] > datetime(1805, 1,1) and d['date'] < datetime(1829, 1,1)])
#calcular el grado de centralidad de cada nodo en el subgrafo
node_and_degree_sub = graph_sub.degree()
#de la variable anterior obtener el nodo de mayor centralidad y adjuntar a cada nodo su grado de centralidad
(largest_hub_sub, degree_sub) = sorted(node_and_degree_sub, key=itemgetter(1))[-1]
for n, d in graph_sub.nodes(data=True):
    graph_sub.nodes[n]['dc'] = node_and_degree_sub[n]
    
#definir el estilo de gráfica con los parámetros que mejor se ajustan dada la masa de nodos y en observancia de la calidad gráfica

pos = nx.spring_layout(graph_sub, k=0.5,scale=15.0,iterations=100)
#generar el grafico en holoviews a partir del grafo producido en networkx
grafo=hv.Graph.from_networkx(graph_sub, pos).opts(tools=['hover','box_select'],title="Red epistolar 1806-1828",fontsize={'title': 16, 'labels': 14},bgcolor='lightgray',node_size=10, edge_line_width=1,node_line_color='darkgray',edge_hover_line_color='yellow', node_hover_fill_color='red',node_color='dc',cmap='prism'])
#guardar el grafo en un archivo html
output_file("graph.html")
#mostrar el grafo
show(grafo)

