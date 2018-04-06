#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:16:59 2018

@author: H Ricardo Rivas G

Este programa permite crear graficas con diferentes atributos
"""

import matplotlib.pyplot as plt
import numpy as np

def plotG(x, y, xlabel = None, ylabel = None, 
          title_graf = None, label = None, kind = None, lw = None):
    """
    Esta funcion permite graficar la funcion ingresada e incluir distintos
    atributos para modificar la grafica
    
    x: variable independiente
    y: variable dependiente
    y/xlabel: Etiqueta del eje Y/X
    title_graf: Titulo de la grafica
    label: Etiqueta de la funcion
    kind: Tipo de linea
    lw: Grosor de linea
    """
    # Se elige la opcion de acuerdo a los atributos ingresados
    if title_graf != None:
        plt.title(title_graf)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    if kind == None:
        kind = '-'
        
    if label != None and lw != None:
        plt.plot(x, y, kind, label = label, lw = lw)
        plt.legend()
    elif label != None:
        plt.plot(x, y, kind, label = label)
        plt.legend()
    elif lw != None:
        plt.plot(x, y, kind, lw = lw)
    else:
        plt.plot(x, y, kind)
    
    plt.grid('on')
    return

if __name__ == '__main__':
    x = np.linspace(0,1,1000)
    y = np.sin(2*np.pi*2e3*x)
    plotG(x, y, xlabel = 'tiempo [s]', ylabel = 'Amplitud [V]', 
          title_graf = 'Onda senoidal', label = 'Señal 1', 
          kind = '--', lw = 4)