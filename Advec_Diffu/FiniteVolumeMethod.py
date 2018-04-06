#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:21:50 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""

import numpy as np
from pandas import DataFrame
from Mesh import Mesh
from Coefficients import Coefficients
from Diffusion import Diffusion1D
from Advection import Advection1D
from TimeCoeff import Temporal1D
from Matrix import Matrix
import time

def crono(f):
 	"""
 	Devuelve el tiempo que tarda en ejecutarse una funcion f.
 	"""
 	def eTime(A,b):
 		t1 = time.time()
 		f(A,b)
 		t2 = time.time()
 		return 'Elapsed time: ' + str((t2 - t1)) + "\n"
 	return eTime

def decorate(f):
    """
    Dercorador de funcion f
    Agrega texto antes y despues del resultado de la funcion
    """
    def nicePrint(**kargs):
        line = '-' * 70
        print('.'+ line + '.')
        print('|{:^70}|'.format('Tarea 1. Computo Cientifico con Alto Valor Agregado'))
        print('.'+ line + '.')
        print('|{:^70}|'.format('Ricardo Rivas, 2018'))
        print('.'+ line + '.')
        f(**kargs)
        print('.'+ line + '.')
    return nicePrint
 
@decorate
def printData(**kargs):
    """
    Imprime los datos del problema
    
    **kargs: dato = valor
    """
    for (key,value) in kargs.items():
        print('|{:^70}|'.format('{0:>15s} = {1:10.5e}'.format(key, value)))

def printFrame(d):
    """ 
    Calcula el error porcentual y agrega al DataFrame
    una columna con esos datos llamada 'Error %'
    
    d = Diccionario con los datos del resultado obtenido y el valor exacto de la solucion
    """
    for i in range(0,len(d['Analytic'])):
        if d['Analytic'][i] == 0:
            d['Analytic'][i] = 1e-200
    d['Error %'] = d['Error'] / d['Analytic']
    print( DataFrame(d) )
    print('.'+ '-'*70 + '.')

def calcError(phiA, phiN):
    """
    Calcula el error absoluto
    
    phiA: Solucion analitica
    phi: Solucion con el metodo FVM
    """
    return np.absolute(phiA - phiN)
        
if __name__ == '__main__':
 
    Coefficients.alloc(5)
    m = Mesh(nodes = 5)
    d = Diffusion1D(m.volumes())
    ma = Matrix(m.volumes())
    a = Advection1D(m.volumes())
    t = Temporal1D(6, 1, 1, 1)

    print(m.delta(), d.aP(), a.aP(), ma.mat(), t.aP(), sep='\n')

    printData(nvx =5, nx = 6, longitud = 1.3)


def grafica(x, phi, title = None, label = None, kind = None):
    plt.title(title)
    plt.xlabel('$x$ [m]')
    plt.ylabel('$\phi$ [...]')
    if kind:
        plt.plot(x,phi,kind,label=label,lw=2)
    else:
        plt.plot(x,phi,'--', label = label)