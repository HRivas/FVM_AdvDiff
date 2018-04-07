#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:03:22 2018

@author: H Ricardo Rivas G

Solucion a la ecuacion de difusion del problema 4.2
"""

import FiniteVolumeMethod as fvm
import numpy as np
import matplotlib.pyplot as plt
import Graphics as plt2
from scipy.sparse.linalg import spsolve

# Datos del problema 4.2
# Ecuacion
#  d       dT
# ---- (k ----) + q =  0
#  dx      dx

Ta = 100   # °C  ---- Frontera izquierda
Tb = 200   # °C  ---- Frontera derecha
k = 0.5    # W/mK
L = 0.02   # m
A = 10e-3  # m^2
q = 1000e3 # W/m^3 --- Fuente constante

N = 5      # Numero de nodos

# Se puede seleccionar el algoritmo de solucion del sistema de ecuaciones
#algoritmo = 'Default (LU Decomp)'
algoritmo = 'Sparse'

# -------------------------------------
#    Se crea la malla
# -------------------------------------
N = N + 1
mesh = fvm.Mesh(nodes = N, length = L)
n_x = mesh.nodes()         # Numero de nodos
n_volx = mesh.volumes()    # Numero de volumenes
delta = mesh.delta()       # Tamaño de volumenes

#  -------------------------------------
#    Imprimir Datos del problema
#  -------------------------------------
fvm.printData(Longitud = L,
              Coef_Conductividad = k,
              Frontera_Izquierda = Ta,
              Frontera_Derecha = Tb,
              Nodos = n_x, 
              Volúmenes = n_volx,
              Delta = delta)

#  -----------------------------------------
#   Reserva memoria para los coeficientes
#  -----------------------------------------
coeff = fvm.Coefficients()
coeff.alloc(n_volx)
#fvm.Coefficients.alloc(nvx)

#  -----------------------------------------------------
#    Calcula coeficientes FVM de la Difusión
#  -----------------------------------------------------
dif = fvm.Diffusion1D(n_volx, Gamma = k, dx = delta)
dif.calcCoef()

#  -----------------------------------------------------
#     Construye arreglo para almacenar la solucion
#     considerando las condiciones de frontera
#  -----------------------------------------------------
T = np.zeros(n_volx)                    # El arreglo de ceros
T[0]  = Ta                           # Condición de frontera izquierda
T[-1] = Tb                           # Condición de frontera derecha
coeff.bcDirichlet('LEFT_WALL', Ta)    # Actualiza los coeficientes considerando
coeff.bcDirichlet('RIGHT_WALL', Tb)   # las condiciones de frontera

#  --------------------------------------------------------
#    Se construye el sistema lineal de ecuaciones a partir 
#    de los coef. de FVM
#  --------------------------------------------------------
coeff.setSu(q)
Su = coeff.Su()                   # Vector del lado derecho
A = fvm.Matrix(mesh.volumes())   # Matriz del sistema
A.build(coeff)                    # Construcción de la matriz en la memoria

#  ---------------------------------------------------------------
#    Se resuelve el sistema de ecuaciones
#  ---------------------------------------------------------------
if algoritmo != 'Sparse':
    T[1:-1] = np.linalg.solve(A.mat(), Su[1:-1])
else:
    Asparse = A.buildSparse(coeff)           # Construcción de la matriz dispersa
    T[1:-1] = spsolve(Asparse, Su[1:-1])    # Resuelve el sistema lineal con matriz dispersa

print('Solución = {}'.format(T))
print('.'+'-'*70+'.')

#  -----------------------------------------------------
#    Se construye un vector de coordenadas del dominio
#  -----------------------------------------------------
x = mesh.createMesh()

#  -----------------------------------------------------
#   Calculamos la solución exacta y el error
#  -----------------------------------------------------
def analyticalSol(x):
    return ((Tb-Ta)/L+q*(L-x)/(2*k))*x+Ta

T_a = analyticalSol(x)
error = fvm.calcError(T, T_a)
datos = {'x(m)': x,
         'T(x)': T,
         'Analytic': T_a,
         'Error': error}
fvm.printFrame(datos)
print('||Error|| = ', np.linalg.norm(error))
print('.'+ '-'*70 + '.')

#  -------------------------------------------------------------------
#   Calculamos la solución exacta en una malla más fina para graficar
#  -------------------------------------------------------------------
x1 = np.linspace(0,L,100)
T_a = analyticalSol(x1)

#  -----------------------------------------------------
#    Se grafica la solución
#  -----------------------------------------------------
plt.close('all')
title_graf = 'Solución de $ \partial (k  \partial T/\partial x)/\partial x + q= 0$ con FVM'
plt2.plotG(x, T, kind = '--o', xlabel = '$x$ [m]', ylabel = 'T [°C]', 
           label = 'Sol. FVM', title_graf = title_graf)
plt2.plotG(x1, T_a, kind = "-", xlabel = '$x$ [m]', ylabel = 'T [°C]', 
           label = 'Sol. analítica', lw=2, title_graf = title_graf)
plt.show()

# Guarda la grafica
#plt.savefig('Tarea2.svg')