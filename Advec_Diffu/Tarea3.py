#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:57:00 2018

@author: H Ricardo Rivas G

Solucion a la ecuacion de difusion del problema 4.3
"""

import FiniteVolumeMethod as fvm
import numpy as np
import matplotlib.pyplot as plt
import Graphics as plt2
from scipy.sparse.linalg import spsolve

# Datos del problema 4.3
# Ecuacion
#  d       dT
# ---- ( ----- ) + n^2*(T-Tinf) =  0
#  dx      dx

Ta = 100   # °C    ---- Frontera izquierda Dirichlet
Tb = 0     # °C/m  ---- Frontera derecha Neumann
Tinf = 20  # °C
n2 = 25    # 1/m^2
L = 1      # m

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
dif = fvm.Diffusion1D(n_volx, Gamma = 1, dx = delta)
dif.calcCoef()

#  -----------------------------------------------------
#     Construye arreglo para almacenar la solucion
#     considerando las condiciones de frontera
#  -----------------------------------------------------
T = np.zeros(n_volx-1)                # El arreglo de ceros
T[0]  = Ta                            # Condición de frontera izquierda
coeff.bcDirichlet('LEFT_WALL', Ta)    # Actualiza los coeficientes considerando
coeff.bcNeumman('RIGHT_WALL', Tb)     # las condiciones de frontera

#  --------------------------------------------------------
#    Se construye el sistema lineal de ecuaciones a partir 
#    de los coef. de FVM
#  --------------------------------------------------------
coeff.setSu(n2*Tinf)
coeff.setSp(-n2)
Su = coeff.Su()                   # Vector del lado derecho
A = fvm.Matrix(mesh.volumes())    # Matriz del sistema
A.build(coeff)                    # Construcción de la matriz en la memoria

#  ---------------------------------------------------------------
#    Se resuelve el sistema de ecuaciones
#  ---------------------------------------------------------------
# Se puede elegir el algoritmo para calcular la solucion
if algoritmo != 'Sparse':
    T[1:] = np.linalg.solve(A.mat(), Su[1:-1])  # Se utiliza un algoritmo de linalg
else:
    Asparse = A.buildSparse(coeff)           # Construcción de la matriz dispersa
    T[1:] = spsolve(Asparse, Su[1:-1])       # Resuelve el sistema lineal con matriz dispersa

print('Solución = {}'.format(T))
print('.'+'-'*70+'.')

#  -----------------------------------------------------
#    Se construye un vector de coordenadas del dominio
#  -----------------------------------------------------
x = mesh.createMesh(n = 1)
print(x)

#  -----------------------------------------------------
#   Calculamos la solución exacta y el error
#  -----------------------------------------------------
def analyticalSol(x):
    return (Ta - Tinf)*np.cosh(np.sqrt(n2)*(L-x))/np.cosh(np.sqrt(n2)*L)+Tinf

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
title_graf = 'Solución de $ \partial (k  \partial T/\partial x)/\partial x + n^2(T-Tinf) = 0$ con FVM'
plt2.plotG(x1, T_a, kind = "-", xlabel = '$x$ [m]', ylabel = 'T [°C]', 
           label = 'Sol. analítica', lw=2, title_graf = title_graf)
plt2.plotG(x, T, kind = '--o', xlabel = '$x$ [m]', ylabel = 'T [°C]', 
           label = 'Sol. FVM', lw=2, title_graf = title_graf)
plt.show()

# Guarda la grafica
#plt.savefig('Tarea3.svg')