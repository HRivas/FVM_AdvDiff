#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 00:53:34 2018

@author: H Ricardo Rivas G

Solucion a la ecuacion de adveccion - difusion estacionaria en una dimension
"""

import FiniteVolumeMethod as fvm
import numpy as np
import matplotlib.pyplot as plt
import Graphics as plt2
from scipy.sparse.linalg import spsolve
import time

# Datos del problema 5.1
# Ecuacion
#  d                    d          dphi
# ----( rho*u*phi ) = ---- ( Gamma ---- ) 
#  dx                  dx           dx

phi_0 = 1   # Frontera izquierda Dirichlet
phi_L = 0   # Frontera derecha Dirichlet
rho = 1.0   # kg/m^3
Gamma = 0.1 # kg/m*s
u = 0.2     # m/s Case i
#u = 2.5     # m/s Case ii, iii
L = 1       # m

N = 7       # Numero de nodos
#N = 20      # iii

# Se puede seleccionar el metodo de aproximacion en las caras
#method = 'Upwind'
#method = 'CDS'
#method = 'Upwind2'
method = 'Quick'

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
              Densidad = rho,
              Velocidad = u,
              Coef_Diff = Gamma,
              Prop_0 = phi_0,
              Prop_L = phi_L,
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
dif = fvm.Diffusion1D(n_volx, Gamma = Gamma, dx = delta)
dif.calcCoef()

#  -------------------------------------------------------
#    Calcula coeficientes de FVM de la Advección
#  -------------------------------------------------------
adv = fvm.Advection1D(nvx = n_volx, rho = rho, dx = delta)
adv.setU(u)
adv.calcCoef(method)

#  -----------------------------------------------------
#     Construye arreglo para almacenar la solucion
#     considerando las condiciones de frontera
#  -----------------------------------------------------
phi = np.zeros(n_volx)                  # El arreglo para almacenar solucion
phi[0]   = phi_0                        # Condición de frontera izquierda
phi[-1]  = phi_L                        # Condición de frontera derecha
coeff.bcDirichlet('LEFT_WALL', phi_0, method, rho, u)      # Actualiza los coeficientes considerando
coeff.bcDirichlet('RIGHT_WALL', phi_L, method, rho, u)     # las condiciones de frontera

#  --------------------------------------------------------
#    Se construye el sistema lineal de ecuaciones a partir 
#    de los coef. de FVM
#  --------------------------------------------------------
Su = coeff.Su()                   # Vector del lado derecho
A = fvm.Matrix(mesh.volumes())    # Matriz del sistema

#  ---------------------------------------------------------------
#    Se resuelve el sistema de ecuaciones
#  ---------------------------------------------------------------
# Se puede elegir el algoritmo para calcular la solucion
start = time.time()
if algoritmo != 'Sparse':
    A.build(coeff)                                   # Construcción de la matriz en la memoria
    phi[1:-1] = np.linalg.solve(A.mat(), Su[1:-1])   # Se utiliza un algoritmo de linalg
else:
    Asparse = A.buildSparse(coeff)            # Construcción de la matriz dispersa
    phi[1:-1] = spsolve(Asparse, Su[1:-1])    # Resuelve el sistema lineal con matriz dispersa
end = time.time()
print("hello")
print(end - start)

print('Solución = {}'.format(phi))
print('.'+'-'*70+'.')

#  -----------------------------------------------------
#    Se construye un vector de coordenadas del dominio
#  -----------------------------------------------------
x = mesh.createMesh()

#  -----------------------------------------------------
#   Calculamos la solución exacta y el error
#  -----------------------------------------------------
def analyticalSol(x):
    """
    Esta funcion permite calcular la solucion analitica de la ecuacion 
    de adveccion-difusion estacionaria con las condiciones 
    phi(0) = 1
    phi(L) = 0
    """
    return (phi_L - phi_0)*(np.exp(rho*u*x/Gamma) - 1)/(np.exp(rho*u*L/Gamma) - 1) + phi_0

phi_a = analyticalSol(x)
error = fvm.calcError(phi, phi_a)
datos = {'x(m)': x,
         'phi(x)': phi,
         'Analytic': phi_a,
         'Error': error}
fvm.printFrame(datos)
print('||Error|| = ', np.linalg.norm(error))
print('.'+ '-'*70 + '.')

#  -------------------------------------------------------------------
#   Calculamos la solución exacta en una malla más fina para graficar
#  -------------------------------------------------------------------
x1 = np.linspace(0, L, 100)
phi_a = analyticalSol(x1)

#  -----------------------------------------------------
#    Se grafica la solución
#  -----------------------------------------------------
plt.close('all')
title_graf = 'Solución de $\partial(p u \phi)/\partial x= \partial (\Gamma \partial\phi/\partial x)/\partial x$ con FVM'
plt2.plotG(x, phi, kind = "--o", xlabel = '$x$ [m]', ylabel = '$\phi[...]$', 
           label = 'Sol. FVM', title_graf = title_graf)
plt2.plotG(x1, phi_a, kind = "-", xlabel = '$x$ [m]', ylabel = '$\phi[...]$', 
           label = 'Sol. analítica', lw=2, title_graf = title_graf)
##plt.savefig('example04.pdf')
plt.show()