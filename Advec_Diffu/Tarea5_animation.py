#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:52:42 2018

@author: H Ricardo Rivas G

Animacion de la solucion a la ecuacion de adveccion - difusion no estacionaria en una dimension
"""

import FiniteVolumeMethod as fvm
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy import special
from matplotlib.animation import FuncAnimation

# Ecuacion adveccion-difusion no estacionaria
#      dphi      d                  d           dphi
# rho*-----  +  ---( rho*u*phi ) = ---- ( Gamma ---- ) 
#      dt        dx                 dx           dx

u = 1.0         # m
rho = 1.0       # kg / m^3
dt = 2e-3       # Paso de tiempo
tmax = 1.0      # Tiempo maximo
Gamma = 1e-3    # kg/ (m * s)
phi_0 = 1
phi_L = 0
L = 2.5         # m
N = 50          # Numero de nodos
#N = 350
steps = 500     # Numero de pasos de tiempo

# Se puede seleccionar el metodo de aproximacion en las caras
#scheme = 'Upwind'
#scheme = 'CDS'
#scheme = 'Upwind2'
scheme = 'Quick'

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
x = mesh.createMesh()      # Vector de coordendas del dominio

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
              Delta = delta,
              DeltaT = dt)

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
#    inicializa valores de los coeficientes de la Advección
#  -------------------------------------------------------
adv = fvm.Advection1D(nvx = n_volx, rho = rho, dx = delta)
adv.setU(u)

#  -------------------------------------------------------
#    Calcula terminos de la parte temporal
#  -------------------------------------------------------
tem = fvm.Temporal1D(n_volx, rho = rho, dx = delta, dt = dt)

#  -------------------------------------------------------
#    Calcula coeficientes de FVM de la Advección
#  -------------------------------------------------------
adv.calcCoef(scheme)

#  -----------------------------------------------------
#     Construye arreglo para almacenar la solucion
#     considerando las condiciones de frontera
#  -----------------------------------------------------
phi = np.zeros(n_volx)             # El arreglo para almacenar solucion
phi[0]   = phi_0                   # Condición de frontera izquierda
phi[-1]  = phi_L                   # Condición de frontera derecha

def implicitMethod(i):
    time_step = i * dt
    
    #  -----------------------------------------------------
    #     Ajusta los coeficientes en cada iteracion
    #  -----------------------------------------------------    
    coeff.cleanCoefficients()       # Pone los coeficientes en ceros
    dif.calcCoef()                  # Calcula los coeficientes de la parte difusiva
    adv.setU(u)                     # Establece la velocidad u
    adv.calcCoef(scheme)            # Calcula los coeficientes de la parte advectiva utilizando el metodo de aproximacion elegido
    tem.calcCoef(phi)               # Calcula los coeficientes propios de la parte temporal
    coeff.bcDirichlet('LEFT_WALL', phi_0, scheme, rho, u)      # Actualiza los coeficientes considerando
    coeff.bcDirichlet('RIGHT_WALL', phi_L, scheme, rho, u)     # las condiciones de frontera
    
    Su = coeff.Su()                     # Vector del lado derecho
    A = fvm.Matrix(mesh.volumes())      # Matriz del sistema
    
    #  ---------------------------------------------------------------
    #    Se resuelve el sistema de ecuaciones
    #    Se puede elegir el algoritmo para calcular la solucion
    #  ---------------------------------------------------------------
    if algoritmo != 'Sparse':
        A.build(coeff)                                   # Construcción de la matriz en la memoria
        phi[1:-1] = np.linalg.solve(A.mat(), Su[1:-1])   # Se utiliza un algoritmo de linalg
    else:
        Asparse = A.buildSparse(coeff)                   # Construcción de la matriz dispersa
        phi[1:-1] = spsolve(Asparse, Su[1:-1])           # Resuelve el sistema lineal con matriz dispersa    
        
    line.set_ydata(phi)                                  # Cambia los datos en la dirección y
    label.set_text('Step = {:>8d} \n Time = {:>8.5f}'.format(i, time_step))
        
    return

#  ---------------------------------------------------------------
# Grafica de la solución exacta para t = 1.0 (500 pasos)
#  ---------------------------------------------------------------
def analyticSol(x, u, t, Gamma):
    """
    Esta funcion permite calcular la solucion analitica de la ecuacion 
    de adveccion-difusion no estacionaria con las condiciones 
    phi(x,0) = 0  para 0 <= x <= inf
    phi(0,t) = 1  para t > 0
    phi(L,t) = 0  para t > 0, L -> inf
    """
    divisor = 2 * np.sqrt(Gamma * t)
    phi = 0.5 * (special.erfc((x - u * t)/ divisor) + 
 		np.exp(u * x) * np.exp(-Gamma) * special.erfc((x + u * t)/divisor))
    return phi

fig = plt.figure(figsize=(8,4))                           # Se crea una nueva figura
ax = plt.axes(xlim=(0, 3), ylim=(0, 1.2))                 # Se crean los ejes coordenados

exac = analyticSol(x, u, dt * steps, Gamma)
ax.plot(x,exac,'b-',label='Sol. Exac',lw=2)

line, = ax.plot(x, phi, '--', label='FVM')                # Se grafica la funcion inicial
label = ax.text(2.6, 0.5, 'Time = {:>8.5f}'.format(0),    # Despliega informacion del paso de tiempo
                ha='center', va='center',fontsize=12)
title_graf = 'Solucion a la ecuacion: $p \partial \phi / \partial t + \partial(p u \phi)/\partial x= \partial (\Gamma \partial\phi/\partial x)/\partial x$' + '\n Utilizando el esquema ' + scheme
ax.set_title(title_graf)
plt.xlabel('$x$ [m]')
plt.ylabel('$\phi[...]$')
ax.grid("on")
ax.legend()

#  ---------------------------------------------------------------
# Función que controla el cálculo y la animación
#  ---------------------------------------------------------------
anim = FuncAnimation(fig,              # La figura
                     implicitMethod,   # la función que cambia los datos
                     interval=1,       # Intervalo entre cuadros en milisegundos
                     frames=steps+1,   # Cuadros
                     repeat=False)     # Permite poner la animación en un ciclo