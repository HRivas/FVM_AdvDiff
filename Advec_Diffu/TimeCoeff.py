#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:02:11 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""

import numpy as np
from Coefficients import Coefficients

class Temporal1D(Coefficients):
    """
    Esta clase agrega los terminos correspondientes a los coeficientes de la parte 
    temporal de la ecuacion de adveccion - difusion no estacionaria en 1D
    """    
    
    def __init__(self, nvx = None, rho = None, dx = None, dt = None):
        super().__init__(nvx)
        self.__nvx = nvx
        self.__rho = rho
        self.__dx = dx
        self.__dt = dt

    def __del__(self):
        del(self.__nvx)
        del(self.__rho)
        del(self.__dx)
        del(self.__dt)
    
    def dT(self):
        return self.__dt

    def calcCoef(self, phi_old):
        aP = self.aP()
        Su = self.Su()
        rho = self.__rho
        dx_dt = self.__dx / self.__dt

        for i in range(1,self.__nvx-1):
            aP[i] += rho * dx_dt
            Su[i] += phi_old[i] * dx_dt

if __name__ == '__main__':
    
    nx = 6
    phi_old = np.sin(np.linspace(0,1,nx))
    print('-' * 20)
    print(phi_old)
    print('-' * 20)

    tf1 = Temporal1D(6, 1, 1, 1)
    tf1.alloc(6)
    tf1.calcCoef(phi_old)
    print(tf1.aP())
    print('-' * 20)