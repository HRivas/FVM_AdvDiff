#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:07:27 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""

from Coefficients import Coefficients

class Diffusion1D(Coefficients):
    
    def __init__(self, nvx = None, Gamma = None, dx = None):
        """
        Constructor. Inicializa las variables del objeto
        
        nvx: Numero de volumenes
        Gamma: Coeficiente de difusividad
        dx: Intervalo longitudinal
        """
        super().__init__(nvx, dx)
        self.__nvx = nvx
        self.__Gamma = Gamma
        self.__dx = dx

    def __del__(self):
        """
        Destructor. Borra las variables del objeto
        """
        del(self.__Gamma)
        del(self.__dx)
    
    def calcCoef(self):
        """
        Añade los terminos difusivos a los coeficientes
        """
        aE = self.aE()
        aW = self.aW()
        aP = self.aP()
        
        aE += self.__Gamma / self.__dx
        aW += self.__Gamma / self.__dx
        aP += aE + aW
 
if __name__ == '__main__':
    
    df1 = Diffusion1D(5, 5, 1)
    df1.alloc(5)
    df1.calcCoef()
    df1.setSu(100)

    print('-' * 20)  
    print(df1.aP(), df1.aE(), df1.aW(), df1.Su(), sep = '\n')
    print('-' * 20)  

    df1.bcDirichlet('LEFT_WALL', 2)
    df1.bcDirichlet('RIGHT_WALL', 1)
    print(df1.aP(), df1.aE(), df1.aW(), df1.Su(), sep = '\n')
    print('-' * 20)  
