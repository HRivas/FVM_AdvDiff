#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:46:43 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""

import numpy as np
from Coefficients import Coefficients

class Advection1D(Coefficients):
    
    def __init__(self, nvx = None, rho = None, dx = None):
        """
        Constructor. Inicializa las variables del objeto
        
        nvx: Numero de volumenes
        rho: Coeficiente de difusividad
        dx: Intervalo longitudinal
        """
        super().__init__(nvx)
        self.__nvx = nvx
        self.__rho = rho
        self.__dx = dx
        self.__u = np.zeros(nvx-1)

    def __del__(self):
        """
        Destructor. Borra las variables del objeto
        """
        del(self.__nvx)
        del(self.__rho)
        del(self.__dx)
        del(self.__u)

    def setU(self, u):
        """
        Establece el valor de u (velocidad) en la variable del objeto
        """
        if type(u) == float:
            self.__u.fill(u)
        else:
            self.__u = u

    def u(self):
        """
        Devuelve el valor de la velocidad u
        """
        return self.__u
    
    def calcCoef(self, Approx):
        """
        Calcula los coeficientes de acuerdo al esquema de aproximacion elegido
        
        Approx: Esquema de aproximacion ('CDS'/'Upwind2'/'Quick'/Default (Upwind 1er orden))
        """
        aE = self.aE()
        aW = self.aW()
        aP = self.aP()
        aWW = self.aWW()
        aEE = self.aEE()
        u = self.__u
        rho = self.__rho

        for i in range(1,self.__nvx-1):
            Fe = rho * u[i]
            Fw = rho * u[i-1]
            if Approx == 'CDS':
            # Diferencias Centrales
                CE = - Fe * 0.5
                CW =   Fw * 0.5
                aE[i] += CE
                aW[i] += CW
                aP[i] += CE + CW + (Fe - Fw)
                
            elif Approx == 'Upwind2':
            # Upwind 2o orden
                CE = max((Fe,0))
                CW = max((Fw,0))
                CE2 = -max((-Fe,0))
                CW2 = -max((-Fw,0))
                if i > 2 and i < self.__nvx -2:
                    aE_ = -(3*CE2 + CW2)/2
                    aW_ = (CE + 3*CW)/2
                    aWW_ = -CW/2
                    aEE_ = CE2/2
                    aE[i] += aE_
                    aW[i] += aW_
                    aWW[i] += aWW_
                    aEE[i] += aEE_
                    aP[i] += aE_ + aW_ + aWW_ + aEE_ + Fe - Fw
                
            elif Approx == 'Quick':
            # Quick
                CE = max((Fe,0))
                CW = max((Fw,0))
                CE2 = -max((-Fe,0))
                CW2 = -max((-Fw,0))
                
                if i > 2 and i < self.__nvx -2:
                    aE_ = -3*CE/8 - 6*CE2/8 - CW2/8
                    aW_ = CE/8 + 6*CW/8 + 3*CW2/8
                    aE[i] += aE_
                    aW[i] += aW_
                    aWW[i] += -CW/8
                    aEE[i] += CE2/8
                    aP[i] += aE_ + aW_ + aWW[i] + aEE[i] + (Fe - Fw)                
            else:
            # Upwind
                CE = max((-Fe,0))
                CW = max((Fw,0))
                aE[i] += CE
                aW[i] += CW
                aP[i] += CE + CW + (Fe - Fw)

if __name__ == '__main__':
    nx = 6
    u = np.sin(np.linspace(0,1,nx))
    print('-' * 20)
    print(u)
    print('-' * 20)

    af1 = Advection1D(6, 1, 1)
    af1.alloc(6)
    af1.setU(u)
    print(af1.u())
    print('-' * 20)

    af1.calcCoef('CDS')
    print('Central')
    print(af1.aP(), af1.aE(), af1.aW(), af1.aEE(), af1.aWW(), af1.Su(), sep = '\n')
    print('-' * 20)
    
    af1.calcCoef('Upwind2')
    print('Central')
    print(af1.aP(), af1.aE(), af1.aW(), af1.aEE(), af1.aWW(), af1.Su(), sep = '\n')
    print('-' * 20)