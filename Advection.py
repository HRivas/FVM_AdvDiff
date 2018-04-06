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
        super().__init__(nvx)
        self.__nvx = nvx
        self.__rho = rho
        self.__dx = dx
        self.__u = np.zeros(nvx-1)

    def __del__(self):
        del(self.__nvx)
        del(self.__rho)
        del(self.__dx)
        del(self.__u)

    def setU(self, u):
        if type(u) == float:
            self.__u.fill(u)
        else:
            self.__u = u

    def u(self):
        return self.__u
    
    def calcCoef(self, Approx):
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
#                    
#            elif Approx == 'QuickM':
#            # Quick
#                if Fe > 0:
#                    ae = 1
#                    aw = 1
#                else:
#                    ae = 0
#                    aw = 0
#                CE = -3*ae*Fe/8 - 6*(1-ae)*Fe/8 - (1-aw)*Fw/8
#                CW = 6*aw*Fw/8 + ae*Fe/8 + 3*(1-aw)*Fw/8
#                CWW = -aw*Fw/8
#                CEE = (1-ae)*Fe/8
#                aE[i] += CE
#                aW[i] += CW
#                aWW[i] += CWW
#                aEE[i] += CEE
#                aP[i] += CW + CE + CWW + CEE + (Fe-Fw)
                
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
#    u = np.ones(nx)
    print('-' * 20)
    print(u)
    print('-' * 20)

    af1 = Advection1D(6, 1, 1)
    af1.alloc(6)
    af1.setU(u)
    print(af1.u())
    print('-' * 20)

#    af1.calcCoef('Central')
#    print('Central')
#    print(af1.aP(), af1.aE(), af1.aW(), af1.aEE(), af1.aWW(), af1.Su(), sep = '\n')
#    print('-' * 20)
    
    af1.calcCoef('Upwind2')
    print('Central')
    print(af1.aP(), af1.aE(), af1.aW(), af1.aEE(), af1.aWW(), af1.Su(), sep = '\n')
    print('-' * 20)

#    af1.bcDirichlet('LEFT_WALL', 20)
#    af1.bcDirichlet('RIGHT_WALL', 10)
#    print(af1.aP(), af1.aE(), af1.aW(), af1.aEE(), af1.aWW(), af1.Su(), sep = '\n')
#    print('-' * 20)