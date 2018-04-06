#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:11:05 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""

import numpy as np

class Coefficients():
    """
    Esta clase define los arreglos principales para los coeficientes del
    metodo de Volumen Finito. Los arreglos son definidos como variables de
    clase para que sean compartidos por todos los objetos de esta clase.
    """    
    __aP = None
    __aE = None
    __aW = None
    __aWW = None
    __aEE = None
    __Su = None
    __nvx = None
    __delta = None

    def __init__(self, nvx = None, delta = None):
        Coefficients.__nvx = nvx
        Coefficients.__delta = delta

    @staticmethod
    def alloc(n):
        if Coefficients.__nvx:
            nvx = Coefficients.__nvx
        else:
            nvx = n
        Coefficients.__aP = np.zeros(nvx)
        Coefficients.__aE = np.zeros(nvx)
        Coefficients.__aW = np.zeros(nvx)
        Coefficients.__aWW = np.zeros(nvx)
        Coefficients.__aEE = np.zeros(nvx)
        Coefficients.__Su = np.zeros(nvx)
    
    def cleanCoefficients(self):
        Coefficients.__aP[:] = 0.0
        Coefficients.__aE[:] = 0.0
        Coefficients.__aW[:] = 0.0
        Coefficients.__aEE[:] = 0.0
        Coefficients.__aWW[:] = 0.0
        Coefficients.__Su[:] = 0.0
        
    def setVolumes(self, nvx):
        Coefficients.__nvx = nvx
        
    def setDelta(self, delta):
        Coefficients.__delta = delta
        
    def setSu(self, q):
        Su = Coefficients.__Su
        dx = Coefficients.__delta
        Su += q * dx
        
    def setSp(self, Sp):
        aP = Coefficients.__aP
        dx = Coefficients.__delta
        aP -= Sp * dx
        
    def aP(self):
        return Coefficients.__aP

    def aE(self):
        return Coefficients.__aE
    
    def aW(self):
        return Coefficients.__aW
    
    def aWW(self):
        return Coefficients.__aWW
    
    def aEE(self):
        return Coefficients.__aEE
    
    def Su(self):
        return Coefficients.__Su

    @staticmethod
    def bcDirichlet(wall, phi_B, method = None, rho = None, u = None):
        aP = Coefficients.__aP
        aE = Coefficients.__aE
        aW = Coefficients.__aW
        aWW = Coefficients.__aWW
        aEE = Coefficients.__aEE
        Su = Coefficients.__Su

        if method == None or method == 'Upwind' or method == 'CDS':
            if wall == 'LEFT_WALL':
                aP[1] += aW[1]
                Su[1] += 2 * aW[1] * phi_B
            elif wall == 'RIGHT_WALL':
                aP[-2] += aE[-2]
                Su[-2] += 2 * aE[-2] * phi_B
        elif method == 'Quick':
            Fe = rho*u
            Fw = rho*u
            CE = max((Fe,0))
            CW = max((Fw,0))
            CE2 = -max((-Fe,0))
            CW2 = -max((-Fw,0))
            D = aE[0]
            
            if wall == 'LEFT_WALL':
                # First volume
                aE_ = - 3*CE/8 + D/3
                aW_ = 0
                Sp = -(2*CE/8 + CW + 8*D/3)
                aE[1] += aE_
                Su[1] += (8*D/3 + 2*CE/8 + CW) * phi_B
                aP[1] += aE_ - aW[1] + Fe - Fw - Sp
                
                # Second volume
                aE_ = - 3*CE/8
                aW_ = CE/8 + 7*CW/8
                Sp = CW/4
                aE[2] += aE_
                aW[2] += aW_
                Su[2] += -2*CW/8 * phi_B
                aP[2] += aE_ + aW_ + Fe - Fw - Sp
                
            if wall == 'RIGHT_WALL':
                # Last volume
                aE_ = 0                
                aW_ = 6*CW/8 + D/3
                aWW_ = -CW/8
                Sp = CE - 8*D/3
                aE[-2] += aE_
                aW[-2] += aW_
                aWW[-2] += aWW_
                Su[-2] += (8*D/3-CE)*phi_B
                aP[-2] += -aE[-2] + aW_ + aWW_ + Fe - Fw - Sp
                
        elif method == 'Upwind2':
            Fe = rho*u
            Fw = rho*u
            CE = max((Fe,0))
            CW = max((Fw,0))
            CE2 = -max((-Fe,0))
            CW2 = -max((-Fw,0))
            D = aE[0]
            
            if wall == 'LEFT_WALL':
                # First volume
                aE_ = 0
                aW_ = 0
                Sp = -(CE + CW + D)
                aW[1] += -D
                Su[1] += (CE + CW + 2*D) * phi_B
                aP[1] += aE_ + aW_ + (Fe - Fw) - Sp
                
                # Second volume
                aE_ = 0
                aW_ = CE/2 + 2*CW
                Sp = CW
                aW[2] += aW_
                Su[2] += -CW * phi_B
                aP[2] += aE_ + aW_ + (Fe - Fw) - Sp
                
            if wall == 'RIGHT_WALL':
                # Last volume
                aE_ = 0
                aW_ = 3*CW/2
                aWW_ = -CW/2
                Sp = CE - D
                aE[-2] += -D
                aW[-2] += aW_
                aWW[-2] += aWW_
                Su[-2] += (D - CE) * phi_B
                aP[-2] += aE_ + aW_ + aWW_ + (Fe - Fw) - Sp
        
    @staticmethod
    def bcNeumman(wall, flux):
        aP = Coefficients.__aP
        aE = Coefficients.__aE
        aW = Coefficients.__aW
        aWW = Coefficients.__aWW
        aEE = Coefficients.__aEE
        Su = Coefficients.__Su
        dx = Coefficients.__delta

        if wall == 'LEFT_WALL':
            aP[1] -= aW[1]
            Su[1] -= aW[1] * flux * dx
        elif wall == 'RIGHT_WALL':
            aP[-2] -= aE[-2]
            Su[-2] += aE[-2] * flux * dx
        

if __name__ == '__main__':
    
    coef1 = Coefficients(6, 0.25)
    coef1.alloc(6)
    coef1.setSu(100)
    coef1.setSp(-2)
    
    print('-' * 20)  
    print(coef1.aP(), coef1.aE(), coef1.aW(), coef1.Su(), sep = '\n')
    print('-' * 20)  

    ap = coef1.aP()
    ap[2] = 25
    print(ap, coef1.aP(),sep='\n')
    print('-' * 20)  

    ae = coef1.aE()
    aw = coef1.aW()
    su = coef1.Su()
    ae.fill(5)
    aw.fill(5)
    ap.fill(10)
    coef1.setSp(-2)
    coef1.bcDirichlet('LEFT_WALL', 2)
    coef1.bcNeumman('RIGHT_WALL', 1)
    print(coef1.aP(), coef1.aE(), coef1.aW(), coef1.Su(), sep = '\n')
    print('-' * 20)  