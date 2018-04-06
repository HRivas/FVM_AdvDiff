#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:15:28 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""
import numpy as np
from scipy.sparse import csr_matrix

class Matrix():
    
    def __init__(self, nvx = None):
        """
        Constructor. Inicializa las variables del objeto
        
        nvx: Numero de volumenes
        """
        self.__N = nvx - 2 
        self.__A = np.eye(self.__N)

    def __del__(self):
        """
        Destructor. Borra las variables del objeto
        """
        del(self.__N)
        del(self.__A)
        
    def mat(self):
        """
        Devuelve la representacion de la matriz
        """
        return self.__A
    
    def build(self, coefficients = None):
        """
        Construye la matriz a partir de los coeficientes dados
        
        coefficients: Objeco de la clase coefficients que contiene el valor de
        los coeficientes aX
        """
    # nx = 5, nvx = 6
    # 0     1     2     3     4     5  <-- Volumes
    # o--|--x--|--x--|--x--|--x--|--o
    #       0     1     2     3        <-- Unknowns    
    #
    #        0   1   2   3
    # --+---------------------
    # 0 | [[12. -4.  0.  0.]
    # 1 | [ -4.  8. -4.  0.]
    # 2 | [  0. -4.  8. -4.]
    # 3 | [  0.  0. -4. 12.]]

        aP = coefficients.aP()
        aE = coefficients.aE()
        aW = coefficients.aW()
        aWW = coefficients.aWW()
        aEE = coefficients.aEE()
        A = self.__A
        A[0][0] = aP[1]
        A[0][1] = -aE[1]
        A[0][2] = -aEE[1]
        for i in range(1,self.__N-1): # range(1,N-3)  <-- (1,2)
            A[i][i] = aP[i+1]
            A[i][i+1] = -aE[i+1]
            A[i][i-1] = -aW[i+1]
            if i < self.__N-2:
                A[i][i+2] = -aEE[i+1]
            if i > 1:
                A[i][i-2] = -aWW[i+1]
        A[-1][-1] = aP[-2]
        A[-1][-2] = -aW[-2]
        A[-1][-3] = -aWW[-2]
        
    def buildSparse(self, coefficients = None):
        """
        Construye el objeto de matriz dispersa en el formato CSR (Compressed
        Sparse Row)
        
        coefficients: Objeco de la clase coefficients que contiene el valor de
        los coeficientes aX
        """
        # Sparse Matrix
        # Coeficientes
        aP = coefficients.aP()
        aE = coefficients.aE()
        aW = coefficients.aW()
        aEE = coefficients.aEE()
        aWW = coefficients.aWW()
        
        # Almacena en un arreglo los datos distintos de cero
        N = self.__N
        data = np.zeros(N*5-6)
        data[0:3] = [aP[1], -aE[1], -aEE[1]]
        data[3:7] = [-aW[2], aP[2], -aE[2], -aEE[2]]
        
        data[-3:] = [-aWW[-2], -aW[-2], aP[-2]]
        data[-7:-3] = [-aWW[-3], -aW[-3], aP[-3], -aE[-3]]

        j = 2
        for i in range(0,N-4):
            data[5*i+7:5*i+7+5] = (-aWW[j+1], -aW[j+1], aP[j+1], -aE[j+1], -aEE[j+1])
            j += 1
        
        # Almacena en un arreglo los apuntadores correspondientes a cada fila en
        # la matriz a partir del arreglo de datos
        indptr = np.zeros(N+1)
        indptr[1] = 3
        indptr[-2] = len(data)-3
        indptr[-1] = len(data)
        for i in range(0,N-3):
            indptr[i+2] = 7+5*i
        
        # Almacena los indices que establecen a que columna corresponde cada coeficiente
        # en la matriz
        indices = np.zeros(N*5-6)
        indices[:3] = np.arange(0,3)
        indices[3:7] = np.arange(0,4)
        indices[-3:] = np.arange(N-3,N)
        indices[-7:-3] = np.arange(N-4,N)
        
        for i in range(0,N-4):
            indices[5*i+7:5*i+7+5] = (0+i, 1+i, 2+i, 3+i, 4+i)
    
        # Construye el objeto CSR
        Asp = csr_matrix((data, indices.astype(int), indptr.astype(int)))
        return Asp

if __name__ == '__main__':
    n = 9
    a = Matrix(n)
    print('-' * 20)  
    print(a.mat())
    print('-' * 20)  
    
    from Diffusion import Diffusion1D        
    df1 = Diffusion1D(n, 1, 0.25)
    df1.alloc(n)
    df1.calcCoef()
    df1.setSu(100)
    print(df1.aP(), df1.aE(), df1.aW(), df1.Su(), sep = '\n')
    print('-' * 20)  

    df1.bcDirichlet('LEFT_WALL', 2)
    df1.bcDirichlet('RIGHT_WALL', 1)
    print(df1.aP(), df1.aE(), df1.aW(), df1.Su(), sep = '\n')
    print('-' * 20)  

    a.build(df1)
    print(a.mat())
    print('-' * 20)
    
    As = a.buildSparse(df1)
    print(As.toarray())
    print('-' * 20)