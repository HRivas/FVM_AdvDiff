#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:21:50 2018

@author: Luis Miguel De la Cruz Salas
Modificado por: H Ricardo Rivas G
"""

import numpy as np

class Mesh():
    
    def __init__(self, nodes = None, 
                     volumes = None,
                     length = None):
        """
        Constructor. Inicializa las variables del objeto
        
        nodes: Numero de nodos
        volumes: Numero de volumenes
        length: Longitud del dominio
        """
        self.__nodes = nodes
        self.__volumes = volumes
        self.__length = length     
        self.__delta = 1
        self.adjustNodesVolumes(nodes, volumes)
        self.calcDelta()
    
    def __del__(self):
        """
        Destructor. Borra las variables del objeto
        """
        del(self.__nodes)
        del(self.__volumes)
        del(self.__length)
        del(self.__delta)
        
    def adjustNodesVolumes(self, nodes, volumes):
        """
        Ajsuta el numero de nodos a partir de un volumen dado o viceversa
        
        nodes: Numero de nodos
        volumes: Numero de volumenes
        """
        if nodes:
            self.__volumes = self.__nodes + 1
        if volumes:
            self.__nodes = self.__volumes - 1
        
    def nodes(self):
        """
        Devuleve el valor del numero de nodos
        """
        return self.__nodes
    
    def setNodes(self, nodes):
        """
        Establece el numero de nodos en las variables del objeto
        
        nodes: Numero de nodos
        """
        self.__nodes = nodes
        self.adjustNodesVolumes(nodes = nodes, volumes = None)
        
    def volumes(self):
        """
        Devuleve el valor del numero de volumenes
        """
        return self.__volumes

    def setVolumes(self, volumes):
        """
        Establece el numero de volumenes en las variables del objeto
        
        volumes: Numero de volumenes
        """
        self.__volumes = volumes
        self.adjustNodesVolumes(nodes = None, volumes = volumes)
        
    def length(self):
        """
        Devuleve el valor de la longitud del dominio
        """
        return self.__length
        
    def calcDelta(self):
        """
        Calcula el valor del intervalo longitudinal (dx)
        """
        if self.__length:
            self.__delta = self.__length / (self.__nodes - 1)
        
    def delta(self):
        """
        Devuelve el valor del intervalo longitudinal (dx)
        """
        return self.__delta
    
    def createMesh(self, n = 0):
        """
        Establece el valor de la malla correspondiente al dominio de solucion
        
        n: Por default n=0, se utiliza cuando a) n = 1 se tiene una condicion de
        frontera tipo Neumann, b) n = 2 dos condiciones tipo Neumann
        """
        first_volume = self.__delta / 2
        final_volume = self.__length - first_volume
        self.__x = np.zeros(self.__volumes-n)
        self.__x[1:-1] = np.linspace(first_volume, final_volume, self.__volumes-2-n)
        self.__x[-1] = self.__length
        return self.__x
        
if __name__ == '__main__':

    m1 = Mesh()
    print(m1.nodes(), m1.volumes())
    print('_' * 20)   
    
    m1 = Mesh(nodes = 5)
    print(m1.nodes(), m1.volumes())
    print('_' * 20)
   
    m1 = Mesh(volumes = 5)
    print(m1.nodes(), m1.volumes())
    print('_' * 20)
    
    m1 = Mesh(5,5)
    print(m1.nodes(), m1.volumes())
    print('_' * 20)
    
    m1.setNodes(8)
    print(m1.nodes(), m1.volumes())
    print('_' * 20)

    m1.setVolumes(8)
    print(m1.nodes(), m1.volumes())
    print('_' * 20)
    
    m1 = Mesh(nodes =  5, length = 33)
    print(m1.nodes(), m1.volumes(), m1.length())
    print('_' * 20)
    
    m1 = Mesh(volumes =  5, length = 33)
    print(m1.nodes(), m1.volumes(), m1.length())
    print('_' * 20)
    
    m1 = Mesh(nodes = 5, length = 1)
    print(m1.nodes(), m1.volumes(), m1.length(), m1.delta())
    print('_' * 20)    
    
    m1 = Mesh(volumes = 10, length = 1)
    print(m1.nodes(), m1.volumes(), m1.length(), m1.delta())
    print('_' * 20) 

    m1 = Mesh(volumes = 6, length = 1)
    print(m1.nodes(), m1.volumes(), m1.length(), m1.delta())
    m1.createMesh()
    print('_' * 20) 
    