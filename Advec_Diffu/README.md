# Finite Volume Method - Advection - Diffusion

En esta carpeta se encuentran todos los archivos necesarios para ejecutar los diferentes ejercicios propuestos.

Los archivos a ejecutar son los que contienen el nombre TareaX.py

En todos estos ejercicios es posible elegir el algoritmo para solucionar el sistema de ecuaciones. Por ahora las opciones son: la que se encuentra implementada en linalg.solve que corresponde a la descomposición LU y la solución a través de la contrucción de un objeto como matriz dispersa con spsolve del paquete de scipy.

Para seleccionar qué algoritmo utilizar se encuentra la línea de:

	algoritmo = 'Opción'

donde 'Opción' puede ser 'Sparse' o 'Default (LU Decomp)'

En aquellos donde se incluye el término advectivo se puede elegir el esquema de aproximación (Tarea 4 y 5) a través de la variable scheme:

	scheme = 'Opción'

donde 'Opcion' puede ser:
  'Upwind'  --- Upwind de primer orden
  'CDS'	    --- Diferencias centrales
  'Upwind2' --- Upwind de segundo orden
  'Quick'   --- Quadratic Upstream Interpolation for Convective Kinematics

En el archivo Tarea4.py se consideran los casos:
  i)   Cuando u = 0.1 m/s y 5 nodos
  ii)  Cuando u = 2.5 m/s y 5 nodos
  iii) Cuando u = 2.5 m/s y 20 nodos

En la variable case se deberá elegir 'i', 'ii' o 'iii' para cada caso estudiado.

En el archivo Tarea5.py se consideran los casos:
	a) N = 50
	b) N = 350

Se deberá elegir en la línea correspondiente a la variable N
