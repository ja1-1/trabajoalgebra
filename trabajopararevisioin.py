import numpy as np
from sympy import symbols, Matrix, det, simplify, solve, Eq

# Ejercicio 1

# Definimos el valor de los vectores u, v, w, z
u = np.array([24, 1, 6])
v = np.array([19, 37, -1])
w = np.array([34, -46, 7])
z = np.array([13, -4, 28])

# a) Se calcula (u + z - v) · (z - 3w + v)
A = u + z - v
B = z - 3*w + v
resultado_a = np.dot(A, B)

# b) Se calcula ||z - w|| - ||w × u||
C = z - w
norma_C = np.linalg.norm(C)
producto_cruzado = np.cross(w, u)
norma_cruzado = np.linalg.norm(producto_cruzado)
resultado_b = norma_C - norma_cruzado

# c) Se calcula ||u||(v+z) + ||2z-3w||z
norma_u = np.linalg.norm(u)
suma_vz = v + z
producto1 = norma_u * suma_vz
expresion2 = 2*z - 3*w
norma_exp2 = np.linalg.norm(expresion2)
producto2 = norma_exp2 * z
resultado_c = producto1 + producto2

# d) Encontrar p y h tal que el valor de v sea igual a p + h, p paralelo a u, h ortogonal a u
# Descomposición paralela y ortogonal
u_sym = Matrix(u)
v_sym = Matrix(v)
lambda_proj = (v_sym.dot(u_sym)) / (u_sym.dot(u_sym))
p = np.array(lambda_proj * u, dtype=float)
h = v - p


# Ejercicio 2

# Crear la matriz de codificación A
llave = [0, 0, 1, -1, 0, 0, 4, -1, 1, 2, -2, 4, 1, 1, 2, 2, -4, 0, 0, -2, 0, 2, -1, 1, 2]
A = np.array(llave).reshape((5, 5))

# Crear la matriz codificada C
mensaje_codificado = [...]
C = np.array(mensaje_codificado).reshape((5, -1))

# Encontrar el mensaje plano B
A_inv = np.linalg.inv(A)
B = np.matmul(A_inv, C)
B_redondeado = np.round(B).astype(int) % 26
mensaje = ''.join(chr(n + ord('A')) for n in B_redondeado.flatten())

# Ejercicio 3

# Definir la variable simbólica k
def_k = symbols('k')

# Definir la matriz A(k)
A_k = Matrix([
    [-def_k, def_k-1, def_k+1, def_k, 0],
    [1, 2, -2, -1, 1],
    [4-def_k, def_k, def_k+3, -def_k, 1],
    [-2, 3, 4, 6, 2],
    [0, 2, 1, 0, 4]
])

# Se calcula determinante de A(k)
det_A_k = simplify(A_k.det())

# Resolver cuándo det(A)=0
soluciones_inversa = solve(Eq(det_A_k, 0), def_k)

# Resolver cuándo det(A)=5-3k
soluciones_det5_3k = solve(Eq(det_A_k, 5 - 3*def_k), def_k)

# Ejercicio 4

# a) Área del triángulo con puntos dados
M_a = Matrix([
    [1, -3, 1],
    [-1, 5, 1],
    [-7, -4, 1]
])
area_a = (1/2) * abs(M_a.det())

# b) Resolver para k cuando el área cuadrada es 81
M_b = Matrix([
    [def_k, -2, 1],
    [def_k, 1, 1],
    [-1, def_k, 1]
])
det_M_b = simplify(M_b.det())

# Resolver manualmente por módulo del determinante
sol1 = solve(3*def_k + 3 - 18, def_k)
sol2 = solve(3*def_k + 3 + 18, def_k)

# Ejercicio 5

# Vectores base en R^6
v1 = (1, 1, 1, -1, 0, 1)
v2 = (1, 0, -1, 0, -1, 0)
v3 = (0, 0, 0, -1, -2, 0)
v4 = (1, 1, 0, -1, 0, 1)
v5 = (0, 0, 1, 1, 1, -1)
v6 = (0, 1, 1, -1, 0, 1)

# Construir matriz con vectores
M5 = Matrix.hstack(Matrix(v1), Matrix(v2), Matrix(v3), Matrix(v4), Matrix(v5), Matrix(v6))

# Verificar el rango
rango_M5 = M5.rank()

# Resolver las coordenadas de un vector dado
v_objetivo = Matrix([7, -3, 6, 1, 4, -5])
coeficientes = M5.LUsolve(v_objetivo)

# Ejercicio 6

# a) Verificar el neutro aditivo
neutro = np.array([-2, -2])
vector_prueba = np.array([5, 33])
suma_neutro = neutro + vector_prueba + 2

# b) Se calcula 8 \odot [(-11,3) \oplus (5,-7)]
suma_interna = np.array([-11, 3]) + np.array([5, -7]) + 2
res_b = (8 + 8*suma_interna - 1)

# c) Se calcula 8 \odot (-11,3) \oplus 8 \odot (5,-7)
res_1 = 8 + 8*np.array([-11, 3]) - 1
res_2 = 8 + 8*np.array([5, -7]) - 1
suma_final = res_1 + res_2 + 2

# d) Confirmar si V es un espacio vectorial
espacio_vectorial = False
