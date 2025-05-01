# Laboratorio Álgebra

Este laboratorio incluye una serie de ejercicios enfocados en operaciones vectoriales, criptografía, determinantes simbólicos, áreas de triángulos, cambio de base y estructuras algebraicas. Se utilizan herramientas como NumPy y SymPy para resolver los problemas de manera computacional.

---

## Ejercicio 1 – Operaciones Vectoriales

Se emplean funciones de **NumPy** como:

- `np.dot` para el producto escalar
- `np.cross` para el producto cruzado
- `np.linalg.norm` para la norma de un vector

También se utiliza `Matrix` de **SymPy** para realizar una **descomposición ortogonal** de vectores.

---

## Ejercicio 2 – Criptografía con Matrices

Se trabaja con el **método de Hill** usando matrices. Se utilizan:

- `np.linalg.inv` para invertir la matriz de codificación
- `np.matmul` para descifrar el mensaje
- `np.round` y el operador módulo 27 (`% 27`) para redondear y mapear los resultados a letras del alfabeto.

---

## Ejercicio 3 – Determinante Simbólico

Se usa `Matrix.det()` de **SymPy** junto con `simplify()` para obtener el determinante de una matriz simbólica. Luego se emplea `solve()` con `Eq()` para **resolver ecuaciones en función de un parámetro** \( k \).

---

## Ejercicio 4 – Área de un Triángulo

Se aplica la fórmula del área mediante determinantes usando `Matrix.det()`. En el inciso b, se plantea una **ecuación cuadrática** con `solve()` para encontrar los valores de \( k \) que generan un área específica.

---

## Ejercicio 5 – Coordenadas en Base

- Se construye una matriz con `Matrix.hstack`
- Se determina el rango con `Matrix.rank()`
- Se resuelve un sistema lineal con `LUsolve()` para **encontrar las coordenadas de un vector en una base dada**.

---

## Ejercicio 6 – Estructura Algebraica

Se realizan operaciones personalizadas con **NumPy** simulando definiciones alternativas de suma y producto escalar. Se verifica si se cumple la definición de **espacio vectorial** a través de pruebas básicas.

---
