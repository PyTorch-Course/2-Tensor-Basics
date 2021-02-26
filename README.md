# 2-Tensor Basics

1. [Introudcción](#schema1)
2. [Empty](#schema2)
3. [Rand, ones, zeros](#schema3)
4. [Tamaño, Tipos de datos, construir con datos](#schema4)
5. [Operaciones básicas](#schema5)
6. [Slicing](#schema6)
7. [Reshape con torch.view()](#schema7)
8. [Convertir a numpy y viceversa](#schema8)

<hr>

<a name="schema1"></a>

# 1. Introducción

Un tensor de PyTorch es básicamente lo mismo que una matriz numpy,  solo una matriz n-dimensional genérica que se utiliza para cálculos numéricos arbitrarios. ... Para ejecutar operaciones en la GPU, simplemente transmita el tensor a un tipo de datos cuda

<hr>

<a name="schema2"></a>

# 2. Empty

- 1 solo número
~~~python
x = torch.empty(1)
~~~
- Vector de 3 componentes
~~~python
y = torch.empty(3) 
~~~
- Matriz de 2D con tres elementos
~~~python
z = torch.empty(2,3) 
~~~

<hr>

<a name="schema3"></a>

# 3. Rand, ones, zeros


- rellena con números aleatorios
~~~python
x = torch.rand(5,3)
~~~
- Rellena con 1
~~~python
x = torch.ones(5,3)
~~~
- Rellena con 0
~~~python
x = torch.zeros(5,3)
~~~

<hr>

<a name="schema4"></a>

# 4. Tamaño, Tipos de datos, construir con datos

- Obtenemos el tamaño
~~~python
x.size()
~~~
- Obtenemos el tipo de datos
~~~python
x.dtype
~~~
- Especificamos el tipo de datos
~~~python
x = torch.rand(5, 3, dtype=torch.float16)
~~~
- Construyendo con datos
~~~python
x = torch.tensor([5.5, 3])
~~~

<hr>

<a name="schema5"></a>

# 5. Operaciones básicas
Toda función que en pytorch lleve `_` aplica la operación a la 1º varialbe con la que está entre `()`

~~~python
y = torch.rand(2, 2)
x = torch.rand(2, 2)
~~~
- Suma
~~~python
z = x + y
z = torch.add(x,y)
y.add_(x)
~~~

- Resta
~~~python
z = x - y
z = torch.sub(x,y)
y.sub_(x)
~~~

- Multiplicación
~~~python
z = x * y
z = torch.mul(x,y)
y.mul_(x)
~~~
- Division
~~~python
z = x / y
z = torch.div(x,y)
y.div(x)
~~~

<hr>

<a name="schema6"></a>

# 6. Slicing
~~~python
x = torch.rand(5,3)
~~~
- Todas las filas de la columna 0
~~~python
x[:, 0]
~~~

- Fila 1 y todas las columnas
~~~python
x[1, :]
~~~
- El elemento 1,1
~~~python
x[1,1]
~~~
- Obtenerl el valor actual, pero solo si hay un solo elemento
~~~python
print(x[1,1].item())
~~~

<hr>

<a name="schema7"></a>

# 7  Reshape con torch.view()

~~~python

x = torch.randn(4, 4)
y = x.view(16)
~~~
- El tamaño -1 se infiere de otras dimensiones,  -1 it pytorch determinará automáticamente el tamaño necesario
~~~python
z = x.view(-1, 8)  
~~~


<hr>

<a name="schema8"></a>

# 8. Convertir a numpy y viceversa


~~~python
a = torch.ones(5)
~~~

* De torch a numpy con .numpy()
~~~python    
b = a.numpy()
~~~
Hay que tener cuidado porque tensor esta en la CPU ( no en la GPU), por lo tanto ambos objetos estan en la misma localización de memoria.  Si se cambia uno, cambia el otro
~~~python
a.add_(1)
~~~


- De numpy a torch con .from_numpy(x)
~~~python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
~~~

También hay que tener cuidado
~~~python
a += 1
~~~