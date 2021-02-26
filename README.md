# 2-Tensor Basics

1. [Introudcción](#schema1)
2. [Empty](#schema2)
3. [Rand, ones, zeros](#schema3)
4. [Tamaño, Tipos de datos, construir con datos](#schema4)
5. [Operaciones básicas](#schema5)

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