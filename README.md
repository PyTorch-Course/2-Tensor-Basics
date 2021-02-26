# 2-Tensor Basics

1. [Introudcción](#schema1)
2. [Empty](#schema2)

<hr>

<a name="schema1"></a>

# 1. Introducción

Un tensor de PyTorch es básicamente lo mismo que una matriz numpy,  solo una matriz n-dimensional genérica que se utiliza para cálculos numéricos arbitrarios. ... Para ejecutar operaciones en la GPU, simplemente transmita el tensor a un tipo de datos cuda

<hr>

<a name="schema2"></a>

# Empty

#### torch.empty(tamaño)
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