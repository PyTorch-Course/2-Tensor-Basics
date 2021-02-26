import torch

# torch.empty(size): uninitiallized
x = torch.empty(1) # scalar
print(f' x  = {x}')
y = torch.empty(3) #vector 3 componentes
print(f'y = {y}')
z = torch.empty(2,3) #matriz de 2D con tres elementos
print(f'z = {z}')

# rand, ones, zeros
# rellena con números aleatorios
x = torch.rand(5,3)
print(x)
# Rellena con 1
x = torch.ones(5,3)
print(x)
# Rellena con 0
x = torch.zeros(5,3)
print(x)

#obtenemos el tamaño
print(x.size())
#obtenemos el tipo de datos
print(x.dtype)

# especificamos el tipo de datos
x = torch.rand(5, 3, dtype=torch.float16)
print(x)
print(x.dtype)

# construyendo con datos
x = torch.tensor([5.5, 3])
print(x.size())

# operaciones
y = torch.rand(2, 2)
x = torch.rand(2, 2)


#Suma
z = x + y
print(z)

z = torch.add(x,y)
print(z)

y.add_(x)
print(y)

# Resta
z = x - y
print(z)

z = torch.sub(x,y)
print(z)

y.sub_(x)
print(y)

#Multiplicación

z = x * y
print(z)

z = torch.mul(x,y)
print(z)

y.mul_(x)
print(y)

#Division
z = x / y
print(z)
z = torch.div(x,y)
print(z)
y.div(x)
print(y)

# Slicing
x = torch.rand(5,3)
print(x)


# Todas las filas de la columna 0
print(x[:, 0])


#fila 1 y todas las columnas
print(x[1, :])
# El elemento 1,1
print(x[1,1]) 
#Obtenerl el valor actual, pero solo si hay un solo elemento
print(x[1,1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())


# Numpy
# Convertir de tensor a numpy y viceversa
a = torch.ones(5)

#De torch a numpy con .numpy()
b = a.numpy()

# Hay que tener cuidado porque tensor esta en la CPU ( no en la GPU), por lo tanto ambos objetos estan en la misma localización de memoria
# Si se cambia uno, cambia el otro

a.add_(1)
print(a)
print(b)


# numpy a torch con .from_numpy(x)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# También hay que tener cuidado
a += 1
print(a)
print(b)

print(torch.cuda.is_available())



# by default all tensors are created on the CPU,
# but you can also move them to the GPU (only if it's available )
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    # z = z.numpy() # not possible because numpy cannot handle GPU tenors
    # move to CPU again
    z.to("cpu")       # ``.to`` can also change dtype together!
    # z = z.numpy()

# requires_grad argument
# This will tell pytorch that it will need to calculate the gradients for this tensor
# later in your optimization steps
# i.e. this is a variable in your model that you want to optimize
x = torch.tensor([5.5, 3], requires_grad=True)



