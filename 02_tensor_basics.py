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

'''

# requires_grad argument
# This will tell pytorch that it will need to calculate the gradients for this tensor
# later in your optimization steps
# i.e. this is a variable in your model that you want to optimize
x = torch.tensor([5.5, 3], requires_grad=True)


# division
z = x / y
z = torch.div(x,y)

# Slicing
x = torch.rand(5,3)
print(x)
print(x[:, 0]) # all rows, column 0
print(x[1, :]) # row 1, all columns
print(x[1,1]) # element at 1, 1

# Get the actual value if only 1 element in your tensor
print(x[1,1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# if -1 it pytorch will automatically determine the necessary size
print(x.size(), y.size(), z.size())

# Numpy
# Converting a Torch Tensor to a NumPy array and vice versa is very easy
a = torch.ones(5)
print(a)

# torch to numpy with .numpy()
b = a.numpy()
print(b)
print(type(b))

# Carful: If the Tensor is on the CPU (not the GPU),
# both objects will share the same memory location, so changing one
# will also change the other
a.add_(1)
print(a)
print(b)

# numpy to torch with .from_numpy(x)
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# again be careful when modifying
a += 1
print(a)
print(b)

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
'''