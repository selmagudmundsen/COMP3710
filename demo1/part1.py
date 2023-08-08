import torch
import numpy as np

print("Pytorch version: ", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian function
z = torch.exp(-(x**2 + y**2)/2.0)

# 2D cosine and sine function
zsine = torch.sin(x+2)
zcosine = torch.cos(x)

# multiply sine and Gaussian
gs = z * zsine

# plot
import matplotlib.pyplot as plt

plt.imshow(gs.numpy())
plt.tight_layout()
plt.show()