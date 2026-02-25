import torch
import numpy as np
import os
import argparse

# Parse command line arguments for output directory
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--in_dir', type=str, default='./', help='Input/output directory')
args = parser.parse_args()
in_dir = args.in_dir

n = 100
k = 10

# Create matrix u: first k columns of identity matrix, rotated by random orthogonal matrix
e = torch.eye(n)
u = e[:, :k]  # Get first k columns
q_init, _ = torch.linalg.qr(torch.randn(k, k))
u = torch.matmul(u, q_init)

# Save dense matrix
np.savetxt(os.path.join(in_dir, "dense"), u.detach().numpy())

def unpack(qtilde):
    """Extract orthogonal matrix from qtilde using QR decomposition"""
    q, _ = torch.linalg.qr(qtilde)
    return q

def loss(qtilde):
    """Loss function: mean of absolute values of rotated u"""
    q = unpack(qtilde)
    u_rotated = torch.matmul(u, q)
    return torch.mean(torch.abs(u_rotated))

def loss2(qtilde):
    """Alternative loss function: negative normalized fourth moment"""
    q = unpack(qtilde)
    u_rotated = torch.matmul(u, q)
    tmp = u_rotated - torch.mean(u_rotated)
    mean_sqr = torch.mean(torch.square(tmp))
    mean_sqr_sqr = torch.mean(torch.square(torch.square(tmp)))
    return -mean_sqr_sqr / torch.square(mean_sqr)

# Initialize Adam optimizer
# Note: We'll optimize qtilde directly
qtilde = torch.nn.Parameter(torch.randn(k, k), requires_grad=True)
optimizer = torch.optim.Adam([qtilde], lr=0.004, eps=1e-4)

# Remove existing loss file
loss_file = os.path.join(in_dir, "loss")
if os.path.exists(loss_file):
    os.remove(loss_file)

# Optimization loop
for iteration in range(10000):
    optimizer.zero_grad()
    loss_value = loss(qtilde)
    loss_value.backward()
    optimizer.step()
    
    # Save loss value
    with open(loss_file, 'a') as f:
        f.write(f"{loss_value.item()}\n")
    
    if iteration % 1000 == 0:
        print(f"Iteration {iteration}, Loss: {loss_value.item()}")

# Get optimized q
q_opt = unpack(qtilde)

# Save recovered matrix
recovered = torch.matmul(u, q_opt)
np.savetxt(os.path.join(in_dir, "recovered"), recovered.detach().numpy())
