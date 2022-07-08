from dolfin import *
import matplotlib.pyplot as plt

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False


# 1) Mesh
# ------------------------------------------------------

# a) Build .xmdf file from .msh one
def convert_mesh_to_xdmf(input_file, output_file):
    import numpy as np
    import meshio
    msh = meshio.read(input_mesh_file)
    clls = np.vstack((
        cell.data for cell in msh.cells if cell.type == "triangle"
    ))  # only write 2D cells
    meshio.xdmf.write(output_file, meshio.Mesh(msh.points, cells = {"triangle": clls}))

mesh_dir = "../mesh/"
input_mesh_file = mesh_dir + "rectang_circ_hole.msh"
mesh_file = mesh_dir + "rectang_circ_hole.xdmf"
convert_mesh_to_xdmf(input_mesh_file, mesh_file)

# b) Read .xmf file
mesh = Mesh()
with XDMFFile(mesh_file) as infile:
    infile.read(mesh)

x_coords = mesh.coordinates()[:, 0]
y_coords = mesh.coordinates()[:, 1]
X0, Y0 = x_coords.min(), y_coords.min()
X1, Y1 = x_coords.max(), y_coords.max()
print(f"Mesh bounding box: ({X0}, {Y0}) .. ({X1}, {Y1})")
plot(mesh, title="Ship 2D mesh")
plt.show()

# 2) Define function spaces (P2-P1) and functions
# ------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
bu = TestFunction(V)
bu = TestFunction(Q)

# 3) Define data and coefficients
# ------------------------------------------------------

# Set parameter values
dt = 0.01
T = 3
nu = 0.1
k = Constant(dt)
f = Constant((0, 0))

# 4) Define boundary conditions
# ------------------------------------------------------

# Velocity boundary conditions
u_in = (1,0)
u_inflow = DirichletBC(V, u_in, "x[1] > 1.0 - DOLFIN_EPS")
u_noslip = DirichletBC(V, (0, 0),
                       "on_boundary && \
                        (x[0] > X0+DOLFIN_EPS && x[1] > Y0+DOLFIN_EPS && \
                         x[0] < X1-DOLFIN_EPS && x[1] < Y1-DOLFIN_EPS) | \
                        (x[1] < X0+DOLFIN_EPS) | (x[1] > X1-DOLFIN_EPS)")
bcu = [u_inflow, u_noslip]

# Time-dependent pressure boundary condition
# p_in = Expression("sin(3.0*t)", t=0.0)
# p_inflow  = DirichletBC(Q, p_in, "x[1] > 1.0 - DOLFIN_EPS")
# p_outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")
# bcp = [inflow, outflow]


#
# Functions where solution will be stored
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

# 5) Variational formulation

F = nu*inner(grad(u), grad(bu))*dx + inner(grad(p),bu) \
    + div(u)*bp*dx \
    - inner(f, bu)*dx

# Define bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Assemble into matix and vector
A = assemble(a)
b = assemble(L)
[bc.apply(A, b) for bc in bcu] # Apply b.c. to A and b

# Compute solution
solve(A1, u1.vector(), b1, "gmres", "default")
