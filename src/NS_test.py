
# 0) Libraries and previous definitions
# ------------------------------------------------------
from dolfin import *
import matplotlib.pyplot as plt
from ufl import nabla_div, max_value
# from progress.bar import Bar
import numpy as np

from flux_correct import (
    assemble_mass_lumping,
    compute_artificial_dffusion,
)

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False
do_plot = False

# 1) Mesh
# ------------------------------------------------------

mesh_dir = "../mesh/"
mesh_file = mesh_dir + "rectang_circ_hole.xml"
mesh_file = mesh_dir + "rectang_eliptic_holes.xml"

# # Function for build .xmdf file from .msh one
# def convert_mesh_to_xdmf(input_file, output_file):
#     import numpy as np
#     import meshio
#     msh = meshio.read(input_mesh_file)
#     clls = np.vstack((
#         cell.data for cell in msh.cells if cell.type == "triangle"
#     ))  # only write 2D cells
#     meshio.xdmf.write(output_file, meshio.Mesh(msh.points, cells = {"triangle": clls}))
# mesh_file = mesh_dir + "rectang_circ_hole.xdmf"
# rebuild_mesh = False
# if rebuild_mesh:
#     input_mesh_file = mesh_dir + "rectang_circ_hole.msh"
#     convert_msh_to_xdmf(input_mesh_file, mesh_file)
# Read mesh from .xmf file
# mesh = Mesh()
# with XDMFFile(mesh_file) as infile:
#     infile.read(mesh)

# Read mesh from .xml file
mesh = Mesh(mesh_file)

x_coords = mesh.coordinates()[:, 0]
y_coords = mesh.coordinates()[:, 1]
X0, Y0 = x_coords.min(), y_coords.min()
X1, Y1 = x_coords.max(), y_coords.max()
print(f"Mesh bounding box: ({X0}, {Y0}) .. ({X1}, {Y1})")
d = mesh.topology().dim()
print("Mesh dimension:", d)
if do_plot:
    plot(mesh, title="Ship 2D mesh")
    plt.show()


# 2) Finite element spaces and functions
# ------------------------------------------------------

#define mixed functionspace
P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
element = MixedElement([P2, P1])
X = FunctionSpace(mesh, element)

# #define functions
xh = Function(X)
# xh = project( Expression((('0*x[0]*x[1]', '0*x[0]*x[1]'), '0*x[0]'), degree=2), X)
f1 = Expression( ("0", "0"), degree=2)
V1 = VectorFunctionSpace(mesh, "CG", 2)
f2 = Expression( ("0"), degree=2)
Q1 = FunctionSpace(mesh, "CG", 1)
# Initial value for u ( (0, 0) )
xh.sub(0, deepcopy=False).assign(interpolate(f1, V1))
# Initial value for p
# xh.sub(1, deepcopy=False).assign(interpolate(f2, Q1))

# tentative = interpolate(Constant((0,0,0)), X) #Function(X)
last_solution = Function(X)
u0, p0 = split(last_solution)

#define trial and test functions
(u, p) = TrialFunctions(X) #for velocity, pressure
(v, q) = TestFunctions(X)

#>> free surface
fs_fe_order = 1
fs_Uh = FunctionSpace(mesh, "Lagrange", fs_fe_order)
fs_Wh = VectorFunctionSpace(mesh, "DG", fs_fe_order-1)
fs_velocity = Function(fs_Wh)
eta, etab = TrialFunction(fs_Uh), TestFunction(fs_Uh)
# Variable to store solution at two time steps
eta1 = Function(fs_Uh)
# Initial value for free surface
eta_init = Expression("C1*exp(-C2*pow(x[0]-XX, 2))",
                      C1=1, C2=1.e-1, XX=X0+0.025*(X1-X0), degree=4)
eta1.assign(interpolate(eta_init, fs_Uh))
#<< free surface


# 3) Define data and coefficients
# ------------------------------------------------------

t = 0.0; dt = 0.1;
T = 10  # 1
force = Constant((0., 0.))  # External force
viscosity = 1.e-4  # Viscosity coefficient
inflow_velocity = Expression(("0.01*sqrt((x[1]-Y0)*(Y1-x[1]))", "0."),
                             Y0=Y0, Y1=Y1, degree=4)
normal = FacetNormal(mesh)

# 4) Define bounday conditions
# ------------------------------------------------------

# Define classes of boundaries
class LeftWall(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], X0))

class RightWall(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], X1))

class BottomWall(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], Y0))

class TopWall(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], Y1))

class ShipHull(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (not near(x[0], X0)) and (not near(x[1], Y0)) and \
            (not near(x[0], X1)) and (not near(x[1], Y1))

# Define boundaries of each class
top_wall = TopWall()
bottom_wall = BottomWall()
left_wall = LeftWall()
right_wall = RightWall()
ship_hull = ShipHull()

# Function definined as:
#  0: interior edges/faces
#  id: bounary edges/faces, where id is a humber representing each boundary
all_boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
# Function != 0 on outflow bounary (right wall), 0 on other edges/faces
outflow_boundary = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
# Function != 0 on ship hull bounary, 0 on other edges/faces
ship_boundary = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

# Id names for easyness
top_id = 1
bottom_id = 2
left_id  = 3
right_id = 4
ship_id = 5
walls_id = [1, 2, 3, 4]

# Mark edges/faces
top_wall.mark(all_boundaries, top_id)
bottom_wall.mark(all_boundaries, bottom_id)
left_wall.mark(all_boundaries, left_id)
right_wall.mark(all_boundaries, right_id)
ship_hull.mark(all_boundaries, ship_id)

right_wall.mark(outflow_boundary, right_id)
ship_hull.mark(ship_boundary, ship_id)

# For integration on boundaries
# ds = Measure('ds', domain=mesh, subdomain_data=all_boundaries)
# dsOut = Measure('ds', domain=mesh, subdomain_data=outflow_boundary)
# dsShip = Measure('ds', domain=mesh, subdomain_data=ship_boundary)
# ds = Measure('ds', domain=mesh, subdomain_data=all_boundaries)
# dsOut = Measure('ds', domain=mesh, subdomain_data=all_boundaries, subdomain_id=right_wall)
# dsShip = Measure('ds', domain=mesh, subdomain_data=all_boundaries, subdomain_id=ship_wall)
ds = Measure('ds', domain=mesh, subdomain_data=all_boundaries)
dsOut = ds(subdomain_id=right_id)
dsShip = ds(subdomain_id=ship_id)

# Save for testing
boundary_file = File("all_boundaries.pvd")
boundary_file << all_boundaries
boundary_file = File("out_boundaries.pvd")
boundary_file << outflow_boundary
boundary_file = File("ship_boundaries.pvd")
boundary_file << ship_boundary

bc_left = DirichletBC(X.sub(0), inflow_velocity, all_boundaries, left_id)
bc_top = DirichletBC(X.sub(0), (0, 0), all_boundaries, top_id)
bc_bottom = DirichletBC(X.sub(0), (0, 0), all_boundaries, bottom_id)
bc_ship = DirichletBC(X.sub(0), (0, 0), all_boundaries, ship_id)
dirichlet_bc = [bc_left, bc_top, bc_bottom] #, bc_ship]

# 5) Solve varitatioal formulation at time iterations
# ------------------------------------------------------
    # - p*div(v)*dx \
    # + inner(v,normal)*p*dsOut \
    # - viscosity*inner(grad(u.sub(0)), normal)*v.sub(0)*dsShip \

# Variational formultion
F1 = dot((u - u0) / dt, v)*dx \
    + dot(dot(u, nabla_grad(u0)), v)*dx \
    + viscosity*inner(grad(u), grad(v))*dx \
    + dot(force, v)*dx \
    + inner(grad(p), v)*dx \
    + 1.e-9*p*q*dx
    # + dot(nabla_grad(p), nabla_grad(q))*dx\    # + (dot(p*n, v)- dot(nu*nabla_grad(U)*n, v))*ds(inlet) \
    # + (dot(p*n, v)- dot(nu*nabla_grad(U)*n, v))*ds(right)

F_div_v1 = div(u)*q*dx
F_div_v2 = - inner(u, grad(q))*dx + inner(u, normal)*q*ds
F_ship_bnd = - viscosity*inner(dot(grad(u), normal), v)*dsShip - inner(u, normal)*q*dsShip

a1 = lhs(F1) + lhs(F_div_v2) + lhs(F_ship_bnd)
# a1 = lhs(F1) + lhs(F_div_v1)
L1 = rhs(F1)

#>> free surface
# Mass lumping matrix
fs_M = assemble(eta*etab*dx)
ML = assemble_mass_lumping(eta*etab*dx)
eta_vtk = File("/tmp/eta.pvd")
eta_vtk << (eta1, t)
#<< free surface

# set_log_active(False)
# bar = Bar('Processing', max=T/dt)
one = interpolate(Constant(1), Q1)
measure_of_domain = assemble(one*dx)
print("Measure of domain:", measure_of_domain)
u_vtk = File("/tmp/u.pvd")
p_vtk = File("/tmp/p.pvd")
u_vtk << (xh.sub(0), t)
p_vtk << (xh.sub(1), t)
m = 0
while t + dt < T + DOLFIN_EPS:
    dt = min(T-t, dt)
    t += dt
    print(f"Time iteration {m}, t={t}")
    last_solution.vector()[:] = xh.vector()
    u0, p0 = split(last_solution)

    solve(a1 == L1, xh, dirichlet_bc)
    u_vtk << (xh.sub(0), t)
    p_vtk << (xh.sub(1), t)

    u_norm =  sqrt(assemble(inner(xh.sub(0), xh.sub(0))*dx))
    print(f"  Reynolds number: (¿dimensions?) {u_norm*(X1-X0)/viscosity:.3e}")
    print("  L2 norm of velocity:", u_norm)
    print("  Mean of divergence:", abs(assemble(div(xh.sub(0))*dx))/measure_of_domain)
    print("  Mean of p:", abs(assemble(xh.sub(1)*dx))/measure_of_domain)

    #>> free surface
    #
    # Bulid matrix and RHS
    #
    fs_velocity = project(xh.sub(0), fs_Wh)
    # We integrate by parts the convection equation (assuming free divergence)
    K = assemble(-inner(fs_velocity, grad(eta))*etab*dx)
    # Flux correction: compte artificial diffusion and add it to advection
    # matrix K (=> we eliminate all negative off-diagonal coefficients of K)
    D, D_vals, _, _ = compute_artificial_dffusion(K)
    KL = D + K
    A = ML + dt*KL
    b = ML * eta1.vector()
    solve(A, eta1.vector(), b)
    print(f"  eta max/min: {max(eta1.vector())}/{min(eta1.vector())}")
    eta_vtk << (eta1, t)
    #<< free surface

    # u1.assign(xh.sub(0, deepcopy=True))
    # p1.assign(xh.sub(1, deepcopy=True))

    #update
    # bar.next()
# bar.finish()


# >>>> end
print("OK")
