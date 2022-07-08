
# 0) Libraries and previous definitions
# ------------------------------------------------------
from dolfin import *
import matplotlib.pyplot as plt
from ufl import nabla_div, max_value
# from progress.bar import Bar
import numpy as np

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False
do_plot = True

# 1) Mesh
# ------------------------------------------------------

# Function for build .xmdf file from .msh one
def convert_mesh_to_xdmf(input_file, output_file):
    import numpy as np
    import meshio
    msh = meshio.read(input_mesh_file)
    clls = np.vstack((
        cell.data for cell in msh.cells if cell.type == "triangle"
    ))  # only write 2D cells
    meshio.xdmf.write(output_file, meshio.Mesh(msh.points, cells = {"triangle": clls}))

# Define mesh file and eventually rebuild mesh
mesh_dir = "../mesh/"

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
mesh_file = mesh_dir + "rectang_circ_hole.xml"
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


# 2) Bounday conditions
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

class ShipWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (not near(x[0], X0)) and (not near(x[1], Y0)) and \
            (not near(x[0], X1)) and (not near(x[1], Y1))


# Define boundaries of each class
top_wall = TopWall()
bottom_wall = BottomWall()
left_wall = LeftWall()
right_wall = RightWall()
ship_wall = ShipWall()

#Function definined as:
#  0: interior edges/faces
#  id: bounary edges/faces, where id is a humber representing each boundary
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)

# Id names for easyness
top_id = 1
bottom_id = 2
left_id  = 3
right_id = 4
ship_id = 5

# Mark edges/faces
top_wall.mark(boundaries, top_id)
bottom_wall.mark(boundaries, bottom_id)
left_wall.mark(boundaries, left_id)
right_wall.mark(boundaries, right_id)
ship_wall.mark(boundaries, ship_id)

# For integration on all boundaries
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Save for testing
boundary_file = File("boundaries.pvd")
boundary_file << boundaries

# 3) Define data and coefficients
# ------------------------------------------------------

t = 0.0; dt = 0.01;
T = 1
force = Constant((0., 0.))  # External force
viscosity = Constant(1)  # Viscosity coefficient
nn = FacetNormal(mesh)
#endregion----------------------------------------


#region: define functionspaces and functions-------

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
xh.sub(0, deepcopy=False).assign(interpolate(f1, V1))
f2 = Expression( ("0"), degree=2)
Q1 = FunctionSpace(mesh, "CG", 1)
xh.sub(1, deepcopy=False).assign(interpolate(f2, Q1))

# tentative = interpolate(Constant((0,0,0)), X) #Function(X)
tentative = Function(X)
u0, p0 = split(tentative)

#define trial and test functions
(u, p) = TrialFunctions(X) #for velocity, pressure
(v, q) = TestFunctions(X)
#endregion-----------------------------------------

#region: define boundary conditions----------------
pd=Constant(1)
# bcp1 = DirichletBC(X.sub(1), pd, boundaries, left)
# bcp2 = DirichletBC(X.sub(1), Constant(0), boundaries, right)

# bcu1 = DirichletBC(X.sub(0), (0,0), boundaries, Top)
# bcu2 = DirichletBC(X.sub(0), (0,0), boundaries, Bottom)

bcp1 = DirichletBC(X.sub(0), (1, 0), boundaries, left_id)
# bcp2 = DirichletBC(X.sub(0), (0, 0), boundaries, right_id)

bcu1 = DirichletBC(X.sub(0), (0, 0), boundaries, top_id)
bcu2 = DirichletBC(X.sub(0), (0, 0), boundaries, bottom_id)

bcu_ship = DirichletBC(X.sub(0), (0, 0), boundaries, ship_id)

bc = [bcp1, bcu1, bcu2, bcu_ship]



F1 = dot((u - u0) / dt, v)*dx
F1 += dot(dot(u, nabla_grad(u0)), v)*dx
F1 += inner(grad(u), grad(v))*dx
F1 -= dot(force, v)*dx
F1 += inner(grad(p), v)*dx
F1 += div(u)*q*dx
F1 += 1.e-15*p*q*dx
    # + dot(nabla_grad(p), nabla_grad(q))*dx\    # + (dot(p*n, v)- dot(nu*nabla_grad(U)*n, v))*ds(inlet) \
    # + (dot(p*n, v)- dot(nu*nabla_grad(U)*n, v))*ds(right)

a1 = lhs(F1)
L1 = rhs(F1)


#region: solve problem---------------------------

# set_log_active(False)
# bar = Bar('Processing', max=T/dt)
u_vtk = File("/tmp/u.pvd")
p_vtk = File("/tmp/p.pvd")
u_vtk << (xh.sub(0), t)
p_vtk << (xh.sub(1), t)
m = 0
while t + dt < T + DOLFIN_EPS:
    dt = min(T-t, dt)
    t += dt
    print(f"Time iteration {m}, t={t}")
    tentative.vector()[:] = xh.vector()
    u0, p0 = split(tentative)

    solve(a1 == L1, xh, bc)
    u_vtk << (xh.sub(0), t)
    p_vtk << (xh.sub(1), t)

    # u1.assign(xh.sub(0, deepcopy=True))
    # p1.assign(xh.sub(1, deepcopy=True))

    #update
    # bar.next()
# bar.finish()
#endregion-----------------------------------------


# >>>> end
print("OK")
