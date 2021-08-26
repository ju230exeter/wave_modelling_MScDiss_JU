import argparse
import itertools
import time
import matplotlib.pyplot as plt
from firedrake import *
from timesteppers import SymplecticEuler
from timesteppers import RK4

def run_wave_model(method, initial_condition, boundary_condition, linear, dim, numlayers, L, H, A, dt, tmax):
    
    print("in run_wave_model")

    # set up a periodic interval mesh and extrude in the vertical
    if dim == 2:
        if boundary_condition == "periodic":
            m = PeriodicIntervalMesh(numlayers, L)
        else:
            m = IntervalMesh(numlayers, L)
        mesh = ExtrudedMesh(m, 2*numlayers, H/(2*numlayers))
        x, z = SpatialCoordinate(mesh)
        coords = mesh.coordinates
        Xstart = Function(coords.function_space())
        Xstart.interpolate(as_vector((x, z)))
    else:
        if boundary_condition == "periodic":
            m = PeriodicSquareMesh(numlayers, numlayers, L)
        else:
            m = SquareMesh(numlayers, numlayers, L)
        mesh = ExtrudedMesh(m, 2*numlayers, H/(2*numlayers))
        x, y, z = SpatialCoordinate(mesh)
        coords = mesh.coordinates
        Xstart = Function(coords.function_space())
        Xstart.interpolate(as_vector((x, y, z)))
    
    # set up function spaces
    Vcg = FunctionSpace(mesh, "CG", 1)
    Vcg_R = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

    eta0 = Function(Vcg_R)
    phis0 = Function(Vcg_R)
    phi0 = Function(Vcg)
    eta_n = Function(Vcg_R, name="eta")
    phis_n = Function(Vcg_R, name="phis")
    phi_n = Function(Vcg, name="phi")
    eta_np1 = Function(Vcg_R)
    phis_np1 = Function(Vcg_R)
    phi_np1 = Function(Vcg)

    # set up functions for analytical solutions (for linear case only)
    eta_ex = Function(Vcg_R, name="eta exact")
    phis_ex = Function(Vcg_R, name="phis exact")
    
    # initialise model
    # parameters
    k = 2*pi
    l = 2*pi
    g = 9.81

    if boundary_condition != "piston" or boundary_condition != "hinged":

        # in the case of shallow water
        if initial_condition == "shallow":
            om = sqrt(g*k*tanh(k*H))
            if dim == 2:
                eta_n.interpolate(A*cos(k*x))
                phis_n.interpolate((A*om/(k*tanh(H*k)))*sin(k*x))
                if linear:
                    eta_ex.interpolate(A*cos(k*x))
                    phis_ex.interpolate((A*om/(k*tanh(H*k)))*sin(k*x))
                else:
                    coords.interpolate(as_vector((x, Xstart[1]+(Xstart[1]/H)*eta_n)))
            else:
                eta_n.interpolate(A*cos(k*x+l*y))
                phis_n.interpolate((A*om/(k*tanh(H*k)))*sin(k*x+l*y))
                if linear:
                    eta_ex.interpolate(A*cos(k*x+l*y))
                    phis_ex.interpolate((A*om/(k*tanh(H*k)))*sin(k*x+l*y))
                else:
                    coords.interpolate(as_vector((x, y, Xstart[2]+(Xstart[2]/H)*eta_n)))

        # in the case of deep water
        elif initial_condition == "deep":
            om = sqrt(g*k)
            if dim == 2:
                eta_n.interpolate(A*cos(k*x))
                phis_n.interpolate((A*om/k)*sin(k*x))
                if linear:
                    # initialising eta_ex and phis_ex for linear case
                    eta_ex.interpolate(A*cos(k*x))
                    phis_ex.interpolate((A*om/k)*sin(k*x))
                else:
                    # moving the mesh
                    coords.interpolate(as_vector((x, Xstart[1]+(Xstart[1]/H)*eta_n)))
            else:
                eta_n.interpolate(A*cos(k*x+l*y))
                phis_n.interpolate((A*om/k)*sin(k*x+l*y))
                if linear:
                    eta_ex.interpolate(A*cos(k*x+l*y))
                    phis_ex.interpolate((A*om/k)*sin(k*x+l*y))
                else:
                    coords.interpolate(as_vector((x, y, Xstart[2]+(Xstart[2]/H)*eta_n)))

        # in the case of a NONLINEAR wave packet
        elif initial_condition == "packet":
            om = 0
            if dim == 2:
                if linear:
                    print("This is a nonlinear test case. Try again with nonlinear arg.")
                else:
                    eta_n.interpolate(A*exp(-0.75*(x-pi)**2)*cos(10*x))
                    coords.interpolate(as_vector((x, Xstart[1]+(Xstart[1]/H)*eta_n)))
            else:
                print("This case only exists in 2 dimensions.")

        # in the case of a NONLINEAR wave solution
        elif initial_condition == "exactnonlin":
            if dim == 2:
                if linear:
                    print("This is a nonlinear test case. Try again with nonlinear arg.")
                else:
                    eta_n.interpolate(A*cos(2*pi*x/L))
                    eta_ex.interpolate(A*cos(2*pi*x/L))
                    coords.interpolate(as_vector((x, Xstart[1]+(Xstart[1]/H)*eta_n)))
            else:
                print("This case only exists in 2 dimensions.")

    # in the case of wavemaker, i.e. boundary_condition == "piston" or == "hinged",
    if boundary_condition == "piston" or boundary_condition == "hinged":
        om = 1.8138

    # setup output file and write out initial values
    outfile = File("output.pvd")
    hov_data_ex = open("eta_ex.txt", "a")
    x_vals = np.linspace(0, L, 101)
    z_vals = [0]
    
    if dim == 2:
        points = np.array([p for p in itertools.product(x_vals, z_vals)])
    else:
        y_vals = [0.5]
        points = np.array([p for p in itertools.product(x_vals, y_vals, z_vals)])
        
    if method == "symplectic":
        timestepper = SymplecticEuler(Vcg, Vcg_R, phis_n, dt, g, linear, dim, boundary_condition)
        hov_data = open("eta_sym.txt", "a")
    else:
        timestepper = RK4(Vcg, Vcg_R, phis_n, dt, g, linear, dim, boundary_condition)
        hov_data = open("eta_rk4.txt", "a")
    
    np.savetxt(hov_data, eta_n.at(points, tolerance=0.1*0.01))
    np.savetxt(hov_data_ex, eta_ex.at(points, tolerance=0.1*0.01))
    outfile.write(eta_n, phis_n, phi_n, eta_ex, phis_ex)

    # Energy calculation
    E0 = assemble((0.5*dot(grad(phi_n), grad(phi_n))+0.5*g*eta_n**2)*dx)
    E = E0

    # Setting up empty arrays to append result into
    StartTime = time.time()
    t = 0
    error = np.array([])
    energy = np.array([])
    Time = np.array([])
    while t < tmax:
        print("time: ", t)
        t += dt
        Time = np.append(Time, t)
        energy = np.append(energy, abs((E-E0)/E0))

        # exact solutions through the iteration
        if linear:
            if initial_condition == "shallow":
                if dim == 2:
                    eta_ex.interpolate(A*cos(k*x-om*t))
                    phis_ex.interpolate((A*om/(k*tanh(H*k)))*sin(k*x-om*t))
                else:
                    eta_ex.interpolate(A*cos(k*x+l*y-om*t))
                    phis_ex.interpolate((A*om/(k*tanh(H*k)))*sin(k*x+l*y-om*t))
            elif initial_condition == "deep_water":
                # Set up analytical solution
                if dim == 2:
                    eta_ex.interpolate(A*cos(k*x-om*t))
                    phis_ex.interpolate((A*om/k)*sin(k*x-om*t))
                else:
                    eta_ex.interpolate(A*cos(k*x+l*y-om*t))
                    phis_ex.interpolate((A*om/k)*sin(k*x+l*y-om*t))

            # Calculate error between exact and numerical data for linear case
            error = np.append(error, errornorm(eta_ex, eta_n))

        # for nonlinear case
        else:
            if initial_condition == "exactnonlin":
                k2 = 2*pi/L
                om = sqrt(g*k2*tanh(k2*H))
                eta_ex.interpolate(A*cos(om*t)*cos(k2*x))

            error = np.append(error, errornorm(eta_ex, eta_n))

        # Now numerical solution
        timestepper.apply(z, t, dt, H, A, om, boundary_condition, eta_n, phis_n, phi_n, eta_np1, phis_np1, phi_np1)
        
        # Now move the mesh for nonlinear case
        if not linear:
            if dim == 2:
                coords.interpolate(as_vector((x, Xstart[1]+(Xstart[1]/H)*eta_n)))
            else:
                coords.interpolate(as_vector((x, y, Xstart[2]+(Xstart[2]/H)*eta_n)))
        
        eta_n.assign(eta_np1)
        phis_n.assign(phis_np1)
        phi_n.assign(phi_np1)
        outfile.write(eta_n, phis_n, phi_n, eta_ex, phis_ex)
        E = assemble((0.5*dot(grad(phi_np1), grad(phi_np1))+0.5*g*eta_np1**2)*dx)
        np.savetxt(hov_data, eta_n.at(points, tolerance=0.1*0.01))
        np.savetxt(hov_data_ex, eta_ex.at(points, tolerance=0.1*0.01))

    print("total time to run:", time.time()-StartTime)
    
    hov_data.close()
    hov_data_ex.close()
    np.savetxt("energy.txt", energy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run wave model")
    parser.add_argument("method", help='specify integrator of choice: symplectic or rk4')
    parser.add_argument("initial_condition", help='specify test case. options include: shallow, deep, packet and exactnonlin. write "none" for wavemakers')
    parser.add_argument("boundary_condition", help='options include: periodic, nonperiodic, piston or hinged')
    parser.add_argument("--linear", action="store_true", help='put --linear if linear, ignore if not')
    parser.add_argument("dim", type=int, help='specify dimension, either 2 or 3')
    parser.add_argument("numlayers", type=int, help='specify number of grid points')
    parser.add_argument("L", type=float, help='specify length of domain')
    parser.add_argument("H", type=float, help='specify depth of domain')
    parser.add_argument("A", type=float, help='specify amplitude of wave')
    parser.add_argument("dt", type=float, help='specify timestep')
    parser.add_argument("tmax", type=float, help='specify end time of iteration')

    args = parser.parse_args()

    run_wave_model(args.method, args.initial_condition, args.boundary_condition, args.linear, args.dim, args.numlayers, args.L, args.H, args.A, args.dt, args.tmax)
