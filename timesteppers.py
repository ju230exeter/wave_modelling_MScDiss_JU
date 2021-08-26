from firedrake import *

class SymplecticEuler(object):

    def __init__(self, Vcg, Vcg_R, phis_n, dt, g, linear, dim, boundary_condition):

        self.eta0 = Function(Vcg_R)
        self.phis0 = Function(Vcg_R)
        self.phi0 = Function(Vcg)

        # setting up timestepping solvers
        # set up eta solver
        h = TestFunction(Vcg_R)
        eta_trial = TrialFunction(Vcg_R)
        self.result_eta = Function(Vcg_R)

        aeta = h*eta_trial*ds_t
        nz = int(dim-1)
        Leta = h*(self.eta0 + dt*self.phi0.dx(nz))*ds_t

        if not linear:
            Leta += -h*dt*inner(self.phi0.dx(0), self.eta0.dx(0))*ds_t
            if dim == 3:
                Leta += -h*dt*inner(self.phi0.dx(1), self.eta0.dx(1))*ds_t

        LVP_eta = LinearVariationalProblem(aeta, Leta, self.result_eta)
        self.LVS_eta = LinearVariationalSolver(LVP_eta)

        # set up phis solver
        p = TestFunction(Vcg_R)
        phis_trial = TrialFunction(Vcg_R)
        self.result_phis = Function(Vcg_R)

        aphis = p*phis_trial*ds_t
        Lphis = p*(self.phis0-dt*(g*self.result_eta))*ds_t
        if not linear:
            Lphis += - p*dt*0.5*grad(self.phis0)**2*ds_t

        LVP_phis = LinearVariationalProblem(aphis, Lphis, self.result_phis)
        self.LVS_phis = LinearVariationalSolver(LVP_phis)

        # set up Laplace equation for phi
        q = TestFunction(Vcg)
        phi_trial = TrialFunction(Vcg)
        self.phis_bc = Function(Vcg)
        self.phi_side = Function(Vcg)

        apoisson = -inner(grad(q), grad(phi_trial))*dx
        Lpoisson = Constant(0)*q*dx

        bc = DirichletBC(Vcg, self.phis_bc, "top")
        bc_side = DirichletBC(Vcg, self.phi_side, 1)

        if boundary_condition == "piston" or boundary_condition == "hinged":
            poisson_prob = LinearVariationalProblem(apoisson, Lpoisson, self.phi0, bcs=[bc, bc_side])
        else:
            poisson_prob = LinearVariationalProblem(apoisson, Lpoisson, self.phi0, bcs=bc)
        self.poisson_solver = LinearVariationalSolver(poisson_prob)


    def apply(self, z, t, dt, H, A, om, boundary_condition, eta_n, phis_n, phi_n, eta_np1, phis_np1, phi_np1):

        # iterating movement of the wavemaker
        if boundary_condition == "piston":
            self.phi_side.assign(A*sin(-3*om*t))
        elif boundary_condition == "hinged":
            self.phi_side.interpolate(A*(z+H)*sin(-3*om*t))

        self.eta0.assign(eta_n)
        self.phis0.assign(phis_n)
        self.phi0.assign(phi_n)
        
        self.LVS_eta.solve()
        eta_np1.assign(self.result_eta)

        self.LVS_phis.solve()
        phis_np1.assign(self.result_phis)
        
        self.phis_bc.interpolate(phis_np1)
        self.poisson_solver.solve()
        phi_np1.assign(self.phi0)

class RK4(object):

    def __init__(self, Vcg, Vcg_R, phis_n, dt, g, linear, dim, boundary_condition):

        self.eta0 = Function(Vcg_R)
        self.phis0 = Function(Vcg_R)
        self.phi0 = Function(Vcg)

        # set up functions for RK4
        self.eta_k1 = Function(Vcg_R)
        self.phis_k1 = Function(Vcg_R)
        self.eta_k2 = Function(Vcg_R)
        self.phis_k2 = Function(Vcg_R)
        self.eta_k3 = Function(Vcg_R)
        self.phis_k3 = Function(Vcg_R)
        self.eta_k4 = Function(Vcg_R)
        self.phis_k4 = Function(Vcg_R)

        # setting up timestepping solvers
        # set up eta solver
        h = TestFunction(Vcg_R)
        eta_trial = TrialFunction(Vcg_R)
        self.result_eta = Function(Vcg_R)
        
        aeta = h*eta_trial*ds_t
        nz = int(dim-1)
        Leta = h*self.phi0.dx(nz)*ds_t
        if not linear:
            Leta += -h*(inner(self.phi0.dx(0), self.eta0.dx(0)))*ds_t
            if dim == 3:
                Leta += -h*dt*inner(self.phi0.dx(1), self.eta0.dx(1))*ds_t
        LVP_eta = LinearVariationalProblem(aeta, Leta, self.result_eta)
        self.eta_solver = LinearVariationalSolver(LVP_eta)

        # set up phis solver
        p = TestFunction(Vcg_R)
        phis_trial = TrialFunction(Vcg_R)
        self.result_phis = Function(Vcg_R)

        aphis = p*phis_trial*ds_t
        Lphis = -p*g*self.eta0*ds_t
        if not linear:
            Lphis += -p*0.5*grad(self.phis0)**2*ds_t

        LVP_phis = LinearVariationalProblem(aphis, Lphis, self.result_phis)
        self.phis_solver = LinearVariationalSolver(LVP_phis)

        # set up Laplace equation for phi
        q = TestFunction(Vcg)
        phi_trial = TrialFunction(Vcg)
        self.phis_bc = Function(Vcg)
        self.phi_side = Function(Vcg)

        apoisson = -inner(grad(q), grad(phi_trial))*dx
        Lpoisson = Constant(0)*q*dx

        bc = DirichletBC(Vcg, self.phis_bc, "top")
        bc_side = DirichletBC(Vcg, self.phi_side, 1)

        if boundary_condition == "piston" or boundary_condition == "hinged":
            poisson_prob = LinearVariationalProblem(apoisson, Lpoisson, self.phi0, bcs=[bc, bc_side])
        else:
            poisson_prob = LinearVariationalProblem(apoisson, Lpoisson, self.phi0, bcs=bc)
        self.poisson_solver = LinearVariationalSolver(poisson_prob)

    def apply(self, z, t, dt, H, A, om, boundary_condition, eta_n, phis_n, phi_n, eta_np1, phis_np1, phi_np1):

        # iterating movement of the wavemaker
        if boundary_condition == "piston":
            self.phi_side.assign(A*sin(-3*om*t))
        elif boundary_condition == "hinged":
            self.phi_side.interpolate(A*(z+H)*sin(-3*om*t))

        # RK4 timestepping
        # stage 1:
        self.eta0.assign(eta_n)
        self.phis0.assign(phis_n)
        
        self.eta_solver.solve()
        self.eta_k1.assign(self.result_eta)

        self.phis_solver.solve()
        self.phis_k1.assign(self.result_phis)
        
        # update boundary condition and solve for phi
        self.phis_bc.interpolate(phis_n + 0.5*dt*self.phis_k1)
        self.poisson_solver.solve()

        # stage 2:
        self.eta0.assign(eta_n + 0.5*dt*self.eta_k1)
        self.phis0.assign(phis_n + 0.5*dt*self.phis_k1)

        self.eta_solver.solve()
        self.eta_k2.assign(self.result_eta)

        self.phis_solver.solve()
        self.phis_k2.assign(self.result_phis)

        # update boundary condition and solve for phi
        self.phis_bc.interpolate(phis_n + 0.5*dt*self.phis_k2)
        self.poisson_solver.solve()

        # stage 3:
        self.eta0.assign(eta_n + 0.5*dt*self.eta_k2)
        self.phis0.assign(phis_n + 0.5*dt*self.phis_k2)

        self.eta_solver.solve()
        self.eta_k3.assign(self.result_eta)
        
        self.phis_solver.solve()
        self.phis_k3.assign(self.result_phis)

        # update boundary condition and solve for phi
        self.phis_bc.interpolate(phis_n + dt*self.phis_k3)
        self.poisson_solver.solve()

        # stage 4:
        self.eta0.assign(eta_n + dt*self.eta_k3)
        self.phis0.assign(phis_n + dt*self.phis_k3)

        self.eta_solver.solve()
        self.eta_k4.assign(self.result_eta)

        self.phis_solver.solve()
        self.phis_k4.assign(self.result_phis)

        # update boundary condition and solve for phi
        self.phis_bc.interpolate(phis_n + (1/6)*dt*(self.phis_k1 + 2*self.phis_k2 + 2*self.phis_k3 + self.phis_k4))
        self.poisson_solver.solve()

        # compute values at np1
        eta_np1.assign(eta_n + (1/6)*dt*(self.eta_k1 + 2*self.eta_k2 + 2*self.eta_k3 + self.eta_k4))
        phis_np1.assign(phis_n + (1/6)*dt*(self.phis_k1 + 2*self.phis_k2 + 2*self.phis_k3 + self.phis_k4))

        # update values at n and write out
        eta_n.assign(eta_np1)
        phis_n.assign(phis_np1)
        phi_n.assign(self.phi0)
        
