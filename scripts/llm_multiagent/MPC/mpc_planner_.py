import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import math

class MPCPlanner_drone:
    def __init__(self, horizon=10, dt=0.5):
        self.horizon = horizon
        self.dt = dt
        self.alpha1 = 1.0    
        self.alpha2 = 0.1    
        self.alpha3 = 1.0  
        self.terminal_alpha = 10.0
        
        self.x_max = 5.0
        self.y_max = 3.0

    def solve(self, current_pos, dog_pos, ref_path):
        ref_path = np.array(ref_path)
        
        x = cp.Variable((self.horizon+1, 2))
        u = cp.Variable((self.horizon, 2))

        constraints = [x[0, :] == current_pos]
        dist_ref = cp.sum_squares(ref_path[1,:] - ref_path[0,:])
        constraints += [cp.sum_squares(x[1,:] - ref_path[0,:]) <= 0.9 * dist_ref]
        cost = 0

        for k in range(self.horizon):
            constraints += [x[k+1, :] == x[k, :] + u[k,:]*self.dt]

            cost += self.alpha1 * cp.sum_squares(x[k,:] - ref_path[k,:])
            
            cost += self.alpha2 * cp.sum_squares(u[k,:])

            x_dog_cam = dog_pos[0] - x[k,0]
            y_dog_cam = dog_pos[1] - x[k,1]

            out_of_x = cp.maximum(0, cp.abs(x_dog_cam) - self.x_max)
            out_of_y = cp.maximum(0, cp.abs(y_dog_cam) - self.y_max)
            cost += self.alpha3 * (out_of_x + out_of_y)

        # final_target = ref_path[-1, :] 
        # cost += self.terminal_alpha * cp.sum_squares(x[self.horizon,:] - final_target)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.SCS, warm_start=True)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            return x.value[1,:], problem.status
        else:
            return current_pos, problem.status

class MPCPlanner:
    def __init__(self, 
                 dt=0.1, 
                 N=20, 
                 v_max=0.5, 
                 v_min=-0.5, 
                 omega_max=np.pi/4, 
                 omega_min=-np.pi/4,
                 w_pos=100.0,
                 w_theta=50.0,
                 w_input=0.1,
                 w_input_diff=0.1):

        self.dt = dt
        self.N = N
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.omega_min = omega_min
        
        self.w_pos = w_pos
        self.w_theta = w_theta
        self.w_input = w_input
        self.w_input_diff = w_input_diff
        
        self.n_states = 3   # x, y, theta
        self.n_controls = 2 # v, omega

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        self.states = ca.vertcat(x, y, theta)
        
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        self.controls = ca.vertcat(v, omega)

        self.f_dyn = self._build_dynamics_function()

        self.solver, self.X_symbol, self.U_symbol, self.P_symbol, self.lbx, self.ubx = self._build_solver()

    def _build_dynamics_function(self):
        x, y, theta = self.states[0], self.states[1], self.states[2]
        v, omega = self.controls[0], self.controls[1]
        
        x_next = x + v*ca.cos(theta)*self.dt
        y_next = y + v*ca.sin(theta)*self.dt
        theta_next = theta + omega*self.dt

        f_dyn = ca.Function('f_dyn', [self.states, self.controls], 
                            [ca.vertcat(x_next, y_next, theta_next)])
        return f_dyn

    def _build_solver(self):
        X = ca.SX.sym('X', self.n_states, self.N+1)
        U = ca.SX.sym('U', self.n_controls, self.N)
        P = ca.SX.sym('P', self.n_states+ self.n_states) 

        x0 = P[0]
        y0 = P[1]
        theta0 = P[2]
        xg = P[3]
        yg = P[4]
        thetag = P[5]

        obj = 0
        g = []
        
        g.append(X[:,0] - ca.vertcat(x0, y0, theta0))

        for k in range(self.N):
            x_next = self.f_dyn(X[:,k], U[:,k])
            g.append(X[:,k+1] - x_next)

            pos_error = (X[0,k+1]-xg)**2 + (X[1,k+1]-yg)**2
            theta_error = (X[2,k+1]-thetag)**2

            input_cost = (U[0,k]**2 + U[1,k]**2)

            obj += self.w_pos*pos_error + self.w_theta*theta_error + self.w_input*input_cost

            if k > 0:
                input_diff = (U[:,k]-U[:,k-1])**2
                obj += self.w_input_diff*ca.sum1(input_diff)

        final_pos_error = (X[0,self.N]-xg)**2 + (X[1,self.N]-yg)**2
        final_theta_error = (X[2,self.N]-thetag)**2
        obj += self.w_pos*final_pos_error + self.w_theta*final_theta_error

        g = ca.vertcat(*g)
        n_g = g.numel()
        lbg = np.zeros(n_g)
        ubg = np.zeros(n_g)

        vars = ca.vertcat(
            ca.reshape(X, self.n_states*(self.N+1), 1),
            ca.reshape(U, self.n_controls*self.N, 1)
        )

        lbx = []
        ubx = []
        for _ in range(self.N+1):
            lbx.extend([-ca.inf, -ca.inf, -ca.inf])
            ubx.extend([ ca.inf,  ca.inf,  ca.inf])

        for _ in range(self.N):
            lbx.extend([self.v_min, self.omega_min])
            ubx.extend([self.v_max, self.omega_max])

        prob = {
            'f': obj,
            'x': vars,
            'g': g,
            'p': P
        }
        opts = {
            'print_time': 0,
            'ipopt.print_level': 0,
            'ipopt.max_iter': 200
        }
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        return solver, X, U, P, lbx, ubx

    def solve(self, current_state, goal_state):
        p_value = np.concatenate((current_state, goal_state))
        
        res = self.solver(
            p=p_value,
            lbg=0,
            ubg=0,
            lbx=self.lbx,
            ubx=self.ubx
        )
        sol = res['x'].full().flatten()
        
        offset_u = self.n_states*(self.N+1)
        U_opt = sol[offset_u : offset_u+self.n_controls*self.N].reshape(self.n_controls, self.N)
        
        v_cmd = U_opt[0,0]
        omega_cmd = U_opt[1,0]
        return v_cmd, omega_cmd

    def step_dynamics(self, state, control):
        return self.f_dyn(state, control).full().flatten()


if __name__ == "__main__":
    mpc = MPCPlanner()

    current_state = np.array([0.0, 0.0, np.pi/2])  
    goal_state = np.array([0.0, -2.0, np.pi/4])  

    max_steps = 100
    goal_tolerance_pos = 0.1
    goal_tolerance_theta = 0.1

    trajectory_states = [current_state.copy()]
    trajectory_controls = []

    for step in range(max_steps):
        v_cmd, omega_cmd = mpc.solve(current_state, goal_state)
        current_state = mpc.step_dynamics(current_state, np.array([v_cmd, omega_cmd]))
        trajectory_states.append(current_state.copy())
        trajectory_controls.append((v_cmd, omega_cmd))

        pos_error = math.sqrt((current_state[0]-goal_state[0])**2 + (current_state[1]-goal_state[1])**2)
        theta_error = abs((current_state[2]-goal_state[2]))
        theta_error = (theta_error + np.pi) % (2*np.pi) - np.pi

        if pos_error < goal_tolerance_pos and abs(theta_error) < goal_tolerance_theta:
            print(f"Step: {step+1}")
            break
    else:
        print("Failed to reach goal in time")

    print("Trajectory:")
    for i, s in enumerate(trajectory_states[:5]):
        print(f"Step {i}: x={s[0]:.3f}, y={s[1]:.3f}, theta={s[2]:.3f}")
    if len(trajectory_states) > 10:
        print("...")
        for i, s in enumerate(trajectory_states[-5:]):
            idx = len(trajectory_states)-5+i
            print(f"Step {idx}: x={s[0]:.3f}, y={s[1]:.3f}, theta={s[2]:.3f}")

    trajectory_states = np.array(trajectory_states)
    plt.figure(figsize=(8,6))
    plt.plot(trajectory_states[:,0], trajectory_states[:,1], 'b.-', label='Trajectory')
    plt.plot([goal_state[0]], [goal_state[1]], 'ro', label='Goal')
    plt.arrow(trajectory_states[0,0], trajectory_states[0,1], 
              0.3*np.cos(trajectory_states[0,2]), 0.3*np.sin(trajectory_states[0,2]), 
              head_width=0.1, fc='g', ec='g', label='Start Heading')
    plt.arrow(goal_state[0], goal_state[1], 
              0.3*np.cos(goal_state[2]), 0.3*np.sin(goal_state[2]),
              head_width=0.1, fc='r', ec='r', label='Goal Heading')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("MPC Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# if __name__ == "__main__":
#     current_pos = np.array([0.0, 0.042])
#     dog_pos = np.array([2.0, 0.0])
#     ref_path = np.zeros((200, 2))
#     for i in range(200):
#         ref_path[i,:] = [0, (i+1)*0.04]  

#     mpc = MPCPlanner_drone(horizon=min(10, len(ref_path)), dt=0.5)
#     next_pos, status = mpc.solve(current_pos, dog_pos, ref_path)

#     print("Solve status:", status)
#     print("Current Pos:", current_pos)
#     print("Dog Pos:", dog_pos)
#     print("Ref Path (last point is goal):", ref_path[-1,:])
#     print("Optimal next pos:", next_pos)
