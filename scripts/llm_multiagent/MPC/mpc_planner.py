import numpy as np
import matplotlib.pyplot as plt
from casadi import SX, vertcat, Function, Opti, DM

class MPCPlanner:
    def __init__(self, dt=0.1, N=10):
        self.dt = dt
        self.N = N
        self.reset()

    def reset(self):
        self.nx = 3   # x, y, theta
        self.nu = 2   # v, omega
        self.Qx = 0.5  
        self.Qy = 0.5   
        self.Qtheta = 0.5
        self.Qterm = 500.0  
        self.Rv = 0.1   
        self.Romega = 0.1
        self.Rdu = 0.1  
        self.L = 3 
       
        x = SX.sym('x')
        y = SX.sym('y')
        theta = SX.sym('theta')
        v = SX.sym('v')
        omega = SX.sym('omega')
        states = vertcat(x, y, theta)
        controls = vertcat(v, omega)

        rhs = vertcat(x + v*self.dt*np.cos(theta),
                      y + v*self.dt*np.sin(theta),
                      theta + omega*self.dt)
        self.f = Function('f', [states, controls], [rhs])

        self.opti = Opti()
        self.X = self.opti.variable(self.nx, self.N+1)  
        self.U = self.opti.variable(self.nu, self.N)    
        self.X0_param = self.opti.parameter(self.nx) 
        self.Xref_param = self.opti.parameter(self.nx, self.N+1)
        obj = 0
        for k in range(self.N):
            x_next = self.f(self.X[:,k], self.U[:,k])
            self.opti.subject_to(self.X[:,k+1] == x_next)
            state_error = self.X[:,k] - self.Xref_param[:,k]
            obj += self.Qx*(state_error[0]**2) + \
                   self.Qy*(state_error[1]**2) + \
                   self.Qtheta*(state_error[2]**2)
            obj += self.Rv*(self.U[0,k]**2) + self.Romega*(self.U[1,k]**2)
            if k < self.N-1:
                obj += self.Rdu * ((self.U[0,k] - self.U[0,k+1])**2 + (self.U[1,k] - self.U[1,k+1])**2)
        state_error_terminal = self.X[:,-1] - self.Xref_param[:,-1]
        obj += self.Qterm*(self.Qx*(state_error_terminal[0]**2) +
                           self.Qy*(state_error_terminal[1]**2) +
                           self.Qtheta*(state_error_terminal[2]**2))
        self.opti.subject_to(self.X[:,0] == self.X0_param)
        self.opti.subject_to(self.U[0,:] <= 1.5)    # v <= 0.4 m/s
        self.opti.subject_to(self.U[0,:] >= -1.5)   # v >= -0.4 m/s
        self.opti.subject_to(self.U[1,:] <= 0.8)    # omega <= 0.8 rad/s
        self.opti.subject_to(self.U[1,:] >= -0.8)   # omega >= -0.8 rad/s
        self.opti.minimize(obj)
        self.opti.solver('ipopt', {"max_iter_eig": 5000,'ipopt.print_level': 0, 'print_time': 0   })

    def solve_mpc(self, current_state, ref_states):
        self.opti.set_value(self.X0_param, current_state)
        self.opti.set_value(self.Xref_param, ref_states)
        dx = ref_states[0,0] - current_state[0]
        dy = ref_states[1,0] - current_state[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle_diff = ref_states[2,0] - current_state[2]
        if angle_diff > np.pi:
            angle_diff -= 2*np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2*np.pi
        if current_state[2] - ref_states[2,-1] > np.pi:
            current_state[2] -= 2*np.pi
        elif current_state[2] - ref_states[2,-1] < -np.pi:
            current_state[2] += 2*np.pi
        v_guess = distance / (self.N * self.dt)
        omega_guess = angle_diff / (self.N * self.dt)
        U_initial = DM(np.tile([v_guess, omega_guess], (self.N, 1)).T)
        self.opti.set_initial(self.U, U_initial)
        self.opti.set_initial(self.X, np.tile(np.array(current_state).reshape(-1,1),(1,self.N+1)))
        sol = self.opti.solve()
        u_opt = sol.value(self.U[:,0])
        return u_opt
    
def main():
    import time
    from planners.path_planner import PathPlanner
    from tool import KalmanFilter
    dt = 0.1
    N = 15
    T = 5.0
    steps = int(T/dt)
    
    path_planner = PathPlanner(num_ctrl_points=8, num_points=30, obstacle_effect_area=3)
    path = path_planner.generate_safe_path((-1.0, 0.5), (1.0, 2.0), [[0.0, 2.0]], num_points=100)
    path_x = np.array(path)[:,0]
    path_y = np.array(path)[:,1]
    path_theta = np.arctan2(np.gradient(path_y), np.gradient(path_x))
    kalman_filter_angle = KalmanFilter(1e-4, 1e-2)
    kalman_filter_vel = KalmanFilter(1e-4, 1e-2)
    
    x0, y0, theta0 = -1.0, 0.5, np.pi/2
    current_state = np.array([x0, y0, theta0])
    planner = MPCPlanner(dt=dt, N=N)

    x_hist = [current_state[0]]
    y_hist = [current_state[1]]
    theta_hist = [current_state[2]]
    v_hist = []
    omega_hist = []

    for i in range(steps):
        idx_end = min(i+N, steps-1)
        idx_ref = np.arange(i, idx_end+1)
  
        while len(idx_ref) < N+1:
            idx_ref = np.append(idx_ref, idx_ref[-1])
        ref_states = np.vstack([path_x[idx_ref],
                                path_y[idx_ref],
                                path_theta[idx_ref]])

        # planner.reset()
        start_time = time.time()
        u = planner.solve_mpc(current_state, ref_states)
        v_cmd, omega_cmd = u[0], u[1]
        v_cmd = kalman_filter_vel.update(v_cmd)
        omega_cmd = kalman_filter_angle.update(omega_cmd)

        new_x = current_state[0] + v_cmd*dt*np.cos(current_state[2])
        new_y = current_state[1] + v_cmd*dt*np.sin(current_state[2])
        new_theta = current_state[2] + omega_cmd*dt
        current_state = np.array([new_x, new_y, new_theta])
        x_hist.append(new_x)
        y_hist.append(new_y)
        theta_hist.append(new_theta)
        v_hist.append(v_cmd)
        omega_hist.append(omega_cmd)
        print(f"Step {i+1}/{steps}, Time: {time.time()-start_time:.4f}s")

    arrow_length = 0.5
    dx = np.cos(theta_hist) * arrow_length
    dy = np.sin(theta_hist) * arrow_length
    plt.figure(figsize=(10,6))
    plt.plot(path_x, path_y, 'r--', label='Reference')
    plt.plot(path_x, path_y, 'ro', label='Ref Points')
    plt.plot(x_hist, y_hist, 'b-', label='MPC')
    plt.plot(x_hist, y_hist, 'bo', label='MPC Points')
    plt.quiver(x_hist, y_hist, dx, dy, angles='xy', scale_units='xy', scale=1, color='g', width=0.002, linewidths=1.5, label='Orientation')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('MPC Planner with Orientation Arrows')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    plt.figure(figsize=(10,4))
    plt.subplot(2,1,1)
    plt.plot(np.arange(steps)*dt, v_hist, label='v')
    plt.ylabel('Vel [m/s]')
    plt.grid(True)
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.arange(steps)*dt, omega_hist, label='omega')
    plt.ylabel('Ang [rad/s]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()