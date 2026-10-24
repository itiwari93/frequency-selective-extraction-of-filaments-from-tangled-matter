"""
Simulation classes and numerical integration methods.

This module contains the SpringPendulumSystem for simulating driven spring-pendulum
dynamics and associated numerical integration utilities.
"""
import numpy as np
from scipy.integrate import solve_ivp

class SpringPendulumSystem:
    """
    Spring-pendulum system implementation of the following equation of motion:
        m \ddot{x} = F_gravity + F_drag + F_spring + F_damping

    where:
    - F_gravity = -mg\hat{z}
    - F_drag = -\gamma v
    - F_spring = -k(r - l)\hat{r}
    - F_damping = -c(v_r \cdot \hat{r})\hat{r}

    and the variables are defined as:
    - Position: x = [x, y, z] is the position vector
    - Velocity: v = \dot{x} is the velocity vector  
    - Piston pos/vel: x_p(t) = A_p sin(\Omega t)\hat{z}, v_p(t) = \Omega A_p cos(\Omega t)\hat{z}
    - Relative pos/vel from piston: r = x - x_p(t), v_r = v - v_p(t)
    """
    
    def __init__(self, m=1.0, g=1.0, gamma=0.0, k=1.0, L0=1.0, half=True, c=0.0, A_p=0.0, Omega=0.0, piston_direction='z', sigma=0.0, X0=None, **kwargs):
        # --- Physical parameters ---
        self.m = m                  # Mass
        self.g = g                  # Gravity
        self.gamma = gamma          # Drag coefficient
        self.k = k                  # Spring constant
        self.L0 = L0                # Spring length
        self.half = half            # Half spring model
        self.c = c                  # Damping coefficient

        # --- Driving parameters ---
        self.A_p = A_p              # Amplitude of piston
        self.Omega = Omega          # Frequency of piston (angular frequency)
        self.piston_direction = piston_direction.lower()    # Direction of piston movement ('x', 'y', or 'z')

        # --- Noise parameters ---
        self.sigma = sigma          # Guassian noise
        self.noise = True if self.sigma > 0 else False

        # --- System state ---
        if X0 is None: self.X = np.zeros(6)
        else: self.X = np.array(X0, dtype=float)            # Initial position and velocity, if provided
        self.X_p = self.piston_state(0)                     # Initial piston state at t=0

    # --- Forces and driving ---
    def piston_state(self, t):
        """Return piston position and velocity vector at time t as a (6,) array."""
        basis_vectors = {'x': np.array([1, 0, 0]),'y': np.array([0, 1, 0]),'z': np.array([0, 0, 1])}
        x_p = self.A_p * np.sin(self.Omega * t) * basis_vectors[self.piston_direction]
        x_p_dot = self.Omega * self.A_p * np.cos(self.Omega * t) * basis_vectors[self.piston_direction]
        X_p = np.append(x_p, x_p_dot)   
        return X_p
    
    def F_gravity(self):
        """Return the gravitational force vector."""
        return -self.m * self.g * np.array([0, 0, 1])
    
    def F_drag(self):
        """Return the drag force vector."""
        v = self.X[3:]
        return -self.gamma * v
    
    def F_spring(self):
        """Return the spring force vector."""
        r_vec = self.X[:3] - self.X_p[:3]
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r if r > 1e-12 else np.zeros_like(r_vec)
        
        F = -self.k * (r - self.L0) * r_hat
        if self.half and r < self.L0: F = np.zeros(3)

        return F
    
    def F_damping(self):
        """Return the damping force vector."""
        vr_vec = self.X[3:] - self.X_p[3:]
        r_vec = self.X[:3] - self.X_p[:3]
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r if r > 1e-12 else np.zeros_like(r_vec)

        F = -self.c * (np.dot(vr_vec, r_hat)) * r_hat
        if self.half and r < self.L0: F = np.zeros(3)
        return F

    # --- Equations of motion and simulation ---
    def equations_of_motion(self, t, X):
        """
        Returns the deterministic time derivative dX of the state X.
        """
        # Update system state
        self.X = np.array(X)
        self.X_p = self.piston_state(t)

        # Compute forces
        F_total = self.F_gravity() + self.F_drag() + self.F_spring() + self.F_damping()

        # Compute state derivative: [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
        x_dot = X[3:6]                  # Current velocity
        x_ddot = F_total / self.m       # Acceleration from Newton's second law
        return np.concatenate([x_dot, x_ddot])
    
    
    def simulate(self, t_span, t_eval=None, method='RK45', dt=0.01, save_interval=1, **kwargs):
        """
        Simulate the spring-pendulum system using scipy.integrate.solve_ivp.
        
        Parameters:
        -----------
        t_span : tuple, (t_start, t_end)
        t_eval : np.ndarray, optional, specific time points to evaluate
        method : str, integration method
        **kwargs : additional arguments passed to solve_ivp
        
        Returns:
        --------
        result : dict, simulation results with keys 't', 'X', 'success', 'message'
        """
        
        # Run simulation
        if not self.noise:
            sol = solve_ivp(
                self.equations_of_motion,
                t_span,
                self.X,  # Initial conditions
                t_eval=t_eval,
                method=method,
                **kwargs)
        
            return {
                't': sol.t,
                'X': sol.y.T,  # Shape: (N, 6) where N is number of time points
                'success': sol.success,
                'message': sol.message}
        else:
            noise_matrix = np.diag([0, 0, 0, self.sigma, self.sigma, self.sigma])/self.m
            result = integrate_sde(
                self.equations_of_motion, 
                noise_matrix, 
                self.X, t_span, 
                dt, 
                save_interval=save_interval,
                **kwargs)
            return result
    

# %% Integrators

def integrate_sde(f, sigma, X0, t_span, dt, random_seed = None, save_interval = 1):
    """
    Integrates dX = f(t, X) dt + sigma dW using Euler-Maruyama.

    Inputs:
        - f (callable): Drift term f(t,X), returns vector of size N.
        - sigma (scalar or np.ndarray): Diffusion scalar or matriX. If scalar, treated as isotropic noise.
        - X0 (scalar or np.ndarray): Initial state vector (N,) or scalar (N=1).
        - t_span (tuple): (t_start, t_end)  
        - dt (float): Time step.
        - random_seed (int, optional): random number generator seed.
        - n_save (int, optional): Save every `n_save` time steps. Default is 1.

    Outputs:
        t_save (np.ndarray): Saved time points, shape (T_save,).
        X_save (np.ndarray): Saved state trajectories, shape (T_save, N).
    """ 
    
    if random_seed is not None: np.random.seed(random_seed)     # set random seed for reproducibility
    if np.isscalar(X0): X0 = np.array([X0])                     # convert scalar X0 to 1D array
    if np.isscalar(sigma): sigma = sigma * np.eye(len(X0))      # treat scalar sigma as isotropic noise
    
    # [1] Initialize system  
    X = X0.copy()
    t = t_span[0] 
    T_total = t_span[1] - t_span[0]  # Total duration
    num_steps = int(T_total / dt)
    
    X_save, t_save = [], []
    
    # [2] Time-stepping
    for step in range(num_steps):
        if step % save_interval == 0:
            X_save.append(X.copy())
            t_save.append(t)
        
        dW = np.sqrt(dt) * np.random.normal(0, 1, size=(sigma.shape[1],))
        dX = f(t, X) * dt + sigma @ dW
        
        X += dX
        t += dt

    return {
        't': np.array(t_save),
        'X': np.array(X_save),
        'success': True,
        'message': 'SDE integration completed'
    }
