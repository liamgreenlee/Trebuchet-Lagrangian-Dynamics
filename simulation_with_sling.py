import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.animation as animation

# --- Symbolic Setup ---
t = sp.symbols('t')
# Generalized coordinates
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
dtheta1 = theta1.diff(t)
dtheta2 = theta2.diff(t)
# Physical parameters
m_cw, m_p, m_arm, L1, L2, L3, g, c1, c2, c3 = sp.symbols('m_cw m_p m_arm L1 L2 L3 g c1 c2 c3')

# Arm length
L = L1 + L2

# Counterweight and projectile positions
x_cw = sp.Matrix([[0],[L1*sp.sin(theta1)]])
x_end = sp.Matrix([[-L*sp.cos(theta1)], [-L2*sp.sin(theta1)]])
x_pivot = sp.Matrix([[-L1*sp.cos(theta1)],[0]])
x_p = x_end + sp.Matrix([[L3*sp.sin(theta2)],[-L3*sp.cos(theta2)]])

Rz = sp.Matrix([[sp.cos(theta1), -sp.sin(theta1)],[sp.sin(theta1), sp.cos(theta1)]])

lcom = -(L2-L1)/2
x_arm = Rz*sp.Matrix([[lcom],[0]])

dx_cw = x_cw.diff(t)
dx_p = x_p.diff(t)
dx_arm = x_arm.diff(t)
dx_pivot = x_pivot.diff(t)

# Moment of inertia about pivot using parallel axis theorem
# I_arm = (1/12) * m_arm * L**2 + m_arm * acom**2
I_arm = (1/12) * m_arm * L**2
I_pivot = I_arm + m_arm**lcom**2

# --- Kinetic Energy ---
# arm translation velocity + arm angular velocity (where arm includes the projectile and counterweight masses)
# T = (1/2)*M*sp.diff(arm_com[0],t)**2 + (1/2)*M*sp.diff(arm_com[1],t)**2 + (1/2)*(I_pivot)*dtheta**2
T = ((1/2)*m_p*dx_p.T*dx_p + (1/2)*m_cw*dx_cw.T*dx_cw + (1/2)*m_arm*dx_arm.T*dx_arm)[0] + (1/2)*I_pivot*dtheta1**2

# --- Potential Energy ---
# Gravity affects both the counterweight and projectile
V_cw = m_cw*g*x_cw[1]
V_p = m_p*g*x_p[1]
V_arm = m_p*g*x_arm[1]
V = V_cw + V_p + V_arm

# --- Lagrangian ---
Lagr = T - V


eq1 = sp.Eq(sp.diff(sp.diff(Lagr,dtheta1),t) - sp.diff(Lagr,theta1) - c1*dx_pivot[0] + c2*dtheta1 - c3*sp.diff(x_cw[1], t), 0)
eq2 = sp.Eq(sp.diff(sp.diff(Lagr,dtheta2),t) - sp.diff(Lagr,theta2) - c2*dtheta2, 0)
ddot_theta1 = sp.solve(eq1, sp.diff(theta1,t,t))[0]
ddot_theta2 = sp.solve(eq2, sp.diff(theta2,t,t))[0]

# --- Substitute parameter values ---
params = {
    m_cw: 150,
    m_p: 0.5,
    m_arm: 0,
    L1: 0.5,
    L2: 3.0,
    L3: 2.5,
    g: 9.81,
    c1: 0.3,  # Friction coefficient (change as needed)
    c2: 0,  # Friction coefficient (change as needed)
    c3: 0
}
# the density of wood is about 500kg/m^3
# assume 2in by 2in (5cm by 5cm) wood arm
params[m_arm] = (0.03)**2*(L.subs(params))*500
# The damping factor due to normal force of friction is mu*m*g. Here c1 is mu
M = m_cw + m_p + m_arm
params[c1] = M.subs(params)*params[g]*params[c1]

def simulate(params, t_release = None):
    eom1 = ddot_theta1.subs(params)
    eom2 = ddot_theta2.subs(params)
    # --- ODE System ---
    def trebuchet_ode(t, y, eom1, eom2):
        th1, dth1, th2, dth2 = y
        ddth1 = eom1.subs({theta1: th1, dtheta1: dth1, theta2: th2, dtheta2: dth2}).doit()
        ddth2 = eom2.subs({theta1: th1, dtheta1: dth1, theta2: th2, dtheta2: dth2}).doit()
        return [dth1, ddth1, dth2, ddth2]

    # --- Simulate ---
    # y0 = [np.pi/4, 0]  # initial theta, dtheta, x, dx
    d_theta = sp.solve(sp.Eq(dx_cw[1], -np.sqrt(2*(9.81)*0.1)), dtheta1)[0].subs(params)
    y0 = [np.pi/4, d_theta.subs({theta1: np.pi/4}), 0, 0]
    t0 = 0
    t_end = 4

    if t_release:
        sol1 = solve_ivp(lambda t, y: trebuchet_ode(t, y, eom1, eom2), [t0, t_release], y0, t_eval=np.linspace(t0, t_release, int(1000*(t_release/t_end))))
        y_release = sol1.y[:,-1]
        release_ind = len(sol1.t)
        params[m_p] = 0
        eom1 = ddot_theta1.subs(params)
        eom2 = ddot_theta2.subs(params)
        sol2 = solve_ivp(lambda t,y: trebuchet_ode(t, y, eom1, eom2), [t_release, t_end], y_release, t_eval=np.linspace(t_release, t_end,int(1000*(1-t_release/t_end))))

        t_full = np.hstack((sol1.t, sol2.t[1:]))
        y_full = np.hstack((sol1.y, sol2.y[:, 1:]))
    else:
        sol1 = solve_ivp(lambda t, y: trebuchet_ode(t, y, eom1, eom2), [t0, t_end], y0, t_eval=np.linspace(t0, t_end, 1000))
        t_full = sol1.t
        y_full = sol1.y
        max_vel_ind = np.argmax(np.abs(y_full[1,:]))
        y_release = y_full[:,max_vel_ind]
    return t_full, y_full, y_release

def state_plot(t, y, params, plot = True):

    x_p_val = np.array([
        np.array(x_p.subs(params).subs({theta1: y[0, i], dtheta1: y[1, i], theta2: y[2,i], dtheta2: y[3,i]})).astype(float)
        for i in range(y.shape[1])
    ]).reshape(y.shape[1],2)
    dx_p_val = np.array([
        np.array(dx_p.subs(params).subs({theta1: y[0, i], dtheta1: y[1, i], theta2: y[2,i], dtheta2: y[3,i]})).astype(float)
        for i in range(y.shape[1])
    ]).reshape(y.shape[1], 2)


    g = 9.81  # m/s^2
    x0 = x_p_val[:, 0]  # horizontal positions
    y0 = x_p_val[:, 1]  # vertical positions
    vx0 = dx_p_val[:, 0]
    vy0 = dx_p_val[:, 1]

    # Time of flight: only real and positive roots
    t_flight = (vy0 + np.sqrt(np.maximum(vy0**2 + 2 * g * y0, 0))) / g

    # Compute range
    range_vals = vx0 * t_flight
    max_range_ind = np.argmax(range_vals)
    optimal_release_time = t[max_range_ind]
    optimal_release_state = y[:,max_range_ind]
    if plot:
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(t, range_vals)
        plt.axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)

        plt.xlabel('Time (s)')
        plt.ylabel('Projectile Range (m)')
        plt.title('Projectile Range vs. Time')
        plt.grid(True)
        plt.tight_layout()

        # --- Plot Results ---
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(t, y[0])
        plt.axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)
        plt.ylabel(r'$\theta$ (rad)')
        plt.title('Floating Arm Trebuchet Simulation with Pivot Friction')

        plt.subplot(2, 1, 2)
        plt.plot(t, y[1])
        plt.axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)
        plt.ylabel(r'$\dot{\theta}$ (rad/s)')
        plt.xlabel('Time (s)')
        plt.tight_layout()

        fig, ax1 = plt.subplots(3, 1, figsize=(10, 8))

        # First subplot
        ax1[0].plot(t, dx_p_val[:, 0])
        ax1[0].axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)
        ax1[0].set_ylabel(r'$\dot{x}_p$ (m/s)')
        ax1[0].set_title('Floating Arm Trebuchet Simulation with Pivot Friction')

        # Second subplot
        ax1[1].plot(t, dx_p_val[:, 1])
        ax1[1].axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)
        ax1[1].set_ylabel(r'$\dot{y}_p$ (m/s)')
        ax1[1].set_xlabel('Time (s)')

        # Third subplot (magnitude + angle)
        ax_left = ax1[2]
        ax_right = ax_left.twinx()

        speed = np.linalg.norm(dx_p_val, axis=1)
        angle = np.arctan2(dx_p_val[:, 1], dx_p_val[:, 0])

        ax_left.plot(t, speed, label='Speed', color='tab:blue')
        ax_left.axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)
        ax_left.set_ylabel(r'$||\dot{x}_p||$ (m/s)', color='tab:blue')
        ax_left.tick_params(axis='y', labelcolor='tab:blue')

        ax_right.plot(t, angle, label='Angle', color='tab:orange')
        ax_right.axvline(optimal_release_time, color='gray', linestyle='--', linewidth=1)
        ax_right.axhline(np.pi / 4, color='gray', linestyle='--', linewidth=1)  # 45° line
        ax_right.set_ylabel(r'$\angle \dot{x}_p$ (rad)', color='tab:orange')
        ax_right.tick_params(axis='y', labelcolor='tab:orange')

        ax_left.set_xlabel('Time (s)')
        plt.show()
    return max_range_ind, optimal_release_state, x_p_val[max_range_ind], dx_p_val[max_range_ind], range_vals[max_range_ind], t[max_range_ind]

def plot_animation(t, y, y_release, params):
    # --- Real-time Animation ---
    # Extract data
    theta_vals = y[0]


    # Compute point positions
    def get_arm_coords(i):
        state = {theta1: y[0,i], dtheta1: y[1,i], theta2: y[2,i], dtheta2: y[3,i]}
        cw = np.array(x_cw.subs(params).subs(state))
        pivot = np.array(x_pivot.subs(params).subs(state))
        prj = np.array(x_p.subs(params).subs(state))
        end = np.array(x_end.subs(params).subs(state))
        return cw, pivot, end, prj

    # --- Compute Projectile Velocity ---
    theta_vals = y[0]
    dtheta_vals = y[1]

    # --- Setup Animation + Velocity Plot ---
    fig, ax_anim  = plt.subplots(1, 1, figsize=(10, 10))

    # Animation plot
    ax_anim.set_xlim(-6, 70)
    ax_anim.set_ylim(-6, 50)
    ax_anim.grid()
    ax_anim.set_aspect('equal')
    ax_anim.set_title("Floating Arm Trebuchet Animation with Pivot Friction")
    ax_anim.set_xlabel("x (m)")
    ax_anim.set_ylabel("y (m)")

    x_p_val = y_release[0]
    dx_p_val= y_release[1]
    t_traj = np.linspace(0, 20, 1000).reshape(1, 1000)

    # Assuming the acceleration is gravity acting in the second dimension
    acceleration = np.array([0, -9.81])  # gravity acceleration in the second dimension

    # Reshape t_traj to be (1000, 1) to allow broadcasting with acceleration
    t_traj_reshaped = t_traj.T  # Now t_traj_reshaped is shape (1000, 1)

    # Calculate the trajectory
    x_p_traj = x_p_val + dx_p_val * t_traj_reshaped + 0.5 * acceleration * t_traj_reshaped**2
        
    # Find index where y-position becomes non-positive
    y_vals = x_p_traj[:,1]
    valid_idx = np.where(y_vals >= 0)[0]
    if valid_idx.size > 0:
        last_valid = valid_idx[-1]
        ax_anim.plot(x_p_traj[:last_valid+1,0], y_vals[:last_valid+1])

    arm_line, = ax_anim.plot([], [], 'o-', lw=3)

    # Init function
    def init():
        arm_line.set_data([], [])
        return (arm_line,)

    # Animation update
    def update(frame):
        cw, pivot, end, prj = get_arm_coords(frame)
        xs = [cw[0], pivot[0], end[0], prj[0]]
        ys = [cw[1], pivot[1], end[1], prj[1]]

        arm_line.set_data(xs, ys)
        return (arm_line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(t), init_func=init,
        blit=True, interval=1000 * (t[1] - t[0])
    )

    plt.tight_layout()
    plt.show()

def plot_trajectories(p_pos, p_vel):
    fig, ax = plt.subplots()

    for i in range(len(p_pos)):
        x_p_val = p_pos[i]
        dx_p_val= p_vel[i]
        t_traj = np.linspace(0, 20, 1000).reshape(1, 1000)

        # Assuming the acceleration is gravity acting in the second dimension
        acceleration = np.array([0, -9.81])  # gravity acceleration in the second dimension

        # Reshape t_traj to be (1000, 1) to allow broadcasting with acceleration
        t_traj_reshaped = t_traj.T  # Now t_traj_reshaped is shape (1000, 1)

        # Calculate the trajectory
        x_p_traj = x_p_val + dx_p_val * t_traj_reshaped + 0.5 * acceleration * t_traj_reshaped**2
        
        # Find index where y-position becomes non-positive
        y_vals = x_p_traj[:,1]
        valid_idx = np.where(y_vals >= 0)[0]
        if valid_idx.size > 0:
            last_valid = valid_idx[-1]
            ax.plot(x_p_traj[:last_valid+1,0], y_vals[:last_valid+1])
        
        ax.grid()
    # ax.set_aspect('equal')
    plt.show()

t_full, y_full, _ = simulate(params)
max_range_ind, optimal_release_state, x_p_val, dx_p_val, range, release_time = state_plot(t_full, y_full, params)
# t_full, y_full, _ = simulate(params, release_time)
plot_animation(t_full, y_full, [x_p_val, dx_p_val], params)