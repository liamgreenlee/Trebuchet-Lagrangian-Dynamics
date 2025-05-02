import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp
import matplotlib.animation as animation

# --- Symbolic Setup ---
t = sp.symbols('t')
# Generalized coordinates
theta = sp.Function('theta')(t)
dtheta = theta.diff(t)
# Physical parameters
m_cw, m_p, m_arm, L1, L2, g, c1, c2, c3 = sp.symbols('m_cw m_p m_arm L1 L2 g c1 c2 c3')

# Arm length
L = L1 + L2

# Counterweight and projectile positions
x_cw = sp.Matrix([[0],[L1*sp.sin(theta)]])
x_p = sp.Matrix([[-L*sp.cos(theta)], [-L2*sp.sin(theta)]])
dx_p = x_p.diff(t)


Rz = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],[sp.sin(theta), sp.cos(theta)]])

M = m_p + m_arm + m_cw
# Center of mass distance from pivot
acom = (0*m_cw - L*m_p - (L/2)*m_arm)/M+L1
# COM position wrt pivot
arm_com = Rz * sp.Matrix([[acom],[0]]) 


# Moment of inertia about pivot using parallel axis theorem
# I_arm = (1/12) * m_arm * L**2 + m_arm * acom**2
I_arm = (1/12) * m_arm * L**2
I_acom = I_arm + m_arm**acom**2
# I_acom = m_cw*L1**2 + m_p*L2**2 + I_arm
I_pivot =  m_cw*L1**2 + m_p*L2**2 + I_arm + I_acom

# --- Kinetic Energy ---
# arm translation velocity + arm angular velocity (where arm includes the projectile and counterweight masses)
T = (1/2)*M*sp.diff(arm_com[0],t)**2 + (1/2)*M*sp.diff(arm_com[1],t)**2 + (1/2)*(I_pivot)*dtheta**2

# --- Potential Energy ---
# Gravity affects both the counterweight and projectile
V_cw = m_cw * g *(L1 * sp.sin(theta))  # Counterweight potential energy
V_proj = m_p * g *(-L2 * sp.sin(theta))  # Projectile potential energy
V_arm = m_arm * g * arm_com[1] # Arm center of mass potential energy

V = V_cw + V_proj + V_arm

# --- Lagrangian ---
Lagr = T - V

# Pivot position and velocity used for frictional damping
x_pivot = -L1*sp.cos(theta)
dx_pivot = x_pivot.diff(t)

eq = sp.Eq(sp.diff(sp.diff(Lagr,dtheta),t) - sp.diff(Lagr,theta) - c1*dx_pivot + c2*dtheta - c3*sp.diff(x_cw[1], t), 0)
ddot_theta = sp.solve(eq, sp.diff(theta,t,t))[0]

# --- Substitute parameter values ---
params = {
    m_cw: 100,
    m_p: 0.4,
    m_arm: 0,
    L1: 1,
    L2: 3.0,
    g: 9.81,
    c1: 0.5,  # Friction coefficient (change as needed)
    c2: 0,  # Friction coefficient (change as needed)
    c3: 0
}
# the density of wood is about 500kg/m^3
# assume 2in by 2in (5cm by 5cm) wood arm
params[m_arm] = (0.03)**2*(L.subs(params))*500
# The damping factor due to normal force of friction is mu*m*g. Here c1 is mu
params[c1] = M.subs(params)*params[g]*params[c1]

def simulate(params, release=True):
    eom = ddot_theta.subs(params)
    # M_func = sp.lambdify((theta), M.subs(params), 'numpy')
    # Q_func = sp.lambdify((theta, dtheta), Q.subs(params), 'numpy')
    # --- ODE System ---
    def trebuchet_ode(t, y, eom):
        th, dth = y
        ddth = eom.subs({theta: th, dtheta: dth})
        return [dth, ddth]

    # --- Simulate ---
    # y0 = [np.pi/4, 0]  # initial theta, dtheta, x, dx
    d_theta = sp.solve(sp.Eq(dx_pivot, -np.sqrt(2*(9.81)*0.5)), dtheta)[0].subs(params)
    y0 = [np.pi/4, d_theta.subs({theta: np.pi/4})]
    t0 = 0
    t_end = 4

    t_release = t_end
    if release:
        t_release = 0.25

    sol1 = solve_ivp(lambda t, y: trebuchet_ode(t, y, eom), [t0, t_release], y0, t_eval=np.linspace(t0, t_release, int(1000*(t_release/t_end))))

    if release:
        y_release = sol1.y[:,-1]
        release_ind = len(sol1.t)
        params[m_p] = 0
        eom = ddot_theta.subs(params)
        sol2 = solve_ivp(lambda t,y: trebuchet_ode(t, y, eom), [t_release, t_end], y_release, t_eval=np.linspace(t_release, t_end,int(1000*(1-t_release/t_end))))

        t_full = np.hstack((sol1.t, sol2.t[1:]))
        y_full = np.hstack((sol1.y, sol2.y[:, 1:]))
    else:
        t_full = sol1.t
        y_full = sol1.y
        max_vel_ind = np.argmax(np.abs(y_full[1,:]))
        y_release = y_full[:,max_vel_ind]
    return t_full, y_full, y_release

def state_plot():
    # --- Plot Results ---
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t_full, y_full[0])
    plt.ylabel(r'$\theta$ (rad)')
    plt.title('Floating Arm Trebuchet Simulation with Pivot Friction')

    plt.subplot(2, 1, 2)
    plt.plot(t_full, y_full[1])
    plt.ylabel(r'$\dot{\theta}$ (rad/s)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def plot_animation(t, y, y_release):
    # --- Real-time Animation ---
    # Extract data
    theta_vals = y[0]

    # Geometry
    L1_val = params[sp.Symbol('L1')]
    L2_val = params[sp.Symbol('L2')]

    # Compute point positions
    def get_arm_coords(i):
        cw = np.array([0, L1_val * np.sin(theta_vals[i])])
        pivot = np.array([-L1_val*np.cos(theta_vals[i]), 0])
        prj = np.array([-(L1_val + L2_val) * np.cos(theta_vals[i]), -(L2_val) * np.sin(theta_vals[i])])
        return cw, pivot, prj

    # --- Compute Projectile Velocity ---
    theta_vals = y[0]
    dtheta_vals = y[1]

    L1_val = params[L1]
    L2_val = params[L2]

    # --- Setup Animation + Velocity Plot ---
    fig, ax_anim  = plt.subplots(1, 1, figsize=(10, 10))

    # Animation plot
    ax_anim.set_xlim(-6, 40)
    ax_anim.set_ylim(-6, 20)
    ax_anim.grid()
    ax_anim.set_aspect('equal')
    ax_anim.set_title("Floating Arm Trebuchet Animation with Pivot Friction")
    ax_anim.set_xlabel("x (m)")
    ax_anim.set_ylabel("y (m)")
    x_p_val = np.array(x_p.subs(params).subs({theta: y_release[0], dtheta: y_release[1]})).astype(float)
    dx_p_val = np.array(dx_p.subs(params).subs({theta: y_release[0], dtheta: y_release[1]})).astype(float)


    # Compute full trajectory
    t_traj = np.linspace(0,t[-1], 1000)
    x_p_traj = x_p_val + dx_p_val*t_traj + np.array([[0],[0.5*-9.81]])*t_traj**2
    ax_anim.plot(x_p_traj[0,:], x_p_traj[1,:])
    arm_line, = ax_anim.plot([], [], 'o-', lw=3)

    # Init function
    def init():
        arm_line.set_data([], [])
        return (arm_line,)

    # Animation update
    def update(frame):
        cw, pivot, prj = get_arm_coords(frame)
        xs = [cw[0], pivot[0], prj[0]]
        ys = [cw[1], pivot[1], prj[1]]

        arm_line.set_data(xs, ys)
        return (arm_line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(t), init_func=init,
        blit=True, interval=1000 * (t[1] - t[0])
    )

    plt.tight_layout()
    plt.show()

def plot_trajectories(y_release):
    fig, ax = plt.subplots()

    for y in y_release:
        x_p_val = np.array(x_p.subs(params).subs({theta: y[0], dtheta: y[1]})).astype(float)
        dx_p_val = np.array(dx_p.subs(params).subs({theta: y[0], dtheta: y[1]})).astype(float)
        
        t_traj = np.linspace(0, 20, 1000)
        x_p_traj = x_p_val + dx_p_val * t_traj + np.array([[0], [0.5 * -9.81]]) * t_traj**2

        # Find index where y-position becomes non-positive
        y_vals = x_p_traj[1, :]
        valid_idx = np.where(y_vals >= 0)[0]
        if valid_idx.size > 0:
            last_valid = valid_idx[-1]
            ax.plot(x_p_traj[0, :last_valid+1], y_vals[:last_valid+1])
        
        ax.grid()
    ax.set_aspect('equal')
    plt.show()


release_states = []
for length in np.linspace(0.1, 3.9, 20):
    params[L1] = length
    params[L2] = 4 - length
    t_full, y_full, y_release = simulate(params, False)
    plot_animation(t_full, y_full, y_release)
    release_states.append(y_release)

# state_plot()
# plot_animation(t_full, y_full, y_release)
plot_trajectories(release_states)