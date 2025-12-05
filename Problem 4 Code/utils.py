import numpy as np
from scipy.integrate import solve_ivp
def lorenz96(t, x, F=8):
    """Lorenz 96 model with constant forcing"""
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt


def lorenz96_twoscale(t, u, F=8.0, N=40, n=5, h=1.0, c=10.0, b=10.0):

    u = np.asarray(u, dtype=float)
    if u.size != (n + 1) * N:
        raise ValueError(f"len(u) = {u.size}, 但期望 (n+1)*N = {(n+1)*N}")

    U = u.reshape((n + 1, N))   # 对应 Julia reshape(u, n+1, N)
    x = U[0, :]                 # Julia: u[1, :]
    y = U[1:, :]                # Julia: u[2:end, :], 形状 (n, N)

    dx = np.zeros(N, dtype=float)
    dy = np.zeros((n, N), dtype=float)

    # ----------------- 慢变量 x_i -----------------
    for i in range(N):  # i = 0..N-1
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N

        dx[i] = (
            (x[ip1] - x[im2]) * x[im1]
            - x[i]
            + F
            - h * c / b * np.sum(y[:, i])
        )

        # ----------------- 快变量 y_{j,i} -----------------
        for j in range(n):  # j = 0..n-1 (对应 Julia 的 1..n)

            if j == n - 1:          # Julia: j == n
                jp1 = 0             # 1
                jp2 = 1             # 2
                jm1 = n - 2         # n-1
                ip1_y = (i + 1) % N
                ip2_y = (i + 1) % N
                im1_y = i

            elif j == n - 2:        # Julia: j == n-1
                jp1 = n - 1         # n
                jp2 = 0             # 1
                jm1 = n - 3         # n-2
                ip1_y = i
                ip2_y = (i + 1) % N
                im1_y = i

            elif j == 0:            # Julia: j == 1
                jp1 = 1             # 2
                jp2 = 2             # 3
                jm1 = n - 1         # n
                ip1_y = i
                ip2_y = i
                im1_y = (i - 1) % N

            else:                   # Julia: 2 <= j <= n-2
                jp1 = j + 1
                jp2 = j + 2
                jm1 = j - 1
                ip1_y = i
                ip2_y = i
                im1_y = i

            dy[j, i] = (
                c * b * y[jp1, ip1_y] * (y[jm1, im1_y] - y[jp2, ip2_y])
                - c * y[j, i]
                + h * c / b * x[i]
            )

    du = np.empty((n + 1, N), dtype=float)
    du[0, :] = dx
    du[1:, :] = dy

    return du.ravel()

def integral_system(state, steps, system, dt=0.005,
                    method='RK45', rtol=1e-8, atol=1e-10):
    """
    使用 solve_ivp 以固定步长输出，system 不再接收额外参数，
    只需形如 system(t, y)（内部自己用默认参数）。
    """
    state = np.asarray(state, dtype=float)

    T = steps * dt
    t_eval = np.arange(0.0, T + dt/100.0, dt)
    t_span = (0.0, T)

    sol = solve_ivp(
        fun=system,       # 不再用 lambda，也不传 kwargs
        t_span=t_span,
        y0=state,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    return sol



def rk4_step_l96(x, dt, F=8.0):
    """One RK4 step for Lorenz-96."""
    k1 = lorenz96(0.0, x, F)
    k2 = lorenz96(0.0, x + 0.5 * dt * k1, F)
    k3 = lorenz96(0.0, x + 0.5 * dt * k2, F)
    k4 = lorenz96(0.0, x + dt * k3, F)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0



