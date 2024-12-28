import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Hiragino Sans GB']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def solve_wave_equation_3d(grid_size, time_steps, dt, dx, tol=1e-6):
    # 初始化网格 (w(x, y, z, t))
    w = np.zeros((grid_size, grid_size, grid_size, time_steps))
    x = np.linspace(0, np.pi, grid_size)
    y = np.linspace(0, np.pi, grid_size)
    z = np.linspace(0, np.pi, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 初始条件
    w[:, :, :, 0] = np.sin(X) * np.sin(Y) * np.sin(Z)
    w[:, :, :, 1] = w[:, :, :, 0]

    # 边界条件：方势阱内边界值为零
    w[0, :, :, :] = w[-1, :, :, :] = 0
    w[:, 0, :, :] = w[:, -1, :, :] = 0
    w[:, :, 0, :] = w[:, :, -1, :] = 0

    # 迭代求解
    for t in range(2, time_steps - 2):
        for i in range(2, grid_size - 2):
            for j in range(2, grid_size - 2):
                for k in range(2, grid_size - 2):
                    # 四阶时间导数 w_tttt
                    w_tttt = (w[i, j, k, t+2] - 4*w[i, j, k, t+1] + 6*w[i, j, k, t] -
                              4*w[i, j, k, t-1] + w[i, j, k, t-2]) / dt**4

                    # 四阶空间导数 w_xxxx, w_yyyy, w_zzzz
                    w_xxxx = (w[i+2, j, k, t] - 4*w[i+1, j, k, t] + 6*w[i, j, k, t] -
                              4*w[i-1, j, k, t] + w[i-2, j, k, t]) / dx**4

                    w_yyyy = (w[i, j+2, k, t] - 4*w[i, j+1, k, t] + 6*w[i, j, k, t] -
                              4*w[i, j-1, k, t] + w[i, j-2, k, t]) / dx**4

                    w_zzzz = (w[i, j, k+2, t] - 4*w[i, j, k+1, t] + 6*w[i, j, k, t] -
                              4*w[i, j, k-1, t] + w[i, j, k-2, t]) / dx**4

                    # 计算空间和时间的混合项 w_xxtt, w_yytt, w_zztt
                    w_xxtt = (w[i+1, j, k, t+1] - 2*w[i, j, k, t+1] + w[i-1, j, k, t+1] -
                              2*w[i+1, j, k, t-1] + 2*w[i, j, k, t-1] - w[i-1, j, k, t-1]) / (dx**2 * dt**2)

                    w_yytt = (w[i, j+1, k, t+1] - 2*w[i, j, k, t+1] + w[i, j-1, k, t+1] -
                              2*w[i, j+1, k, t-1] + 2*w[i, j, k, t-1] - w[i, j-1, k, t-1]) / (dx**2 * dt**2)

                    w_zztt = (w[i, j, k+1, t+1] - 2*w[i, j, k, t+1] + w[i, j, k-1, t+1] -
                              2*w[i, j, k+1, t-1] + 2*w[i, j, k, t-1] - w[i, j, k-1, t-1]) / (dx**2 * dt**2)

                    # 更新网格点
                    w[i, j, k, t+1] = w_tttt - w_xxxx - w_xxtt - w_yyyy - w_yytt - w_zzzz - w_zztt

    return w

def create_animation(grid_size, solution):
    x = np.linspace(0, np.pi, grid_size)
    y = np.linspace(0, np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)

    surf = [ax.plot_surface(X, Y, solution[:, :, grid_size // 2, 0], cmap='viridis')]

    def update(t):
        ax.clear()
        ax.set_zlim(-1, 1)
        ax.set_title(f"Time Step {t}")
        surf[0] = ax.plot_surface(X, Y, solution[:, :, grid_size // 2, t], cmap='viridis')
        return surf

    ani = animation.FuncAnimation(fig, update, frames=solution.shape[-1], interval=100, blit=False)
    ani.save('wave_equation_3d_square_potential.mp4', writer='ffmpeg', fps=20)
    plt.show()

if __name__ == "__main__":
    grid_size = 50
    time_steps = 50
    dt = 0.3
    dx = np.pi / (grid_size - 1)

    solution = solve_wave_equation_3d(grid_size, time_steps, dt, dx)
    create_animation(grid_size, solution)
