import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Hiragino Sans GB']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def analytical_solution(grid_size, time_steps, dt):
    x = np.linspace(0, np.pi, grid_size)
    y = np.linspace(0, np.pi, grid_size)
    t = np.linspace(0, time_steps * dt, time_steps)
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')

    # 解析解（假设本征值为简单形式）
    solution = np.sin(X) * np.sin(Y) * np.cos(T) * np.cos(np.sqrt(2) * T)
    return solution

def create_animation(grid_size, solution):
    x = np.linspace(0, np.pi, grid_size)
    y = np.linspace(0, np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)

    surf = [ax.plot_surface(X, Y, solution[:, :, 0], cmap='viridis')]

    def update(t):
        ax.clear()
        ax.set_zlim(-1, 1)
        ax.set_title(f"Time Step {t}")
        surf[0] = ax.plot_surface(X, Y, solution[:, :, t], cmap='viridis')
        return surf

    ani = animation.FuncAnimation(fig, update, frames=solution.shape[-1], interval=100, blit=False)
    ani.save('wave_equation_4th_order_analytical_solution.mp4', writer='ffmpeg', fps=20)
    plt.show()

if __name__ == "__main__":
    grid_size = 50
    time_steps = 50
    dt = 0.3

    solution = analytical_solution(grid_size, time_steps, dt)
    create_animation(grid_size, solution)
