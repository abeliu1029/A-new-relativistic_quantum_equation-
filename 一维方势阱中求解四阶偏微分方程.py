import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_wave_equation_1d_advanced(grid_size, time_steps, dt, dx):
    # 初始化网格 (w(x, t))
    w = np.zeros((grid_size, time_steps))

    # 定义空间坐标
    x = np.linspace(0, 1, grid_size)

    # 初始条件
    w[:, 0] = np.sin(np.pi * x)  # 初始位移
    w[:, 1] = w[:, 0]            # 初始速度为 0

    # 边界条件（固定边界）
    w[0, :] = 0
    w[-1, :] = 0

    # 迭代求解
    for t in range(2, time_steps - 2):
        for i in range(2, grid_size - 2):
            # 计算各项导数
            w_tttt = (w[i, t+2] - 4*w[i, t+1] + 6*w[i, t] - 4*w[i, t-1] + w[i, t-2]) / dt**4
            w_xxxx = (w[i+2, t] - 4*w[i+1, t] + 6*w[i, t] - 4*w[i-1, t] + w[i-2, t]) / dx**4
            w_xxtt = ((w[i+1, t+1] - 2*w[i, t+1] + w[i-1, t+1] -
                       2*w[i+1, t-1] + 2*w[i, t-1] - w[i-1, t-1]) / (dx**2 * dt**2))
            
            # 更新网格点
            w[i, t + 1] = w_tttt - w_xxxx - w_xxtt

    return x, w

def create_animation_1d_advanced(x, solution):
    fig, ax = plt.subplots()
    line, = ax.plot(x, solution[:, 0], label="Wave", color="blue")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("Wave Amplitude")
    ax.legend()

    def update(t):
        line.set_ydata(solution[:, t])
        ax.set_title(f"Time Step {t}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=solution.shape[1], interval=50, blit=True)

    # 保存为 GIF
    ani.save('wave_equation_1d_advanced.gif', writer='pillow', fps=20)
    plt.show()

if __name__ == "__main__":
    grid_size = 100
    time_steps = 50
    dt = 0.01
    dx = 1 / (grid_size - 1)

    x, solution = solve_wave_equation_1d_advanced(grid_size, time_steps, dt, dx)
    create_animation_1d_advanced(x, solution)
