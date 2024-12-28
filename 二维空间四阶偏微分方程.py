
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Hiragino Sans GB']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def solve_wave_equation(grid_size, time_steps, dt, dx, tol=1e-6):
    """
    用有限差分法求解方程：w_{tttt} - w_{xxxx} - w_{xxtt} - w_{yytt} - w_{yyyy} = 0

    参数：
        grid_size: 网格大小（正方形网格）。
        time_steps: 时间步数。
        dt: 时间步长。
        dx: 空间步长。
        tol: 收敛条件的容差。
        max_iter: 最大迭代次数。

    返回：
        解的网格。
    """
    # 初始化网格（w(x, y, t)）
    w = np.zeros((grid_size, grid_size, time_steps))

    # 设置初始条件，创建一个二维的正弦波形
    x = np.linspace(-2*np.pi, 2*np.pi, grid_size)
    y = np.linspace(-2*np.pi, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)

    # 设置w的初始条件为正弦波
    w[:, :, 0] = np.sin(X) * np.cos(Y)

    # 初始时刻的时间导数为零
    w[:, :, 1] = 0

    # 边界条件：可以根据实际问题设置
    # w[0, :, :] = np.pi  # 上边界
    # w[-1, :, :] = np.pi  # 下边界
    # w[:, 0, :] = np.pi  # 左边界
    # w[:, -1, :] = np.pi  # 右边界

    w[0, :, :] = 0  # 上边界
    w[-1, :, :] = 0  # 下边界
    w[:, 0, :] = 0  # 左边界
    w[:, -1, :] = 0  # 右边界

    # 迭代求解
    for t in range(2, time_steps - 2):
        new_w = w.copy()  # 创建一个新的网格来保存当前步的更新结果

        # 更新内部点，使用有限差分模板
        for i in range(2, grid_size - 2):
            for j in range(2, grid_size - 2):
                # 计算 w_{tttt} (四阶时间导数)
                w_tttt = (w[i, j, t+2] - 4*w[i, j, t+1] + 6*w[i, j, t] - 4*w[i, j, t-1] + w[i, j, t-2]) / dt**4

                # 计算 w_{xxxx} (四阶空间导数，x方向)
                w_xxxx = (w[i+2, j, t] - 4*w[i+1, j, t] + 6*w[i, j, t] - 4*w[i-1, j, t] + w[i-2, j, t]) / dx**4

                # 计算 w_{xxtt} (x方向和时间的二阶导数)
                w_xxtt = (w[i+1, j, t+1] - 2*w[i, j, t+1] + w[i-1, j, t+1] - 2*w[i+1, j, t-1] + 2*w[i, j, t-1] - w[i-1, j, t-1]) / (dx**2 * dt**2)

                # 计算 w_{yytt} (y方向和时间的二阶导数)
                w_yytt = (w[i, j+1, t+1] - 2*w[i, j, t+1] + w[i, j-1, t+1] - 2*w[i, j+1, t-1] + 2*w[i, j, t-1] - w[i, j-1, t-1]) / (dx**2 * dt**2)

                # 计算 w_{yyyy} (四阶空间导数，y方向)
                w_yyyy = (w[i, j+2, t] - 4*w[i, j+1, t] + 6*w[i, j, t] - 4*w[i, j-1, t] + w[i, j-2, t]) / dx**4

                # 更新网格点的值
                new_w[i, j, t] = w_tttt - w_xxxx - w_xxtt/2 - w_yytt/2 - w_yyyy

        # 检查收敛性
        diff = np.max(np.abs(new_w - w))
        if diff < tol:
            print(f"Converged after {t} iterations.")
            break

        w = new_w  # 更新网格值

    return w


def animate_solution(w, interval=50):
    """
    绘制动态图：每个时间步更新解的图像。

    参数：
        w: 方程的解。
        interval: 每帧之间的间隔时间，单位为毫秒。
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 初始化颜色范围
    vmin = np.min(w)
    vmax = np.max(w)

    # 设置第一个帧的图像
    im = ax.imshow(w[:, :, 0], cmap='jet', origin='lower', animated=True, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='Value')
    ax.set_title(' 四阶偏微分方程:$w_{tttt}$ - $w_{xxxx}$ - $w_{xxtt}$ - $w_{yytt}$ - $w_{yyyy}$ = 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def update(frame):
        # 更新每一帧的图像
        im.set_data(w[:, :, frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=w.shape[2], interval=interval, blit=True)

    # 保存动画
    ani.save('wave_equation_animation.gif', writer='imagemagick', fps=30)

    plt.show()

if __name__ == "__main__":
    grid_size = 314  # 网格大小
    time_steps = 100  # 时间步数
    # 下面条件给出特殊图像
    # dt = 0.1  # 时间步长
    # dx = 0.1  # 空间步长
    # dt = 0.2  # 时间步长
    # dx = 0.2  # 空间步长
    # 下面条件会引起发散
    # dt = 0.3  # 时间步长
    # dx = 0.3  # 空间步长
    # 下面条件给出稍微正常图像
    # dt = 0.3  # 时间步长
    # dx = 0.1  # 空间步长
    # 下面条件给出正常图像
    dt = 0.01 # 时间步长
    dx = 0.1  # 空间步长

    solution = solve_wave_equation(grid_size, time_steps, dt, dx)
    animate_solution(solution)
