import numpy as np
import matplotlib.pyplot as plt

# 三维解析解
def analytical_solution_3d(x, y, z, t):
    return np.sin(x) * np.sin(y) * np.sin(z) * np.cos(2 * t) * np.cos(np.sqrt(2) * t)

# 时间范围和空间点集合
t = np.linspace(0, 10, 500)  # 时间从 0 到 10，分为 500 步
x_vals = np.linspace(0, np.pi, 10)  # 选择 5 个 x 值
y_vals = np.linspace(0, np.pi, 10)  # 选择 5 个 y 值
z_vals = np.linspace(0, np.pi, 10)  # 选择 5 个 z 值

# 设置画布大小
plt.figure(figsize=(10, 6))

# 计算每个空间点随时间变化的波动
for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            w_t = analytical_solution_3d(x, y, z, t)
            plt.plot(t, w_t, label=f'x={x:.2f}, y={y:.2f}, z={z:.2f}')

# 图形设置
plt.title("Wave Oscillation at Multiple Spatial Points", fontsize=14)
plt.xlabel("Time $t$", fontsize=12)
plt.ylabel("Amplitude $w$", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.1, 1))
plt.grid(alpha=0.6)
plt.show()
