import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ===== 1. 目标能量函数 E(x,y) =====
def energy(x, y):
    return (x - 1.0)**2 + (y + 2.0)**2  # 最优解 (1, -2)

def energy_prefix(ys, j):
    x = ys[:, 0]
    y = ys[:, 1]

    if j == 0:
        # 只看 x 的偏差
        return (x - 1.0) ** 2          # E^0(x)
    elif j == 1:
        # 看完整的 (x, y)
        return (x - 1.0) ** 2 + (y + 2.0) ** 2  # E^1(x, y)
    else:
        raise ValueError("j 只能是 0 或 1（这个 toy 是 2 维）")
    
# ===== 2. Algorithm 2（极简自回归版，无训练） =====
def autoregressive_optimizer_2d(
    num_samples=10,
    N_iters=5,
    sigma_init=1.0,
    K=0.5,
    bounds=(-5, 5),
):
    low, high = bounds

    # 初始化样本 (num_samples, 2)
    ys = np.random.uniform(low, high, size=(num_samples, 2))
    sigma = sigma_init

    # 存储轨迹 (每一帧的样本)
    frames = []

    # 保存初始状态
    frames.append(ys.copy())

    # 维度 m = 2
    for it in range(N_iters):
        for j in range(2):
            E = energy_prefix(ys, j)   # ★ 用前缀能量，而不是 full energy

            negE = -E
            negE -= np.max(negE)
            p = np.exp(negE)
            p /= p.sum()

            idx = np.random.choice(len(ys), size=len(ys), replace=True, p=p)
            ys = ys[idx]

            noise = np.zeros_like(ys)
            noise[:, j] = sigma * np.random.randn(len(ys))  # 只更新第 j 维
            ys = ys + noise
            ys = np.clip(ys, low, high)

            frames.append(ys.copy())

        sigma *= K

    return frames


# ===== 3. 动画可视化 =====

def visualize_optimization(frames, save_gif=False, gif_filename='optimization.gif'):
    """
    可视化优化过程
    
    Args:
        frames: 优化过程中每一步的样本点
        save_gif: 是否保存为gif动画
        gif_filename: gif文件名
    """
    # 画能量函数等高线
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = energy(X, Y)

    fig, ax = plt.subplots(figsize=(18, 18))
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)
    
    # 标记最优点
    ax.plot(1.0, -2.0, 'g*', markersize=15, label='Optimal Point (1, -2)')
    
    # 初始化散点图
    scat = ax.scatter([], [], s=80, c='red', alpha=0.8, edgecolors='black', linewidth=1)
    
    # 设置坐标轴
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 添加颜色条
    plt.colorbar(contour, ax=ax, label='Energy')
    
    def update(frame_id):
        # 更新散点图位置
        scat.set_offsets(frames[frame_id])
        
        # 计算当前帧的能量
        current_points = frames[frame_id]
        energies = [energy(pt[0], pt[1]) for pt in current_points]
        min_energy = min(energies)
        avg_energy = np.mean(energies)
        
        # 更新标题，显示步骤和能量信息
        ax.set_title(f'Algorithm 2 - Step {frame_id}\n'
                    f'Min Energy: {min_energy:.3f}, Avg Energy: {avg_energy:.3f}\n'
                    f'Samples: {len(current_points)}', fontsize=14)
        
        return scat,

    # 创建动画
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000, blit=True, repeat=True
    )
    
    # 保存为gif（可选）
    if save_gif:
        ani.save(gif_filename, writer='pillow', fps=1)
        print(f"动画已保存为: {gif_filename}")
    
    # 显示动画
    plt.tight_layout()
    plt.show()
    
    return ani


def plot_convergence(frames):
    """
    绘制收敛曲线
    """
    min_energies = []
    avg_energies = []
    
    for frame in frames:
        energies = [energy(pt[0], pt[1]) for pt in frame]
        min_energies.append(min(energies))
        avg_energies.append(np.mean(energies))
    
    plt.figure(figsize=(18, 18))
    
    plt.subplot(1, 2, 1)
    plt.plot(min_energies, 'b-o', label='Min Energy')
    plt.plot(avg_energies, 'r-s', label='Avg Energy')
    plt.axhline(y=0, color='g', linestyle='--', alpha=0.7, label='Global Optimum')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Energy Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 绘制最终分布
    final_points = frames[-1]
    x_coords = [pt[0] for pt in final_points]
    y_coords = [pt[1] for pt in final_points]
    
    plt.scatter(x_coords, y_coords, c='red', s=80, alpha=0.8, edgecolors='black')
    plt.plot(1.0, -2.0, 'g*', markersize=15, label='Optimal Point')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Final Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# 运行可视化
if __name__ == "__main__":
    # 运行优化算法
    print("Running Algorithm 2...")
    frames = autoregressive_optimizer_2d(num_samples=15, N_iters=5, sigma_init=1.5)
    
    print(f"Total frames: {len(frames)}")
    print(f"Initial energy range: {min([energy(pt[0], pt[1]) for pt in frames[0]]):.3f} - {max([energy(pt[0], pt[1]) for pt in frames[0]]):.3f}")
    print(f"Final energy range: {min([energy(pt[0], pt[1]) for pt in frames[-1]]):.3f} - {max([energy(pt[0], pt[1]) for pt in frames[-1]]):.3f}")
    
    # 显示动画（可以选择保存为gif）
    ani = visualize_optimization(frames, save_gif=False)
    
    # 显示收敛曲线
    plot_convergence(frames)

