import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.font_manager as fm

# 设置中文显示
fm.fontManager.addfont('/home/cxm/NotoSansCJKsc-Regular.otf')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 能量函数（示例：二维空间中的非凸函数，包含多个局部最小值）
def energy_function(a):
    # 增加复杂度以展示非凸特性
    term1 = (a[:, 0] - 0.3)**2 + (a[:, 1] - 0.3)** 2
    term2 = 0.5 * ((a[:, 0] + 0.2)**2 + (a[:, 1] + 0.2)** 2)
    term3 = np.sin(5 * a[:, 0]) * np.sin(5 * a[:, 1]) * 0.1
    return term1 + term2 + term3

# 优化器函数，保存每轮迭代的样本和能量
def derivative_free_optimizer_track(N_samples=100, N_iters=10, sigma_init=0.33, K=0.2, a_min=-1, a_max=1):
    samples = np.random.uniform(a_min, a_max, (N_samples, 2))  # 初始样本（二维）
    sigma = sigma_init
    history = []
    history.append((samples.copy(), energy_function(samples)))
    
    for t in range(N_iters):
        energies = energy_function(samples)
        exp_energies = np.exp(-energies)
        probs = exp_energies / np.sum(exp_energies)
        
        if t < N_iters - 1:
            # 按概率重采样
            indices = np.random.choice(range(N_samples), size=N_samples, p=probs)
            new_samples = samples[indices]
            # 添加噪声
            noise = np.random.normal(0, sigma, (N_samples, 2))
            new_samples += noise
            # 裁剪到有效范围
            new_samples = np.clip(new_samples, a_min, a_max)
            samples = new_samples
            sigma *= K  # 缩小噪声
            history.append((samples.copy(), energy_function(samples)))
    
    # 找到最终最优解
    final_energies = energy_function(samples)
    best_idx = np.argmin(final_energies)
    a_hat = samples[best_idx]
    final_energy = final_energies[best_idx]
    
    return a_hat, final_energy, history

# 生成能量函数的等高线数据
def generate_energy_contour(a_min=-1, a_max=1, grid_size=100):
    x = np.linspace(a_min, a_max, grid_size)
    y = np.linspace(a_min, a_max, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])  # 转换为(N, 2)格式
    Z = energy_function(grid_points).reshape(grid_size, grid_size)
    return X, Y, Z

# 可视化主函数
def visualize_optimization():
    # 运行优化器（增加迭代次数以便观察）
    a_hat, final_energy, history = derivative_free_optimizer_track(N_iters=15)
    N_iters = len(history) - 1  # 迭代次数
    
    # 生成能量函数等高线
    X, Y, Z = generate_energy_contour()
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("EBM无导数优化器迭代过程可视化", fontsize=15)
    
    # 左侧：等高线 + 样本点动态展示
    contour = ax1.contourf(X, Y, Z, levels=20, cmap="viridis_r")  # 能量低=浅色
    plt.colorbar(contour, ax=ax1, label="能量值")
    ax1.set_xlabel("x轴")
    ax1.set_ylabel("y轴")
    ax1.set_title("样本分布与能量等高线")
    # 标记理论最优解（这里通过网格搜索找近似最优）
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    grid_energies = energy_function(grid_points)
    best_theo_idx = np.argmin(grid_energies)
    best_theo = grid_points[best_theo_idx]
    ax1.scatter(best_theo[0], best_theo[1], c="red", s=100, marker="*", label="理论最优解")
    # 初始化散点图（关键修复：使用二维空数组）
    scatter = ax1.scatter([], [], c=[], cmap="viridis_r", edgecolors="black", alpha=0.7)
    text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, 
                    verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # 右侧：能量变化趋势
    ax2.set_xlabel("迭代次数")
    ax2.set_ylabel("平均能量值")
    ax2.set_title("迭代过程中平均能量变化")
    avg_energies = [np.mean(energies) for (_, energies) in history]
    line, = ax2.plot([], [], "b-o", markersize=5)
    ax2.axhline(y=np.min(grid_energies), color="r", linestyle="--", label="理论最小能量")
    ax2.legend()
    
    # 初始化动画（修复：确保散点图数据维度正确）
    def init():
        scatter.set_offsets(np.empty((0, 2)))  # 明确设置为二维空数组
        scatter.set_array(np.array([]))
        line.set_data([], [])
        text.set_text("")
        return scatter, line, text
    
    # 更新动画
    def update(frame):
        samples, energies = history[frame]
        # 更新样本点
        scatter.set_offsets(samples)  # 样本是二维数组，符合要求
        scatter.set_array(energies)
        # 更新能量曲线
        line.set_data(range(frame+1), avg_energies[:frame+1])
        ax2.set_xlim(0, N_iters)
        ax2.set_ylim(0, max(avg_energies) * 1.1)
        # 更新迭代信息
        text.set_text(f"迭代次数: {frame}\n当前平均能量: {avg_energies[frame]:.4f}")
        return scatter, line, text
    
    # 创建动画（修复：禁用blit以避免某些环境下的属性错误）
    ani = FuncAnimation(
        fig, update, frames=range(N_iters+1), 
        init_func=init, interval=800, blit=False  # 关键修复：blit=False
    )
    
    # 标记最终优化结果
    ax1.scatter(a_hat[0], a_hat[1], c="blue", s=150, marker="X", 
                label=f"优化结果: {a_hat.round(4)}")
    ax1.legend()
    
    plt.tight_layout()
    plt.show()
    return ani

# 运行可视化
if __name__ == "__main__":
    animation = visualize_optimization()
