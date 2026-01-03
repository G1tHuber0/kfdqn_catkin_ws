import numpy as np
import matplotlib.pyplot as plt
import math

# ================= 配置区 =================
# 你可以随意修改这里的数量和数值
CENTERS =  [0.5, 1.5]
WIDTHS  =  [0.4, 0.6]
# ==========================================

def gaussian_mf(x, mu, w):
    sigma = w
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def main():
    # 1. 自动转换宽度为列表格式
    c_array = np.array(CENTERS)
    if isinstance(WIDTHS, (int, float)):
        w_array = np.array([WIDTHS] * len(c_array))
    else:
        w_array = np.array(WIDTHS)

    # 2. 自适应宽度计算 (X轴)
    # 计算每个模糊集的边界（中心 +/- 1.5倍宽度，确保高斯尾部也能显示）
    left_edge = np.min(c_array - 3 * w_array)
    right_edge = np.max(c_array + 3 * w_array)
    
    x = np.linspace(left_edge, right_edge, 1000)

    # 3. 绘图
    plt.figure(figsize=(12, 6))

    for i, (mu, w) in enumerate(zip(c_array, w_array)):
        y = gaussian_mf(x, mu, w)
        line, = plt.plot(x, y, label=f'Set {i} (μ={mu}, w={w})', linewidth=2)
        plt.fill_between(x, y, color=line.get_color(), alpha=0.1)

    # 4. 自适应高度与装饰 (Y轴)
    plt.ylim(0, 1.1)  # 隶属度固定在 0-1 之间，给顶部留一点点空间
    plt.xlim(left_edge, right_edge) # 强制 X 轴自适应计算出的边缘
    
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Threshold 0.5')
    plt.title("Adaptive Gaussian Membership Functions", fontsize=14)
    plt.xlabel("Input State Range (Auto-scaled)")
    plt.ylabel("Membership Degree")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0)) # 标签放在外侧防止遮挡曲线
    
    plt.tight_layout() # 自动调整布局，防止标签被切掉
    plt.show()

if __name__ == "__main__":
    main()