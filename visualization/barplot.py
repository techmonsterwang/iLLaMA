import matplotlib.pyplot as plt
import numpy as np

# 数据设置
colored_bar_values = np.array([73.9, 74.3, 74.5, np.nan, 71.9, 72.6, 73.2, 74.3, 75.0])
gray_bar_values = np.array([81.4, 82.0, 81.7, np.nan, 80.6, 81.2, 81.2, 81.3, 81.6])
star_values = np.array([3.450, 3.407, 3.406, np.nan, 3.599, 3.618, 3.531, 2.990, 2.955])
# 生成索引，逆序以满足从顶部开始的要求
index = np.arange(len(colored_bar_values))[::-1]  # 逆序索引

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(4,5))

# 绘制带颜色的条形bar，使用逆序的'spring'色彩映射
color_map = plt.get_cmap('spring')
# 逆序应用色彩映射
colors = color_map(np.linspace(1, 0, len(colored_bar_values)))
colored_bars = ax1.barh(index, colored_bar_values, color=colors, label='Accuracy')

# 绘制浅灰色条形bar，确保等宽
gray_bars = ax1.barh(index, gray_bar_values, color='lightgray', alpha=0.5, label='Training Loss')

# 设置坐标轴
ax1.set_xlabel('Accuracy / Training Loss for Tiny Regime')
ax1.set_xlim(63, 85)  # 更新x轴范围
ax1.set_yticks(index)
ax1.set_yticklabels([])  # 纵轴没有实际意义，所以不显示标签

# 绘制星号和折线
ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
ax2.plot(star_values, index, 'r*-')  # 使用红色星号和线
ax2.set_xlim(0.8, 3.7)  # 根据star_values的范围调整
ax2.axis('off')  # 隐藏star标记的坐标轴

plt.tight_layout()
plt.savefig('bar_charts.svg', format='svg')
plt.savefig('bar_charts.pdf', format='pdf')

plt.show()
