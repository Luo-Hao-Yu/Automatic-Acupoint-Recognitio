import numpy as np
import matplotlib.pyplot as plt
import os

# 数据
x = np.linspace(1, 11, 10)
y = [0.02246803, 0.03832424, 0.02346068, 0.02780752, 0.02820636, 0.02525377, 0.03302605, 0.03854782, 0.02391011,
     0.02350967]

# 绘制折线图
plt.plot(x, y)
plt.title('Acupuncture point error')
plt.xlabel('number of Graph')
plt.ylabel('error')
plt.axhline(y=0.05, color='r', linestyle='-')

# 保存图片
save_folder = "lineGraph"
save_path = os.path.join(save_folder, '10Graph')
plt.savefig(save_path)

plt.show()