from matplotlib import pyplot as plt
import json


def draw_pic(points, scales=None):
    # 将点列表分解为两个列表：x坐标和y坐标
    x, y = zip(*points)

    # 创建一个图表
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    plt.plot(x, y, marker='', linestyle='dashdot', color='green')  # 'o'表示点的样式
    # 在相同的坐标上绘制不同大小的点
    plt.scatter(x, y, s=scales, color='green')

    # 设置图表的标题和坐标轴标签
    plt.title("Time per Ratio")
    plt.xlabel("Time")
    plt.ylabel("Ratio")

    # 显示图表
    plt.show()


json_file_path = r"D:\wise_transportation\wrong-way-cycling\mmyolo\data\78-2.json"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
points = []
scales = []
for pack in data:
    t, f = pack['right'], pack['wrong']
    points.append((pack['time'], f / (t + f)))
    scales.append((t + f) * 20)
draw_pic(points, scales)
