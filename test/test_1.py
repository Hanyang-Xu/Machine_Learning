import matplotlib.pyplot as plt

# 数据定义
actuators = [
    {"name": "ZABER LAC Series", "frequency": 50, "power_output_per_unit_volume": 80000},
    {"name": "LINAK LA25", "frequency": 25, "power_output_per_unit_volume": 40000},
    {"name": "LINAK LA14", "frequency": 25, "power_output_per_unit_volume": 30000},
    {"name": "Timotion TA16 Series", "frequency": 50, "power_output_per_unit_volume": 25000},
    {"name": "EBAD TiNi™ Pin Puller Nano P5", "frequency": 0.5, "power_output_per_unit_volume": 10000},
    {"name": "EBAD TiNi™ Pin Puller P5", "frequency": 0.5, "power_output_per_unit_volume": 8000},
    {"name": "EBAD TiNi™ Pin Puller P100", "frequency": 0.5, "power_output_per_unit_volume": 6000},
    {"name": "HR1 Nanomotion Motor", "frequency": 100, "power_output_per_unit_volume": 1000000},
    {"name": "P-601 PiezoMove Linear Actuator", "frequency": 200, "power_output_per_unit_volume": 500000},
]

# 提取数据
names = [actuator["name"] for actuator in actuators]
frequencies = [actuator["frequency"] for actuator in actuators]
power_outputs = [actuator["power_output_per_unit_volume"] for actuator in actuators]

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(frequencies, power_outputs, color='blue')

# 添加标签和标题
for i, name in enumerate(names):
    plt.text(frequencies[i], power_outputs[i], name, fontsize=9)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Output per Unit Volume (W/m³)')
plt.title('Frequency vs Power Output per Unit Volume for Linear Actuators')
plt.grid(True, which="both", ls="--")
plt.show()
