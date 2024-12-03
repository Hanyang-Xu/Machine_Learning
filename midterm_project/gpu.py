import torch
print(torch.cuda.is_available())  # 检查是否可以使用 GPU
print(torch.cuda.device_count())  # 打印可用 GPU 的数量
print(torch.cuda.current_device())  # 获取当前 GPU 的索引
print(torch.cuda.get_device_name(0))  # 获取 GPU 名称
