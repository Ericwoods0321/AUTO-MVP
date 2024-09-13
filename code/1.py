import torch

view_types = ["A2C", "A3C", "A4C", "A5C", "PLAX"]
view_mapping = {view: index for index, view in enumerate(view_types)}
print(view_mapping)
view = "A3C"  # 单个视图类型

# 将视图类型转换为整数索引
view_index = view_mapping[view]
print(view_index)
# 进行 one-hot 编码
one_hot = torch.eye(len(view_types))[view_index]

print(one_hot)
