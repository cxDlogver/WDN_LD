""" 数据集加载 """
import pandas as pd

from model import *

# 2018年无泄漏数据
dir = "../Dataset Generator/"
filename = "no_leak_dataset_2018"
read_dataset(dir, filename)
print("2018年无泄漏数据加载完成")

# 2018年有泄漏数据
filename = "leak_dataset_2018"
read_dataset(dir, filename)
print("2018年有泄漏数据加载完成")

# 2019年无泄漏数据
filename = "no_leak_dataset_2019"
read_dataset(dir, filename)
print("2019年无泄漏数据加载完成")

# 2019年有泄漏数据
filename = "leak_dataset_2019"
read_dataset(dir, filename)
print("2019年有泄漏数据加载完成")

# 模拟DMA C区域泄漏数据，用于拟合 AB区域传感器与水泵流量之间的函数关系
filename = "PUMP_DMA_C"
read_dataset(dir, filename)
print("模拟DMA C区域泄漏数据加载完成")

# 模拟DMA AB区域泄漏数据，用于拟合 C区域传感器与水泵流量之间的函数关系
filename = "PUMP_DMA_AB"
read_dataset(dir, filename)
print("模拟DMA AB区域泄漏数据加载完成")

# 验证压力校正的数据
filename = 'pressure_reset'
read_dataset(dir, filename)
print("验证压力校正的数据加载完成")


# 2018 和 2019 数据 合并
fix_pressure_2018 = read_dataset(File_PATH, "fix_pressure_2018")
fix_pressure_2019 = read_dataset(File_PATH, "fix_pressure_2019")
fix_pressure = pd.concat([fix_pressure_2018, fix_pressure_2019], axis=0)
with open(os.path.join(File_PATH, "fix_pressure.csv"), "w", newline="") as f:
    fix_pressure.to_csv(f, index=True)
with open(os.path.join(File_PATH, "fix_pressure.pickle"), "wb") as f:
    pickle.dump(fix_pressure, f)
print(fix_pressure.shape)

# 2018 和 2019 数据 合并
pressure_2018 = read_dataset(File_PATH, "pressure_2018")
pressure_2019 = read_dataset(File_PATH, "pressure_2019")
pressure = pd.concat([pressure_2018, pressure_2019], axis=0)
with open(os.path.join(File_PATH, "pressure.csv"), "w", newline="") as f:
    pressure.to_csv(f, index=True)
with open(os.path.join(File_PATH, "pressure.pickle"), "wb") as f:
    pickle.dump(pressure, f)
print(pressure.shape)

# 2018 和 2019 泄漏数据合并
leak_demand_2018 = pd.read_csv(os.path.join(File_PATH, "2018_leak_demand.csv"))
leak_demand_2019 = pd.read_csv(os.path.join(File_PATH, "2019_leak_demand.csv"))
leak_demands = pd.concat([leak_demand_2018, leak_demand_2019], axis=0)
leak_demands.fillna(0, inplace=True)
with open(os.path.join(File_PATH, "leak_demands.csv"), "w", newline="") as f:
    leak_demands.to_csv(f, index=False)

# 真实场景泄漏数据
# filename = "real_dataset_2018"
# read_dataset(dir, filename)
# print("真实场景2018年泄漏数据加载完成")
#
# filename = "real_no_dataset_2018"
# read_dataset(dir, filename)
# print("真实场景2018年无泄漏数据加载完成")
#
# _, _, _, real_pressure_2018 = get_dataset(f"{File_PATH}/real_no_dataset_2018.pickle",
#                                           f"{File_PATH}/real_dataset_2018.pickle")
# with open(os.path.join(File_PATH, "real_pressure_2018.csv"), "w", newline="") as f:
#     real_pressure_2018.to_csv(f, index=True)
# with open(os.path.join(File_PATH, "real_pressure_2018.pickle"), "wb") as f:
#     pickle.dump(real_pressure_2018, f)



# filename = "real_dataset_2019"
# read_dataset(dir, filename)
# print("真实场景2019年泄漏数据加载完成")
#
# filename = "real_no_dataset_2019"
# read_dataset(dir, filename)
# print("真实场景2019年无泄漏数据加载完成")
#
# _, _, _, real_pressure_2019 = get_dataset(f"{File_PATH}/real_no_dataset_2019.pickle",
#                                           f"{File_PATH}/real_dataset_2019.pickle")
#
# with open(os.path.join(File_PATH, "real_pressure_2019.csv"), "w", newline="") as f:
#     real_pressure_2019.to_csv(f, index=True)
# with open(os.path.join(File_PATH, "real_pressure_2019.pickle"), "wb") as f:
#     pickle.dump(real_pressure_2019, f)

#
# filename = "real_dataset_2018"
# read_dataset(dir, filename)
# print("真实场景2018年泄漏数据加载完成")
#
# filename = "real_no_dataset_2018"
# read_dataset(dir, filename)
# print("真实场景2018年无泄漏数据加载完成")
#
# _, _, _, real_pressure_2018 = get_dataset(f"{File_PATH}/real_no_dataset_2018.pickle",
#                                           f"{File_PATH}/real_dataset_2018.pickle")
#
# with open(os.path.join(File_PATH, "real_pressure_2018.pickle"), "wb") as f:
#     pickle.dump(real_pressure_2018, f)


# 2018 和 2019 数据 合并
fix_pressure_2018 = read_dataset(File_PATH, "real_fix_pressure_2018")
fix_pressure_2019 = read_dataset(File_PATH, "real_fix_pressure_2019")
fix_pressure = pd.concat([fix_pressure_2018, fix_pressure_2019], axis=0)
with open(os.path.join(File_PATH, "real_fix_pressure.csv"), "w", newline="") as f:
    fix_pressure.to_csv(f, index=True)
with open(os.path.join(File_PATH, "real_fix_pressure.pickle"), "wb") as f:
    pickle.dump(fix_pressure, f)
print(fix_pressure.shape)

# 2018 和 2019 数据 合并
pressure_2018 = read_dataset(File_PATH, "real_pressure_2018")
pressure_2019 = read_dataset(File_PATH, "real_pressure_2019")
pressure = pd.concat([pressure_2018, pressure_2019], axis=0)
with open(os.path.join(File_PATH, "real_pressure.csv"), "w", newline="") as f:
    pressure.to_csv(f, index=True)
with open(os.path.join(File_PATH, "real_pressure.pickle"), "wb") as f:
    pickle.dump(pressure, f)
print(pressure.shape)