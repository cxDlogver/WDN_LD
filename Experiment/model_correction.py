""" 拟合传感器和水泵流量之间的函数关系，进行压力校正 """
from model import *


# w = 0  # 不考虑历史压力
q = 48
isTrainAB = False
isTrainC = False
if os.path.exists(f"{File_PATH}/PUMP_AB_model.pickle") and isTrainAB is False:
    with open(f"{File_PATH}/PUMP_AB_model.pickle", "rb") as f:
        models_DMA_AB = pickle.load(f)
else:
    no_leak_dataset, leak_dataset, pump_flow, pressure = get_dataset(f"{File_PATH}/no_leak_dataset_2018.pickle", f"{File_PATH}/PUMP_DMA_AB.pickle")
    models_DMA_AB = { sensor: train_pump_model(pressure[sensor], pump_flow, q=q) for sensor in DMA_AB_sensors }
    with open(f"{File_PATH}/PUMP_AB_model.pickle", "wb") as f:
        pickle.dump(models_DMA_AB, f)

if os.path.exists(f"{File_PATH}/PUMP_C_model.pickle") and isTrainC is False:
    with open(f"{File_PATH}/PUMP_C_model.pickle", "rb") as f:
        models_DMA_C = pickle.load(f)
else:
    no_leak_dataset, leak_dataset, pump_flow, pressure = get_dataset(f"{File_PATH}/no_leak_dataset_2018.pickle", f"{File_PATH}/PUMP_DMA_C.pickle")
    models_DMA_C = { sensor: train_pump_model(pressure[sensor], pump_flow, q=q) for sensor in DMA_C_sensors }
    with open(f"{File_PATH}/PUMP_C_model.pickle", "wb") as f:
        pickle.dump(models_DMA_C, f)


# 模型校正
# models = {**models_DMA_AB, **models_DMA_C}
# print("2018年压力校正")
# fix_pressure_2018, pre_error_2018, leak_pressure_2018, pump_flow_2018 = evaluate_pump_model(models,year=2018)
# print("2019年压力校正")
# fix_pressure_2019, pre_error_2019, leak_pressure_2019, pump_flow_2019 = evaluate_pump_model(models,year=2019)
# with open("./file/fix_pressure_2018.pickle", "wb") as f:
#     pickle.dump(fix_pressure_2018, f)
# with open("./file/fix_pressure_2019.pickle", "wb") as f:
#     pickle.dump(fix_pressure_2019, f)
# with open("./file/pressure_2018.pickle", "wb") as f:
#     pickle.dump(leak_pressure_2018, f)
# with open("./file/pressure_2019.pickle", "wb") as f:
#     pickle.dump(leak_pressure_2019, f)

# 真实场景
# real_dataset_2019 = read_dataset(filename="real_dataset_2019")
# real_no_dataset_2019 = read_dataset(filename="real_no_dataset_2019")
# fix_pressure_2019, pre_error_2019, leak_pressure_2019, pump_flow_2019 = evaluate_pump_model(models,real_dataset_2019,real_no_dataset_2019)
# with open("./file/real_fix_pressure_2019.pickle", "wb") as f:
#     pickle.dump(fix_pressure_2019, f)
# with open("./file/real_pressure_2019.pickle", "wb") as f:
#     pickle.dump(leak_pressure_2019, f)
# real_dataset_2018 = read_dataset(filename="real_dataset_2018")
# real_no_dataset_2018 = read_dataset(filename="real_no_dataset_2018")
# fix_pressure_2018, pre_error_2018, leak_pressure_2018, pump_flow_2018 = evaluate_pump_model(models,real_dataset_2018,real_no_dataset_2018)
# with open("./file/real_fix_pressure_2018.pickle", "wb") as f:
#     pickle.dump(fix_pressure_2018, f)
# with open("./file/real_pressure_2018.pickle", "wb") as f:
#     pickle.dump(leak_pressure_2018, f)

#===================================================
#展示
# with open("./file/fix_pressure_2018.pickle", "rb") as f:
#     fix_pressure_2018 = pickle.load(f)
# with open("./file/fix_pressure_2019.pickle", "rb") as f:
#     fix_pressure_2019 = pickle.load(f)
# with open("./file/pressure_2018.pickle", "rb") as f:
#     leak_pressure_2018 = pickle.load(f)
# with open("./file/pressure_2019.pickle", "rb") as f:
#     leak_pressure_2019 = pickle.load(f)
#
# draw_plot(leak_pressure_2018,'2018 pressure_diff no correction')
# draw_plot(fix_pressure_2018,'2018 pressure_diff corrected')
# draw_plot(leak_pressure_2019,'2019 pressure_diff no correction')
# draw_plot(fix_pressure_2019,'2019 pressure_diff corrected')
