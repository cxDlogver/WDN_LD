from Experiment.detection import *

with open(f"{File_PATH}/PUMP_AB_model.pickle", "rb") as f:
    models_DMA_AB = pickle.load(f)

models = models_DMA_AB

pump_flow_df = pd.read_csv("./pumps_diff_Q.csv")
pressure_df = pd.read_csv("./sensors_diff_P.csv")
# 第一列重命名为时间
pump_flow_df.rename(columns={pump_flow_df.columns[0]: "Timestamp"}, inplace=True)
pressure_df.rename(columns={pressure_df.columns[0]: "Timestamp"}, inplace=True)
# 时间列转换为时间类型
pump_flow_df["Timestamp"] = pd.to_datetime(pump_flow_df["Timestamp"])
pressure_df["Timestamp"] = pd.to_datetime(pressure_df["Timestamp"])
pump_flow_df.set_index("Timestamp", inplace=True)
pressure_df.set_index("Timestamp", inplace=True)

pump_flow = pump_flow_df["PUMP_1"]
pressure = pressure_df[select_sensors]


fix_pressure = pd.DataFrame(index=pressure.index, columns=pressure.columns)
pre_error = pd.DataFrame(0, index=pressure.index, columns=pressure.columns)

# 模型逐传感器预测
for sensor, model in models.items():
    pre = predict_pump_model(model, pressure[sensor], pump_flow, q=48)
    pre_error.loc[pre_error.index[-len(pre):], sensor] = pre
    fix_pressure[sensor] = pressure[sensor] - pre_error[sensor]
    print(f"{sensor} 修正完成")

select_sensors = DMA_AB_sensors if DMA == 'AB' else DMA_C_sensors
