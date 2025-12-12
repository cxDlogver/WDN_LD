import pandas as pd

from Experiment.detection import *

def read_multiple_leak_pressures():
    if os.path.exists(f"{File_PATH}/pipe_leaks_pressure.pickle"):
        with open(f"{File_PATH}/pipe_leaks_pressure.pickle", "rb") as f:
            pipe_leaks_pressure = pickle.load(f)
        with open(f"{File_PATH}/pipe_leaks_demands.pickle", "rb") as f:
            pipe_leaks_demands = pickle.load(f)
        return pipe_leaks_pressure, pipe_leaks_demands

    DATA_DIR = f"{DATA_PATH}/Single Leak Dataset Leak/"
    no_leak_dataset_file = f"{File_PATH}/no_leak_dataset_2018.pickle"
    with open(no_leak_dataset_file, "rb") as f:
        no_leak_dataset = pickle.load(f)
    no_leak_pressure = no_leak_dataset['Pressures (m)']

    pipe_leak_pressures = {
        'abrupt': {},
        'incipient': {},
    }
    pipe_leak_demands = {
        'abrupt': {},
        'incipient': {},
    }
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".pickle"):
                filename = os.path.join(root, file)
                path = Path(filename)
                pipe_name = path.parent.name.split("_")[0]
                if path.parent.name.endswith('abrupt'):
                    leak_type = 'abrupt'
                else:
                    leak_type = 'incipient'
                with open(filename, 'rb') as f:
                    if pipe_name not in pipe_leak_pressures[leak_type]:
                        pipe_leak_pressures[leak_type][pipe_name] = {}
                    if pipe_name not in pipe_leak_demands[leak_type]:
                        pipe_leak_demands[leak_type][pipe_name] = {}
                    leak_dataset = pickle.load(f)
                    for item in leak_dataset:
                        leak_dataset[item].set_index("Timestamp", inplace=True)
                        leak_dataset[item].index = pd.to_datetime(leak_dataset[item].index)
                    pressure = (no_leak_pressure - leak_dataset['Pressures (m)']).dropna()
                    demand = leak_dataset['LeakDemands (L_h)']
                    pipe_leak_pressures[leak_type][pipe_name] = pressure
                    pipe_leak_demands[leak_type][pipe_name] = demand

    with open(f"{File_PATH}/pipe_leak_pressures.pickle", "wb") as f:
        pickle.dump(pipe_leak_pressures, f)
    with open(f"{File_PATH}/pipe_leak_demands.pickle", "wb") as f:
        pickle.dump(pipe_leak_demands, f)
    return pipe_leak_pressures, pipe_leak_demands

pipe_leak_pressures, pipe_leak_demands = read_multiple_leak_pressures()
incipient_pressure = pipe_leak_pressures['incipient']   # dict: pipe_name -> DataFrame(T×S)

# 2. 构造敏感度矩阵: 行=传感器, 列=管道
all_sensors = DMA_AB_sensors + DMA_C_sensors
all_pipes   = list(incipient_pressure.keys())

Sensitivity_Matrix = pd.DataFrame(0.0, index=all_sensors, columns=all_pipes)

for pipe_name, df in incipient_pressure.items():
    # df: index=Timestamp, columns=传感器
    mean_delta = df.mean(axis=0)   # Series(index=传感器名)
    Sensitivity_Matrix[pipe_name] = mean_delta.reindex(all_sensors).fillna(0.0)

# 3. 整体归一化
Sensitivity_abs = Sensitivity_Matrix.abs()

M_min = Sensitivity_abs.min().min()
M_max = Sensitivity_abs.max().max()

Sensitivity_norm = (Sensitivity_abs - M_min) / (M_max - M_min)
M = Sensitivity_norm
with open(f"{File_PATH}/Sensitivity_Matrix.pickle", "wb") as f:
    pickle.dump(M, f)

sensor_max = pd.Series(M.idxmax(axis=1), index=M.index)
with open(f"{File_PATH}/sensor_max.pickle", "wb") as f:
    pickle.dump(sensor_max, f)

