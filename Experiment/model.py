import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import itertools
from typing import Tuple, Iterable, Optional, Union, Dict
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wntr
# 允许中文绘图
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import itertools
from collections import defaultdict
import os
from copy import deepcopy
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import wntr
import yaml
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

def draw_plot(data, title=""):
    """
    绘制时间序列数据曲线
    :param data: pandas Series 或 DataFrame，表示待绘制的数据
    :param title: 图标题
    """
    data.plot(figsize=(12, 6), title=title)
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def read_dataset(dir="", filename=None):
    """
    读取数据集
    :param dir: 数据集目录
    :param filename: 数据集文件名
    :return: pandas DataFrame，表示读取的数据集
    """
    pickle_file = f"{File_PATH}/{filename}.pickle"
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = pd.read_excel(f"{dir}/{filename}/Measurements.xlsx", sheet_name=None)
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f)
    return dataset

""" 处理数据集 """
def get_dataset(no_leak_dataset_file, leak_dataset_file):
    """
    获取数据集
    :return: 无泄漏数据集, 泄漏数据集, 差值数据集
    """
    with open(no_leak_dataset_file, "rb") as f:
        no_leak_dataset = pickle.load(f)
    with open(leak_dataset_file, "rb") as f:
        leak_dataset = pickle.load(f)
    train_dataset = {}
    for index in no_leak_dataset.keys():
        train_dataset[index] = no_leak_dataset[index].iloc[::3, :] - leak_dataset[index]
        train_dataset[index] = train_dataset[index].dropna()
    pump_flow = train_dataset['Flows (m3_h)'].loc[:, 'PUMP_1']
    pressure = train_dataset['Pressures (m)']
    return no_leak_dataset, leak_dataset, pump_flow, pressure


def train_pump_model(pressure, pump_flow, y=None, w=0, q=48,
                     size=0.9, n_estimators=200, random_state=42):
    """
    训练随机森林回归模型，用于预测泵流量-压力关系

    :param pressure: Series，压力数据
    :param pump_flow: Series，泵流量数据
    :param y: 可选，真实目标值（默认使用 pressure）
    :param w: 压力历史窗口长度
    :param q: 流量历史窗口长度
    :param size: 训练集占比，若=1则不划分测试集
    :param n_estimators: 随机森林树数量
    :param random_state: 随机种子
    :return: 训练好的模型
    """
    df = pd.DataFrame({'Y': y if y is not None else pressure})

    # 添加历史特征
    for i in range(q):
        df[f"pump_flow_{i}"] = pump_flow.shift(i)
    for j in range(w):
        df[f"pressure_{j}"] = pressure.shift(j)

    # 特征拼接
    feature_cols = [f"pump_flow_{i}" for i in range(q)] + [f"pressure_{j}" for j in range(w)]
    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols].values
    Y = df["Y"].values

    # 数据划分
    if size < 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=size, shuffle=False)
    else:
        X_train, Y_train = X, Y
        X_test, Y_test = None, None

    # 模型训练
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, Y_train)

    # 若有测试集，输出评估指标
    if size < 1:
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return model

def predict_pump_model(model, pressure, pump_flow, y=None, w=0, q=48):
    """
    使用训练好的模型预测泵流量修正项
    """
    df = pd.DataFrame({'Y': y if y is not None else pressure})

    # 构造历史特征
    for i in range(q):
        df[f"pump_flow_{i}"] = pump_flow.shift(i)
    for j in range(w):
        df[f"pressure_{j}"] = pressure.shift(j)

    feature_cols = [f"pump_flow_{i}" for i in range(q)] + [f"pressure_{j}" for j in range(w)]
    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols].values
    Y_pred = model.predict(X)
    return Y_pred

# 模型效果评估与压力修正
def evaluate_pump_model(models, leak_dataset=None, no_leak_dataset=None, filename=None, year=2018):
    """
    对泄漏数据进行模型评估并输出修正后的压力差
    :param models: dict[str, RandomForestRegressor] 各传感器模型
    :param filename: 泄漏数据 pickle 文件名
    :param dir: 数据目录
    :return: 修正压力、预测误差、原压力、泵流量
    """
    if leak_dataset is None:
        if filename:
            with open(f"{File_PATH}/{filename}.pickle", 'rb') as f:
                leak_dataset = pickle.load(f)
        else:
            with open(f"{File_PATH}/leak_dataset_{year}.pickle", 'rb') as f:
                leak_dataset = pickle.load(f)
    if no_leak_dataset is None:
        with open(f"{File_PATH}/no_leak_dataset_{year}.pickle", 'rb') as f:
            no_leak_dataset = pickle.load(f)

    train_dataset = {}
    # 计算无泄漏与泄漏数据差值
    for key in no_leak_dataset.keys():
        if 'Timestamp' in no_leak_dataset[key].columns:
            no_leak_dataset[key].set_index('Timestamp', inplace=True)
        if 'Timestamp' in leak_dataset[key].columns:
            leak_dataset[key].set_index('Timestamp', inplace=True)
        no_leak_dataset[key].index = pd.to_datetime(no_leak_dataset[key].index)
        leak_dataset[key].index = pd.to_datetime(leak_dataset[key].index)
        train_dataset[key] = no_leak_dataset[key].iloc[::3, :] - leak_dataset[key].iloc[:, :]
        train_dataset[key] = train_dataset[key].dropna()

    pump_flow = train_dataset['Flows (m3_h)']['PUMP_1']
    pressure = train_dataset['Pressures (m)']
    fix_pressure = pd.DataFrame(index=pressure.index, columns=pressure.columns)
    pre_error = pd.DataFrame(0, index=pressure.index, columns=pressure.columns)
    # 模型逐传感器预测
    for sensor, model in models.items():
        pre = predict_pump_model(model, pressure[sensor], pump_flow, q=48)
        pre_error.loc[pre_error.index[-len(pre):], sensor] = pre
        fix_pressure[sensor] = pressure[sensor] - pre_error[sensor]
        # print(f"{sensor} 修正完成")

    return fix_pressure, pre_error, pressure, pump_flow


# DMA AB
DMA_AB_sensors = ['n54', 'n105', 'n114', 'n163', 'n188', 'n288', 'n296', 'n332', 'n342', 'n410', 'n415', 'n429',
                  'n458', 'n469', 'n495', 'n506', 'n516', 'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
                  'n726', 'n740', 'n752', 'n769', 'n229', 'n215']
# DMA_C
DMA_C_sensors = ['n1', 'n4', 'n31']

# 数据文件路径
File_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "file")
# 数据集路径
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Dataset Generator/"))
# 结果保持路径
RESULT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "result")
# 图片保持路径
FIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "fig")
# 框架图片路径
FRAME_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "frame_fig")



leak_2019_data = [
    ["p257", "2019-01-01 00:00", "2019-12-31 23:55"],
    ["p427", "2019-01-01 00:00", "2019-12-31 23:55"],
    ["p810", "2019-01-01 00:00", "2019-12-31 23:55"],
    ["p654", "2019-01-01 00:00", "2019-12-31 23:55"],
    ["p523", "2019-01-15 23:00", "2019-02-01 09:50"],
    ["p827", "2019-01-24 18:30", "2019-02-07 09:05"],
    ["p280", "2019-02-10 13:05", "2019-12-31 23:55"],
    ["p653", "2019-03-03 13:10", "2019-05-05 12:10"],
    ["p710", "2019-03-24 14:15", "2019-12-31 23:55"],
    ["p514", "2019-04-02 20:40", "2019-05-23 14:55"],
    ["p331", "2019-04-20 10:10", "2019-12-31 23:55"],
    ["p193", "2019-05-19 10:40", "2019-12-31 23:55"],
    ["p277", "2019-05-30 21:55", "2019-12-31 23:55"],
    ["p142", "2019-06-12 19:55", "2019-07-17 09:25"],
    ["p680", "2019-07-10 08:45", "2019-12-31 23:55"],
    ["p586", "2019-07-26 14:40", "2019-09-16 03:20"],
    ["p721", "2019-08-02 03:00", "2019-12-31 23:55"],
    ["p800", "2019-08-16 14:00", "2019-10-01 16:35"],
    ["p123", "2019-09-13 20:05", "2019-12-31 23:55"],
    ["p455", "2019-10-03 14:00", "2019-12-31 23:55"],
    ["p762", "2019-10-09 10:15", "2019-12-31 23:55"],
    ["p426", "2019-10-25 13:25", "2019-12-31 23:55"],
    ["p879", "2019-11-20 11:55", "2019-12-31 23:55"],
]

leak_2018_data = [
    ["p257", "2018-01-08 13:30", "2018-12-31 23:55"],
    ["p461", "2018-01-23 04:25", "2018-04-02 11:40"],
    ["p232", "2018-01-31 02:35", "2018-02-10 09:20"],
    ["p427", "2018-02-13 08:25", "2018-12-31 23:55"],
    ["p673", "2018-03-05 15:45", "2018-03-23 10:25"],
    ["p810", "2018-07-28 03:05", "2018-12-31 23:55"],
    ["p628", "2018-05-02 14:55", "2018-05-29 21:20"],
    ["p538", "2018-05-18 08:35", "2018-06-02 06:05"],
    ["p866", "2018-06-01 09:05", "2018-06-12 03:00"],
    ["p31", "2018-06-28 10:35", "2018-08-12 17:30"],
    ["p654", "2018-07-05 03:40", "2018-12-31 23:55"],
    ["p183", "2018-08-07 02:35", "2018-09-01 17:10"],
    ["p158", "2018-10-06 02:35", "2018-10-23 13:35"],
    ["p369", "2018-10-26 02:05", "2018-11-08 20:25"],
]

leak_data = [
    ["p257", "2018-01-08 13:30", "2019-12-31 23:55"],
    ["p461", "2018-01-23 04:25", "2018-04-02 11:40"],
    ["p232", "2018-01-31 02:35", "2018-02-10 09:20"],
    ["p427", "2018-02-13 08:25", "2019-12-31 23:55"],
    ["p673", "2018-03-05 15:45", "2018-03-23 10:25"],
    ["p810", "2018-07-28 03:05", "2019-12-31 23:55"],
    ["p628", "2018-05-02 14:55", "2018-05-29 21:20"],
    ["p538", "2018-05-18 08:35", "2018-06-02 06:05"],
    ["p866", "2018-06-01 09:05", "2018-06-12 03:00"],
    ["p31", "2018-06-28 10:35", "2018-08-12 17:30"],
    ["p654", "2018-07-05 03:40", "2019-12-31 23:55"],
    ["p183", "2018-08-07 02:35", "2018-09-01 17:10"],
    ["p158", "2018-10-06 02:35", "2018-10-23 13:35"],
    ["p369", "2018-10-26 02:05", "2018-11-08 20:25"],
    ["p523", "2019-01-15 23:00", "2019-02-01 09:50"],
    ["p827", "2019-01-24 18:30", "2019-02-07 09:05"],
    ["p280", "2019-02-10 13:05", "2019-12-31 23:55"],
    ["p653", "2019-03-03 13:10", "2019-05-05 12:10"],
    ["p710", "2019-03-24 14:15", "2019-12-31 23:55"],
    ["p514", "2019-04-02 20:40", "2019-05-23 14:55"],
    ["p331", "2019-04-20 10:10", "2019-12-31 23:55"],
    ["p193", "2019-05-19 10:40", "2019-12-31 23:55"],
    ["p277", "2019-05-30 21:55", "2019-12-31 23:55"],
    ["p142", "2019-06-12 19:55", "2019-07-17 09:25"],
    ["p680", "2019-07-10 08:45", "2019-12-31 23:55"],
    ["p586", "2019-07-26 14:40", "2019-09-16 03:20"],
    ["p721", "2019-08-02 03:00", "2019-12-31 23:55"],
    ["p800", "2019-08-16 14:00", "2019-10-01 16:35"],
    ["p123", "2019-09-13 20:05", "2019-12-31 23:55"],
    ["p455", "2019-10-03 14:00", "2019-12-31 23:55"],
    ["p762", "2019-10-09 10:15", "2019-12-31 23:55"],
    ["p426", "2019-10-25 13:25", "2019-12-31 23:55"],
    ["p879", "2019-11-20 11:55", "2019-12-31 23:55"],
]

leak_data_AB = [
    ["p461", "2018-01-23 04:25", "2018-04-02 11:40"],
    ["p232", "2018-01-31 02:35", "2018-02-10 09:20"],
    ["p427", "2018-02-13 08:25", "2019-12-31 23:55"],
    ["p673", "2018-03-05 15:45", "2018-03-23 10:25"],
    ["p810", "2018-07-28 03:05", "2019-12-31 23:55"],
    ["p628", "2018-05-02 14:55", "2018-05-29 21:20"],
    ["p538", "2018-05-18 08:35", "2018-06-02 06:05"],
    ["p866", "2018-06-01 09:05", "2018-06-12 03:00"],
    ["p654", "2018-07-05 03:40", "2019-12-31 23:55"],
    ["p183", "2018-08-07 02:35", "2018-09-01 17:10"],
    ["p158", "2018-10-06 02:35", "2018-10-23 13:35"],
    ["p369", "2018-10-26 02:05", "2018-11-08 20:25"],
    ["p523", "2019-01-15 23:00", "2019-02-01 09:50"],
    ["p827", "2019-01-24 18:30", "2019-02-07 09:05"],
    ["p653", "2019-03-03 13:10", "2019-05-05 12:10"],
    ["p710", "2019-03-24 14:15", "2019-12-31 23:55"],
    ["p514", "2019-04-02 20:40", "2019-05-23 14:55"],
    ["p331", "2019-04-20 10:10", "2019-12-31 23:55"],
    ["p193", "2019-05-19 10:40", "2019-12-31 23:55"],
    ["p142", "2019-06-12 19:55", "2019-07-17 09:25"],
    ["p680", "2019-07-10 08:45", "2019-12-31 23:55"],
    ["p586", "2019-07-26 14:40", "2019-09-16 03:20"],
    ["p721", "2019-08-02 03:00", "2019-12-31 23:55"],
    ["p800", "2019-08-16 14:00", "2019-10-01 16:35"],
    ["p123", "2019-09-13 20:05", "2019-12-31 23:55"],
    ["p455", "2019-10-03 14:00", "2019-12-31 23:55"],
    ["p762", "2019-10-09 10:15", "2019-12-31 23:55"],
    ["p426", "2019-10-25 13:25", "2019-12-31 23:55"],
    ["p879", "2019-11-20 11:55", "2019-12-31 23:55"],
]

leak_data_C = [
    ["p257", "2018-01-08 13:30", "2019-12-31 23:55"],
    ["p31", "2018-06-28 10:35", "2018-08-12 17:30"],
    ["p280", "2019-02-10 13:05", "2019-12-31 23:55"],
    ["p277", "2019-05-30 21:55", "2019-12-31 23:55"],
]

plt.rcParams.update({
    'font.family':'serif',
    'font.serif': ['Times New Roman','SimSun', 'SimHei'],
    'font.weight': 'normal',
    'font.size': 18,                       # 正文默认字体大小（IEEE建议图表标签8-10磅，此处取10）
    'axes.labelsize': 16,                  # 坐标轴标签字体大小（如"Magnetization (A/m)"）
    'axes.titlesize': 16,                  # 子图标题字体大小（如有）
    'xtick.labelsize': 16,                  # x轴刻度标签字体大小（IEEE建议8-10磅）
    'ytick.labelsize': 16,                  # y轴刻度标签字体大小
    'legend.fontsize': 12,                  # 图例字体大小（简洁清晰）
    'axes.unicode_minus': False
})
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
hexs = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]