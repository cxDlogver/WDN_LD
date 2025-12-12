import pandas as pd

from Experiment.detection import *


def run_experiment(
        DMA,
        fix_pressure,
        inp_file,
        leak_data,
        leak_identification_file = "leak_identification",
        leak_location_file = "leak_location"
):
    """运行一次实验，返回检测结果"""

    """ 初始化数据 """
    # 区域传感器
    select_sensors = DMA_AB_sensors if DMA == 'AB' else DMA_C_sensors
    # FIS 参数
    fis_params = init_fis_params(DMA)
    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_data = leak_data
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 模糊推理泄漏识别
    fis_result, day_result_step, pressure_result = run(fix_pressure, fis_params, sensors=select_sensors)
    leak_flag_times, detection_flag_times = collect_leak_detection_times(fis_result)
    pressure_detection = pressure_result.loc[list(detection_flag_times['time']), select_sensors].astype(
        float)
    # - fis_result: 基于FIS的泄漏识别结果
    # - pressure_result: 修正后的每天传感器压力数据
    # - pressure_detection: 基于FIS的泄漏识别结果中检测到泄漏的传感器压力数据
    # - leak_flag_times: 泄漏事件的起始事件记录
    # - detection_flag_times: 泄漏事件的检测事件和最终起始时间预测
    with open(os.path.join(RESULT_PATH, f"{leak_identification_file}_{DMA}.pickle"), "wb") as f:
        pickle.dump((fis_result, pressure_result, pressure_detection, detection_flag_times), f)
        print(f"区域{DMA} FIS 泄漏识别结果已保存至 {RESULT_PATH}/{leak_identification_file}_{DMA}.pickle")

    # 计算泄漏重心
    center_df = pd.DataFrame(None, index=pressure_detection.index, columns=['x_center', 'y_center', 'pipes'])
    pipe_df = pd.DataFrame(columns=pressure_detection.index)
    for i in pressure_detection.index:
        pi, _, _ = compute_leak_probability(pressure_detection.loc[i])
        pipe_df[i] = pi
        center, S = compute_leakage_center_single(pi, pipe_coords)
        if center is None:
            # 删除该索引行
            center_df.drop(index=i, inplace=True)
            continue
        center_df.loc[i, 'x_center'] = center[0]
        center_df.loc[i, 'y_center'] = center[1]
        center_df.loc[i, 'pipes'] = S
    # 1) 由中心 → 评价表 leakage_center
    leakage_center = get_leakage_center(
        center_df=center_df,
        pipe_coords=pipe_coords,
        leak_info=leak_info,
        distance_thresh=300.0
    )
    # 2) 由 leakage_center + detection_flag_times → 纠错表
    detection_correction, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center,
        leak_info=leak_info
    )
    # 3) 计算纠错表格的经济值
    economic_value = calculate_economic_value(
        detection_correction=detection_correction,
        leakage_center=leakage_center,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    with open(os.path.join(RESULT_PATH, f"{leak_location_file}_{DMA}.pickle"), "wb") as f:
        pickle.dump((pipe_df, center_df, leakage_center, detection_correction, economic_value), f)
        print(f"区域{DMA} FIS 泄漏位置结果已保存至 {RESULT_PATH}/{leak_location_file}_{DMA}.pickle")

def RFLOMA():
    with open(os.path.join(File_PATH, "fix_pressure.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Model.inp")

    DMA = 'AB'
    run_experiment(DMA, fix_pressure, inp_file, leak_data)
    DMA = 'C'
    run_experiment(DMA, fix_pressure, inp_file, leak_data)

    with open(os.path.join(RESULT_PATH, f"leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC FIS 泄漏检测结果已保存至 {RESULT_PATH}/leak_detection_ABC.pickle")

# ===========================================
# 无水泵流量-传感器压力模型修正
# ===========================================
def NO_Correction():
    with open(os.path.join(File_PATH, "pressure.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Model.inp")

    DMA = 'AB'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_cor_leak_identification",
                   leak_location_file="no_cor_leak_location")
    DMA = 'C'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_cor_leak_identification",
                   leak_location_file="no_cor_leak_location")

    with open(os.path.join(RESULT_PATH, f"no_cor_leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_cor_leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_cor_leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_cor_leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,

    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,

    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "no_cor_leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC FIS 泄漏检测结果已保存至 {RESULT_PATH}/no_cor_leak_detection_ABC.pickle")

# ===========================================
# 无历史泄漏剔除
# ===========================================
def run_no_re_experiment(
        DMA,
        fix_pressure,
        inp_file,
        leak_data,
        leak_identification_file="leak_identification",
        leak_location_file="leak_location"
):
        """运行一次实验，返回检测结果"""

        """ 初始化数据 """
        # 区域传感器
        select_sensors = DMA_AB_sensors if DMA == 'AB' else DMA_C_sensors
        # FIS 参数
        fis_params = init_fis_params(DMA)
        # 泄漏点和需求量
        leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
        # 水网络模型
        wn = wntr.network.WaterNetworkModel(inp_file)
        # 管道坐标
        pipe_coords = get_coordinate(wn)
        # 泄漏事件
        leak_data = leak_data
        leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
        # 合并坐标
        leak_info = leak_events.join(pipe_coords, on="linkID")
        # 转换时间格式
        leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
        leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

        # 模糊推理泄漏识别
        fis_result, day_result_step, pressure_result = run(fix_pressure, fis_params, sensors=select_sensors)
        leak_flag_times, detection_flag_times = collect_leak_detection_times(fis_result)
        pressure_detection = fix_pressure.resample("D").mean().loc[list(detection_flag_times['time']), select_sensors].astype(
            float)
        # - fis_result: 基于FIS的泄漏识别结果
        # - pressure_result: 修正后的每天传感器压力数据
        # - pressure_detection: 基于FIS的泄漏识别结果中检测到泄漏的传感器压力数据
        # - leak_flag_times: 泄漏事件的起始事件记录
        # - detection_flag_times: 泄漏事件的检测事件和最终起始时间预测
        with open(os.path.join(RESULT_PATH, f"{leak_identification_file}_{DMA}.pickle"), "wb") as f:
            pickle.dump((fis_result, pressure_result, pressure_detection, detection_flag_times), f)
            print(f"区域{DMA} FIS 泄漏识别结果已保存至 {RESULT_PATH}/{leak_identification_file}_{DMA}.pickle")

        # 计算泄漏重心
        center_df = pd.DataFrame(None, index=pressure_detection.index, columns=['x_center', 'y_center', 'pipes'])
        pipe_df = pd.DataFrame(columns=pressure_detection.index)
        for i in pressure_detection.index:
            pi, _, _ = compute_leak_probability(pressure_detection.loc[i])
            pipe_df[i] = pi
            center, S = compute_leakage_center_single(pi, pipe_coords)
            if center is None:
                # 删除该索引行
                center_df.drop(index=i, inplace=True)
                continue
            center_df.loc[i, 'x_center'] = center[0]
            center_df.loc[i, 'y_center'] = center[1]
            center_df.loc[i, 'pipes'] = S
        # 1) 由中心 → 评价表 leakage_center
        leakage_center = get_leakage_center(
            center_df=center_df,
            pipe_coords=pipe_coords,
            leak_info=leak_info,
            distance_thresh=300.0
        )
        # 2) 由 leakage_center + detection_flag_times → 纠错表
        detection_correction, leak_info_remaining = generate_detection_correction(
            leakage_center=leakage_center,
            leak_info=leak_info,
        )
        # 3) 计算纠错表格的经济值
        economic_value = calculate_economic_value(
            detection_correction=detection_correction,
            leakage_center=leakage_center,
            leak_info=leak_info,
            leak_demands=leak_demands
        )

        with open(os.path.join(RESULT_PATH, f"{leak_location_file}_{DMA}.pickle"), "wb") as f:
            pickle.dump((pipe_df, center_df, leakage_center, detection_correction, economic_value), f)
            print(f"区域{DMA} FIS 泄漏位置结果已保存至 {RESULT_PATH}/{leak_location_file}_{DMA}.pickle")

def NO_Removal():
    with open(os.path.join(File_PATH, "fix_pressure.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Model.inp")



    DMA = 'AB'
    run_no_re_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_re_leak_identification",
                   leak_location_file="no_re_leak_location")
    DMA = 'C'
    run_no_re_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_re_leak_identification",
                   leak_location_file="no_re_leak_location")

    with open(os.path.join(RESULT_PATH, f"no_re_leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_re_leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_re_leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_re_leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "no_re_leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC FIS 泄漏检测结果已保存至 {RESULT_PATH}/no_re_leak_detection_ABC.pickle")

# ===========================================
# 无FIS
# ===========================================
def NO_FIS():
    with open(os.path.join(File_PATH, "fix_pressure.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Model.inp")

    def run_no_FIS(fix_pressure, fis_param, sensors=None):
        day_pressure = fix_pressure.resample('D').mean()
        fix_pressure_2 = fix_pressure.resample('D').mean()
        # print(fix_pressure_2)
        pressure_result = pd.DataFrame(columns=fix_pressure_2.columns)
        fis_result = {col: pd.DataFrame() for col in fix_pressure_2.columns}

        if sensors is None:
            print("未指定传感器. 请指定 DMA_AB_sensors 或 DMA_C_sensors.")
            return None

        leak_thresh = fis_param['leak_thresh']

        def init_state_store():
            state_store = {}
            first_day = fix_pressure_2.index[0]
            for col in fix_pressure_2.columns:
                state_store[col] = {
                    'leak_base': fix_pressure_2.loc[first_day, col],  # 初始基线 = 第一天的压力
                    'leak_cum': 0.0,
                }
            return state_store

        state_store = init_state_store()
        for current_day in fix_pressure_2.index:
            day_result = fix_pressure_2.loc[current_day]
            day_result = day_result.to_frame().T
            day_result.index = [current_day]

            def mark_model_correction(end_dt):
                for colname in sensors:
                    state_store[colname]['leak_base'] = day_result[colname].values[0]
                    state_store[colname]['leak_cum'] = 0

            def mark_leak(colname, current_day):
                fis_result[colname].at[current_day, 'detection_flag'] = True

            # --- 累积窗口结果 ---
            is_correction = False

            for col in sensors:
                if fis_result[col].empty:
                    for fld in ['leak_cum', 'leak_base', 'detection_flag']:
                        fis_result[col][fld] = np.nan

                s = state_store[col]
                s['leak_cum'] = max(0, s['leak_cum'] + day_result.loc[current_day, col] - s['leak_base'])
                if day_result.loc[current_day, col] < s['leak_base']:
                    s['leak_base'] = day_result.loc[current_day, col]
                # 更新结果持久化
                fis_result[col].at[current_day, 'leak_cum'] = s['leak_cum']
                fis_result[col].at[current_day, 'leak_base'] = s['leak_base']

                # 在渐进状态下 达到 阈值 则直接标记并进行模型校正
                if (s['leak_cum'] >= leak_thresh):
                    mark_leak(col, current_day)
                    is_correction = True

            pressure_result = pd.concat([pressure_result, fix_pressure_2.loc[current_day].to_frame().T])
            if is_correction:
                mark_model_correction(current_day)

        return fis_result, fix_pressure_2, pressure_result

    def run_experiment(
            DMA,
            fix_pressure,
            inp_file,
            leak_data,
            leak_identification_file="leak_identification",
            leak_location_file="leak_location"
    ):
        """运行一次实验，返回检测结果"""

        """ 初始化数据 """
        # 区域传感器
        select_sensors = DMA_AB_sensors if DMA == 'AB' else DMA_C_sensors
        # FIS 参数
        fis_params = init_fis_params(DMA)
        # 泄漏点和需求量
        leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
        # 水网络模型
        wn = wntr.network.WaterNetworkModel(inp_file)
        # 管道坐标
        pipe_coords = get_coordinate(wn)
        # 泄漏事件
        leak_data = leak_data
        leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
        # 合并坐标
        leak_info = leak_events.join(pipe_coords, on="linkID")
        # 转换时间格式
        leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
        leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

        # 模糊推理泄漏识别
        fis_result, day_result_step, pressure_result = run_no_FIS(fix_pressure, fis_params, sensors=select_sensors)
        leak_flag_times, detection_flag_times = collect_leak_detection_times(fis_result)
        pressure_detection = pressure_result.loc[list(detection_flag_times['time']), select_sensors].astype(
            float)
        # - fis_result: 基于FIS的泄漏识别结果
        # - pressure_result: 修正后的每天传感器压力数据
        # - pressure_detection: 基于FIS的泄漏识别结果中检测到泄漏的传感器压力数据
        # - leak_flag_times: 泄漏事件的起始事件记录
        # - detection_flag_times: 泄漏事件的检测事件和最终起始时间预测
        with open(os.path.join(RESULT_PATH, f"{leak_identification_file}_{DMA}.pickle"), "wb") as f:
            pickle.dump((fis_result, pressure_result, pressure_detection, detection_flag_times), f)
            print(f"区域{DMA} FIS 泄漏识别结果已保存至 {RESULT_PATH}/{leak_identification_file}_{DMA}.pickle")

        # 计算泄漏重心
        center_df = pd.DataFrame(None, index=pressure_detection.index, columns=['x_center', 'y_center', 'pipes'])
        pipe_df = pd.DataFrame(columns=pressure_detection.index)
        # 计算泄漏概率
        for i in pressure_detection.index:
            pi, _, _ = compute_leak_probability(pressure_detection.loc[i])
            pipe_df.loc[i] = pi
            center, S = compute_leakage_center_single(pi, pipe_coords)
            if center is None:
                # 删除该索引行
                center_df.drop(index=i, inplace=True)
                continue
            center_df.loc[i, 'x_center'] = center[0]
            center_df.loc[i, 'y_center'] = center[1]
            center_df.loc[i, 'pipes'] = S
        # 1) 由中心 → 评价表 leakage_center
        leakage_center = get_leakage_center(
            center_df=center_df,
            pipe_coords=pipe_coords,
            leak_info=leak_info,
            distance_thresh=300.0
        )
        # 2) 由 leakage_center + detection_flag_times → 纠错表
        detection_correction, leak_info_remaining = generate_detection_correction(
            leakage_center=leakage_center,
            leak_info=leak_info,
        )
        # 3) 计算纠错表格的经济值
        economic_value = calculate_economic_value(
            detection_correction=detection_correction,
            leakage_center=leakage_center,
            leak_info=leak_info,
            leak_demands=leak_demands
        )

        with open(os.path.join(RESULT_PATH, f"{leak_location_file}_{DMA}.pickle"), "wb") as f:
            pickle.dump((pipe_df, center_df, leakage_center, detection_correction, economic_value), f)
            print(f"区域{DMA} FIS 泄漏位置结果已保存至 {RESULT_PATH}/{leak_location_file}_{DMA}.pickle")

    DMA = 'AB'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_fis_leak_identification",
                   leak_location_file="no_fis_leak_location")
    DMA = 'C'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_fis_leak_identification",
                   leak_location_file="no_fis_leak_location")

    with open(os.path.join(RESULT_PATH, f"no_fis_leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_fis_leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_fis_leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_fis_leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "no_fis_leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC FIS 泄漏检测结果已保存至 {RESULT_PATH}/no_fis_leak_detection_ABC.pickle")

# ===========================================
# 状态校正
# ===========================================
def NO_State():
    with open(os.path.join(File_PATH, "fix_pressure.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Model.inp")

    def run_no_state(fix_pressure, fis_param, sensors=None):
        """
        基于FIS的泄漏识别
        :param fix_pressure: 修正后的压力时序数据
        :param fis_param: FIS参数
        :param sensors: 传感器列表
        :return: fis_result, day_result_step, pressure_result
        fis_result: 模糊推理结果，每个传感器的状态结果
        day_result_step: 去除已有泄漏的中间状态
        pressure_result: 修正后的压力时序数据（去除已有泄漏的影响）
        """
        day_pressure = fix_pressure.resample('D').mean()
        fix_pressure_2 = fix_pressure.resample('D').mean()
        # fix_pressure_2.loc[:, :] -= (fix_pressure_2.loc[fix_pressure_2.index[0], :])
        # print(fix_pressure_2)
        pressure_result = pd.DataFrame(columns=fix_pressure_2.columns)
        sensor_features = {col: pd.DataFrame() for col in fix_pressure_2.columns}
        fis_result = {col: pd.DataFrame() for col in fix_pressure_2.columns}

        noise_tolerance_days = 3  # 新增：Noise 容忍天数
        if sensors is None:
            print("未指定传感器. 请指定 DMA_AB_sensors 或 DMA_C_sensors.")
            return None

        leak_thresh = fis_param['leak_thresh']
        fis = LeakageFIS(
            fis_param['th_diff1'],
            fis_param['th_diff4'],
            fis_param['th_diff7'],
        )

        # 初始化状态存储

        def init_state_store():
            state_store = {}
            for col in fix_pressure_2.columns:
                state_store[col] = {
                    'prev_state': None,
                    'in_leak': False,
                    'leak_start': None,
                    'neg_cum': 0.0,
                    'leak_cum': 0.0,
                    'leak_days': 0,
                    'recent_noise_days': 0,  # 新增：记录连续Noise天数
                }
            return state_store

        state_store = init_state_store()
        for current_day in fix_pressure_2.index:
            day_result = fix_pressure_2.loc[current_day]
            day_result = day_result.to_frame().T
            day_result.index = [current_day]

            def mark_model_correction(start_dt, end_dt, flag):
                for colname in sensors:
                    rng = pd.date_range(start=start_dt, end=end_dt, freq='D')
                    for d in rng:
                        if d not in fis_result[colname].index:
                            placeholder = fis_result[colname].iloc[[-1]].copy()
                            placeholder.index = [d]
                            placeholder.loc[d, :] = placeholder.loc[placeholder.index[0], :].where(
                                pd.notna(placeholder.loc[placeholder.index[0], :]), np.nan)
                            fis_result[colname] = pd.concat([fis_result[colname], placeholder])
                        fis_result[colname].at[d, 'model_correction'] = True
                    if flag == 'Leak':
                        fix_pressure_2.loc[:, colname] -= (fix_pressure_2.loc[end_dt, colname])
                    elif flag == 'End':
                        fix_pressure_2.loc[:, colname] -= (fix_pressure_2.loc[end_dt, colname])
                    else:
                        current_vals = fix_pressure_2.loc[end_dt, sensors]
                        v_min = current_vals.min()
                        if v_min < 0:
                            for colname in sensors:
                                fix_pressure_2.loc[:, colname] -= v_min

            def mark_leak(colname, at_dt, current_day):
                if at_dt not in fis_result[colname].index:
                    placeholder = fis_result[colname].iloc[[-1]].copy()
                    placeholder.index = [at_dt]
                    fis_result[colname] = pd.concat([fis_result[colname], placeholder])
                fis_result[colname].at[at_dt, 'leak_flag'] = True
                fis_result[colname].at[current_day, 'detection_flag'] = True

            # --- 累积窗口结果 ---
            is_correction = False
            is_flag = None
            start_correction = fix_pressure_2.index.max()
            start_day = current_day - pd.Timedelta(days=7)
            recent_days = fix_pressure_2.loc[max(start_day, fix_pressure_2.index[0]):current_day + pd.Timedelta(days=1)]

            for col in sensors:
                y_recent = recent_days[col].fillna(method='ffill').fillna(0)
                features = compute_multiscale_features(y_recent)
                time = current_day - pd.Timedelta(days=1) if current_day != day_pressure.index[0] else current_day
                fis_out = fis.evaluate_series(day_pressure.loc[time, col], features[['diff_1', 'diff_4', 'diff_7']])

                if current_day in features.index:
                    today_feature = features.loc[[current_day]]
                    today_fis = fis_out.loc[[current_day]]

                    if sensor_features[col].empty:
                        sensor_features[col] = today_feature.copy()
                    else:
                        sensor_features[col] = pd.concat([sensor_features[col], today_feature])

                    if fis_result[col].empty:
                        fis_result[col] = today_fis.copy()
                        for fld in ['leak_cum', 'neg_cum', 'leak_days', 'leak_flag', 'model_correction',
                                    'detection_flag']:
                            fis_result[col][fld] = np.nan if fld not in ['leak_flag', 'model_correction'] else False
                    else:
                        fis_result[col] = pd.concat(
                            [fis_result[col], today_fis[~today_fis.index.isin(fis_result[col].index)]])
                        for fld in ['leak_cum', 'neg_cum', 'leak_days', 'leak_flag', 'model_correction',
                                    'detection_flag']:
                            if fld not in fis_result[col].columns:
                                fis_result[col][fld] = np.nan if fld not in ['leak_flag', 'model_correction'] else False

                    # === 状态机逻辑更新 ===
                    s = state_store[col]
                    curr = str(today_fis['State'].iloc[0]) if 'State' in today_fis.columns else None
                    s['neg_cum'] = min(0, day_result.loc[current_day, col])
                    if s['neg_cum'] < -leak_thresh:
                        is_correction = True
                        start_correction = min(start_correction, current_day)
                        is_flag = 'Neg'

                    # === Leak 状态判定 ===
                    if curr == 'Leak':
                        # 开始或持续渐进
                        if not s['in_leak']:
                            s['in_leak'] = True
                            s['leak_start'] = current_day
                            s['leak_cum'] = max(0, day_result.loc[current_day, col])
                            s['leak_days'] = 1
                        else:
                            # 突变结束（prev was Sudden）
                            s['leak_cum'] = max(0, s['leak_cum'] + day_result.loc[current_day, col])
                            s['leak_days'] += 1
                        s['recent_noise_days'] = 0  # 重置噪声计数

                    elif curr == 'Noise':
                        # 结束标志
                        if s['in_leak']:
                            if (s['leak_cum'] < leak_thresh):
                                s.update({
                                    'in_leak': False,
                                    'leak_start': None,
                                    'leak_cum': 0.0,
                                    'leak_days': 0,
                                    'recent_noise_days': 0,
                                    'is_leak': False
                                })
                    elif curr == 'End':
                        # 状态为End 一定模型校正
                        is_correction = True
                        start_correction = min(start_correction, current_day)
                        is_flag = 'End'

                    # 更新结果持久化
                    fis_result[col].at[current_day, 'leak_cum'] = s['leak_cum']
                    fis_result[col].at[current_day, 'leak_days'] = s['leak_days']

                    # 在渐进状态下 达到 阈值 则直接标记并进行模型校正
                    if (s['leak_cum'] >= leak_thresh):
                        mark_leak(col, s['leak_start'], current_day)
                        is_correction = True
                        start_correction = min(start_correction, s['leak_start'])
                        is_flag = 'Leak'

                    s['prev_state'] = curr
            pressure_result = pd.concat([pressure_result, fix_pressure_2.loc[current_day].to_frame().T])
            if is_correction:
                mark_model_correction(start_correction, current_day, is_flag)
                if is_flag == 'Leak':
                    for col in state_store.keys():
                        s = state_store[col]
                        s.update({
                            'in_leak': False,
                            'leak_start': None,
                            'leak_cum': 0.0,
                            'recent_noise_days': 0,
                            'is_leak': False
                        })

        return fis_result, fix_pressure_2, pressure_result

    def run_experiment(
            DMA,
            fix_pressure,
            inp_file,
            leak_data,
            leak_identification_file="leak_identification",
            leak_location_file="leak_location"
    ):
        """运行一次实验，返回检测结果"""

        """ 初始化数据 """
        # 区域传感器
        select_sensors = DMA_AB_sensors if DMA == 'AB' else DMA_C_sensors
        # FIS 参数
        fis_params = init_fis_params(DMA)
        # 泄漏点和需求量
        leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
        # 水网络模型
        wn = wntr.network.WaterNetworkModel(inp_file)
        # 管道坐标
        pipe_coords = get_coordinate(wn)
        # 泄漏事件
        leak_data = leak_data
        leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
        # 合并坐标
        leak_info = leak_events.join(pipe_coords, on="linkID")
        # 转换时间格式
        leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
        leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

        # 模糊推理泄漏识别
        fis_result, day_result_step, pressure_result = run_no_state(fix_pressure, fis_params, sensors=select_sensors)
        leak_flag_times, detection_flag_times = collect_leak_detection_times(fis_result)
        pressure_detection = pressure_result.loc[list(detection_flag_times['time']), select_sensors].astype(
            float)
        # - fis_result: 基于FIS的泄漏识别结果
        # - pressure_result: 修正后的每天传感器压力数据
        # - pressure_detection: 基于FIS的泄漏识别结果中检测到泄漏的传感器压力数据
        # - leak_flag_times: 泄漏事件的起始事件记录
        # - detection_flag_times: 泄漏事件的检测事件和最终起始时间预测
        with open(os.path.join(RESULT_PATH, f"{leak_identification_file}_{DMA}.pickle"), "wb") as f:
            pickle.dump((fis_result, pressure_result, pressure_detection, detection_flag_times), f)
            print(f"区域{DMA} FIS 泄漏识别结果已保存至 {RESULT_PATH}/{leak_identification_file}_{DMA}.pickle")

        # 计算泄漏重心
        center_df = pd.DataFrame(None, index=pressure_detection.index, columns=['x_center', 'y_center', 'pipes'])
        for i in pressure_detection.index:
            pi, _, _ = compute_leak_probability(pressure_detection.loc[i])
            center, S = compute_leakage_center_single(pi, pipe_coords)
            if center is None:
                # 删除该索引行
                center_df.drop(index=i, inplace=True)
                continue
            center_df.loc[i, 'x_center'] = center[0]
            center_df.loc[i, 'y_center'] = center[1]
            center_df.loc[i, 'pipes'] = S
        # 1) 由中心 → 评价表 leakage_center
        leakage_center = get_leakage_center(
            center_df=center_df,
            pipe_coords=pipe_coords,
            leak_info=leak_info,
            distance_thresh=300.0
        )
        # 2) 由 leakage_center + detection_flag_times → 纠错表
        detection_correction, leak_info_remaining = generate_detection_correction(
            leakage_center=leakage_center,
            leak_info=leak_info,
        )
        # 3) 计算纠错表格的经济值
        economic_value = calculate_economic_value(
            detection_correction=detection_correction,
            leakage_center=leakage_center,
            leak_info=leak_info,
            leak_demands=leak_demands
        )

        with open(os.path.join(RESULT_PATH, f"{leak_location_file}_{DMA}.pickle"), "wb") as f:
            pickle.dump((center_df, leakage_center, detection_correction, economic_value), f)
            print(f"区域{DMA} FIS 泄漏位置结果已保存至 {RESULT_PATH}/{leak_location_file}_{DMA}.pickle")

    DMA = 'AB'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_state_leak_identification",
                   leak_location_file="no_state_leak_location")
    DMA = 'C'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="no_state_leak_identification",
                   leak_location_file="no_state_leak_location")



    with open(os.path.join(RESULT_PATH, f"no_state_leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_state_leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_state_leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"no_state_leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "no_state_leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC 状态泄漏检测结果已保存至 {RESULT_PATH}/no_state_leak_detection_ABC.pickle")

# ===========================================
# 模型迁移
# ===========================================
def Transform():
    with open(os.path.join(File_PATH, "real_fix_pressure.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Real.inp")

    DMA = 'AB'
    run_experiment(DMA, fix_pressure, inp_file, leak_data, leak_identification_file="real_leak_identification", leak_location_file="real_leak_location")
    DMA = 'C'
    run_experiment(DMA, fix_pressure, inp_file, leak_data, leak_identification_file="real_leak_identification", leak_location_file="real_leak_location")

    with open(os.path.join(RESULT_PATH, f"real_leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"real_leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"real_leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"real_leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "leak_demands.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "real_leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC FIS 泄漏检测结果已保存至 {RESULT_PATH}/real_leak_detection_ABC.pickle")

# ===========================================
# 2019 年数据运行
# ===========================================
def Run_2019():
    with open(os.path.join(File_PATH, "fix_pressure_2019.pickle"), 'rb') as f:
        fix_pressure = pickle.load(f)

    inp_file = os.path.join(DATA_PATH, "L-TOWN_v2_Real.inp")
    leak_data = leak_2019_data

    DMA = 'AB'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="2019_leak_identification", leak_location_file="2019_leak_location")
    DMA = 'C'
    run_experiment(DMA, fix_pressure, inp_file, leak_data,
                   leak_identification_file="2019_leak_identification", leak_location_file="2019_leak_location")

    with open(os.path.join(RESULT_PATH, f"2019_leak_identification_AB.pickle"), 'rb') as f:
        fis_result_AB, pressure_result_AB, pressure_detection_AB, detection_flag_times_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"2019_leak_identification_C.pickle"), 'rb') as f:
        fis_result_C, pressure_result_C, pressure_detection_C, detection_flag_times_C = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"2019_leak_location_AB.pickle"), 'rb') as f:
        pi_AB, center_df_AB, leakage_center_AB, detection_correction_AB, economic_value_AB = pickle.load(f)
    with open(os.path.join(RESULT_PATH, f"2019_leak_location_C.pickle"), 'rb') as f:
        pi_C, center_df_C, leakage_center_C, detection_correction_C, economic_value_C = pickle.load(f)

    # 泄漏点和需求量
    leak_demands = pd.read_csv(os.path.join(File_PATH, "2019_leak_demand.csv"))
    # 水网络模型
    wn = wntr.network.WaterNetworkModel(inp_file)
    # 管道坐标
    pipe_coords = get_coordinate(wn)
    # 泄漏事件
    leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
    # 合并坐标
    leak_info = leak_events.join(pipe_coords, on="linkID")
    # 转换时间格式
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"]).dt.normalize()
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"]).dt.normalize()

    # 计算两个区域的整体经济
    detection_correction_AB, leak_info_remaining = generate_detection_correction(
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
    )

    detection_correction_C, _ = generate_detection_correction(
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
    )
    economic_value_AB = calculate_economic_value(
        detection_correction=detection_correction_AB,
        leakage_center=leakage_center_AB,
        leak_info=leak_info,
        leak_demands=leak_demands
    )

    economic_value_C = calculate_economic_value(
        detection_correction=detection_correction_C,
        leakage_center=leakage_center_C,
        leak_info=leak_info_remaining,
        leak_demands=leak_demands
    )

    economic_value = economic_value_AB.copy()
    economic_value['S'] += economic_value_C['S']
    economic_value['cost_w'] += economic_value_C['cost_w']
    economic_value['cost_r'] += economic_value_C['cost_r']
    economic_value['tp'] += economic_value_C['tp']
    economic_value['fp'] += economic_value_C['fp']

    with open(os.path.join(RESULT_PATH, "2019_leak_detection_ABC.pickle"), "wb") as f:
        pickle.dump((detection_correction_AB, detection_correction_C, economic_value), f)
        print(f"区域ABC FIS 泄漏检测结果已保存至 {RESULT_PATH}/2019_leak_detection_ABC.pickle")

if __name__ == '__main__':
    # RFLOMA()
    # NO_FIS()
    # Transform()
    # NO_Correction()
    # NO_Removal()
    Run_2019()
