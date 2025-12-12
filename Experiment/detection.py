import pandas as pd

from Experiment.model import *

# ==========================================================
# 泄漏模糊推理系统（FIS）
# ==========================================================
class LeakageFIS:
    def __init__(self, thr_diff1_rate, thr_diff4_rate, thr_diff7_rate):
        self.th_diff1_rate = thr_diff1_rate
        self.th_diff4_rate = thr_diff4_rate
        self.th_diff7_rate = thr_diff7_rate
        # 置信度映射
        self.CONFIDENCE_MAP = {'Low': 0.5, 'High': 0.8}

        # 规则库示例（可按需扩展）
        # 格式: ((diff1, diff4, diff7), (state, confidence))
        self.RULES = [
            # === Leak Leak (渐进泄漏) ===
            (('PD', 'PD', 'PD'), ('Leak', 'High')),
            (('PD', 'PD', 'ND'), ('Leak', 'High')),
            (('PD', 'PD', 'Z'), ('Leak', 'High')),
            (('PD', 'ND', 'PD'), ('Leak', 'High')),
            (('PD', 'Z', 'PD'), ('Leak', 'High')),
            (('Z', 'PD', 'PD'), ('Leak', 'High')),
            # 低置信度
            (('Z', 'Z', 'PD'), ('Leak', 'Low')),
            (('Z', 'PD', 'Z'), ('Leak', 'Low')),
            (('PD', 'Z', 'Z'), ('Leak', 'Low')),
            (('PD', 'Z', 'ND'), ('Leak', 'Low')),
            (('PD', 'ND', 'Z'), ('Leak', 'Low')),
            (('PD', 'ND', 'ND'), ('Leak', 'Low')),
            (('Z', 'ND', 'PD'), ('Leak', 'Low')),
            (('Z', 'PD', 'ND'), ('Leak', 'Low')),


            # === End (泄漏结束) ===
            # 高置信度
            (('ND', 'ND', 'PD'), ('End', 'High')),
            (('ND', 'ND', 'Z'), ('End', 'High')),
            (('ND', 'ND', 'ND'), ('End', 'High')),
            (('ND', 'PD', 'ND'), ('End', 'High')),
            (('ND', 'Z', 'ND'), ('End', 'High')),
            # 低置信度
            (('ND', 'PD', 'PD'), ('End', 'Low')),
            (('ND', 'PD', 'Z'), ('End', 'Low')),
            (('ND', 'Z', 'Z'), ('End', 'Low')),
            (('ND', 'Z', 'PD'), ('End', 'Low')),

            # === Noise (噪音) ===
            # 高置信度
            (('Z', 'Z', 'Z'), ('Noise', 'High')),
            # 低置信度
            (('Z', 'Z', 'ND'), ('Noise', 'Low')),
            (('Z', 'ND', 'Z'), ('Noise', 'Low')),
            (('Z', 'ND', 'ND'), ('Noise', 'Low')),
        ]

    # 隶属函数定义
    def trapezoid(self, x, a, b, c, d):
        """梯形隶属函数"""
        return max(min((x - a) / (b - a), 1, (d - x) / (d - c)), 0) if b != a and d != c else 0

    def triangle(self, x, a, b, c):
        """三角隶属函数"""
        return max(min((x - a) / (b - a), (c - x) / (c - b)), 0) if b != a and c != b else 0

    def update_params(self, origin):
        self.TH_DIFF1 = self.th_diff1_rate * (origin + 0.1)
        self.TH_DIFF4 = self.th_diff4_rate * (origin + 0.1)
        self.TH_DIFF7 = self.th_diff7_rate * (origin + 0.1)

    # --- 隶属函数 ---
    def fuzzify_diff1(self, x):
        return {
            'ND': self.trapezoid(x, -1e6, -self.TH_DIFF1 * 2, -self.TH_DIFF1 * 1.5, -self.TH_DIFF1 * 0.5),
            'Z': self.triangle(x, -self.TH_DIFF1, 0, self.TH_DIFF1),
            'PD': self.trapezoid(x, self.TH_DIFF1 * 0.5, self.TH_DIFF1 * 1.5, self.TH_DIFF1 * 2, 1e6)
        }

    def fuzzify_diff4(self, x):
        return {
            'ND': self.trapezoid(x, -1e6, -self.TH_DIFF4 * 2, -self.TH_DIFF4, 0),
            'Z': self.triangle(x, -self.TH_DIFF4, 0, self.TH_DIFF4),
            'PD': self.trapezoid(x, 0, self.TH_DIFF4, self.TH_DIFF4 * 2, 1e6)
        }

    def fuzzify_diff7(self, x):
        return {
            'ND': self.trapezoid(x, -1e6, -self.TH_DIFF7 * 2, -self.TH_DIFF7, 0),
            'Z': self.triangle(x, -self.TH_DIFF7, 0, self.TH_DIFF7),
            'PD': self.trapezoid(x, 0, self.TH_DIFF7, self.TH_DIFF7 * 2, 1e6)
        }

    # --- 绘制隶属度函数 ---
    def plot_membership_functions(self):
        x1 = np.linspace(-self.TH_DIFF1 * 3, self.TH_DIFF1 * 3, 500)
        x4 = np.linspace(-self.TH_DIFF4 * 3, self.TH_DIFF4 * 3, 500)
        x7 = np.linspace(-self.TH_DIFF7 * 3, self.TH_DIFF7 * 3, 500)

        plt.figure(figsize=(15, 4))
        for k in ['ND', 'Z', 'PD']:
            plt.plot(x1, [self.fuzzify_diff1(v)[k] for v in x1], label=k)
        plt.title("Membership Functions - diff_1")
        plt.xlabel("diff_1")
        plt.ylabel("Membership degree")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(15, 4))
        for k in ['ND', 'Z', 'PD']:
            plt.plot(x4, [self.fuzzify_diff4(v)[k] for v in x4], label=k)
        plt.title("Membership Functions - diff_4")
        plt.xlabel("diff_4")
        plt.ylabel("Membership degree")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(15, 4))
        for k in ['ND', 'Z', 'PD']:
            plt.plot(x7, [self.fuzzify_diff7(v)[k] for v in x7], label=k)
        plt.title("Membership Functions - diff_7")
        plt.xlabel("diff_7")
        plt.ylabel("Membership degree")
        plt.grid(True)
        plt.legend()
        plt.show()

    # --- 单点评估 ---
    def evaluate_point(self, origin, diff1_val, diff4_val, diff7_val):
        self.update_params(origin)
        mu_diff1 = self.fuzzify_diff1(diff1_val)
        mu_diff4 = self.fuzzify_diff4(diff4_val)
        mu_diff7 = self.fuzzify_diff7(diff7_val)
        # print(mu_diff1, mu_diff4, mu_diff7)

        output_strength = {'Leak': 0, 'End': 0, 'Noise': 0}

        for conds, (state, conf_label) in self.RULES:
            d1, d4, d7 = conds
            strength = min(mu_diff1[d1], mu_diff4[d4], mu_diff7[d7]) * self.CONFIDENCE_MAP[conf_label]
            # print(d1,d4,d7, strength)
            output_strength[state] = max(output_strength[state], strength)

        final_state = max(output_strength, key=lambda k: output_strength[k])
        return final_state, output_strength

    # --- 批量时序处理 ---
    def evaluate_series(self, origin, df):
        results = []
        for _, row in df.iterrows():
            state, conf = self.evaluate_point(origin, row['diff_1'], row['diff_4'], row['diff_7'])
            results.append([origin, state, conf['Leak'], conf['End'], conf['Noise']])
        df_res = df.copy()
        df_res[['origin_data','State', 'Leak', 'End', 'Noise']] = results
        return df_res

        # --- 绘制置信度随时间变化 ---

    def plot_confidence_series(self, df_res):
        plt.figure(figsize=(15, 5))
        x = np.arange(len(df_res))
        for state in ['Leak', 'End', 'Noise']:
            plt.plot(x, df_res[state], label=state)
        plt.xlabel("Time Index")
        plt.ylabel("Confidence")
        plt.title("Leakage FIS Confidence Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

# ==========================================================
# 多尺度特征计算
# ==========================================================
def compute_multiscale_features(y: pd.Series):
    """
    基于时间差分的多尺度特征提取
    :param y: 传感器原始时序数据
    :param short_win: 短期窗口大小
    :param long_win: 长期窗口大小
    :return: DataFrame(diff_1, diff_4, diff_7)
    """
    df = pd.DataFrame({'origin_data': y})
    df['diff_1'] = df['origin_data'].diff(1).fillna(df['origin_data'])
    df['diff_4'] = df['origin_data'].diff(4).fillna(df['origin_data'])
    df['diff_7'] = df['origin_data'].diff(7).fillna(df['origin_data'])
    return df

# ==========================================================
# 初始化FIS参数
# ==========================================================
def init_fis_params(DMA = 'AB'):
    """
    初始化FIS参数
    """
    # DMAC 和 DMA AB 有两组参数
    if DMA == 'C':
        return {
            'leak_thresh': 0.80,
            'th_diff1': 1.00,
            'th_diff4': 0.10,
            'th_diff7': 0.20,
            'top_p': 0.05,
            'max_leak': 5,
            'lasso_threshold': 0.2,
            'lasso_lambda': 0.01,
        }
    else:
        return {
            'leak_thresh': 0.20,
            'th_diff1': 0.25,
            'th_diff4': 0.01,
            'th_diff7': 0.20,
            'top_p': 0.05,
            'max_leak': 5,
            'lasso_threshold': 0.2,
            'lasso_lambda': 0.01,
        }


# ==========================================================
# 获取管道坐标
# ==========================================================
def get_coordinate(wn, pipe_name_list=None):
    """获取指定管道的坐标（取中点）"""
    if pipe_name_list is None:
        pipe_name_list = wn.pipe_name_list
    coords = pd.DataFrame(index=pipe_name_list, columns=['x', 'y'], dtype=float)
    for name in pipe_name_list:
        link = wn.get_link(name)
        start_node = wn.get_node(link.start_node)
        end_node = wn.get_node(link.end_node)
        coords.loc[name, 'x'] = (start_node.coordinates[0] + end_node.coordinates[0]) / 2
        coords.loc[name, 'y'] = (start_node.coordinates[1] + end_node.coordinates[1]) / 2
    return coords

# ==========================================================
# 获取节点坐标
# ==========================================================
def get_coordinate_node(wn, node_name_list=None):
    """获取指定节点的坐标"""
    if node_name_list is None:
        node_name_list = wn.node_name_list
    coords = pd.DataFrame(index=node_name_list, columns=['x', 'y'], dtype=float)
    for name in node_name_list:
        node = wn.get_node(name)
        coords.loc[name, 'x'] = node.coordinates[0]
        coords.loc[name, 'y'] = node.coordinates[1]
    return coords

# ==========================================================
# 基于FIS的泄漏识别
# ==========================================================
def run(fix_pressure, fis_param, sensors = None):
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
    correction_value = pd.DataFrame(0, index=fix_pressure_2.index, columns=fix_pressure_2.columns)

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
                    correction_value.loc[end_dt:, colname] += fix_pressure_2.loc[end_dt, colname]
                    fix_pressure_2.loc[:, colname] -= (fix_pressure_2.loc[end_dt, colname])
                elif flag == 'End':
                    correction_value.loc[end_dt:, colname] += (fix_pressure_2.loc[end_dt, colname])
                    fix_pressure_2.loc[:, colname] -= (fix_pressure_2.loc[end_dt, colname])
                else:
                    current_vals = fix_pressure_2.loc[end_dt, sensors]
                    v_min = current_vals.min()
                    if v_min < 0:
                        for colname in sensors:
                            fix_pressure_2.loc[:, colname] -= v_min
                            correction_value.loc[end_dt:, colname] += v_min



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
            time = current_day-pd.Timedelta(days=1) if current_day != day_pressure.index[0] else current_day
            fis_out = fis.evaluate_series(day_pressure.loc[time,col],features[['diff_1', 'diff_4', 'diff_7']])

            if current_day in features.index:
                today_feature = features.loc[[current_day]]
                today_fis = fis_out.loc[[current_day]]

                if sensor_features[col].empty:
                    sensor_features[col] = today_feature.copy()
                else:
                    sensor_features[col] = pd.concat([sensor_features[col], today_feature])

                if fis_result[col].empty:
                    fis_result[col] = today_fis.copy()
                    for fld in ['leak_cum', 'neg_cum' ,'leak_days', 'leak_flag', 'model_correction',
                                'detection_flag']:
                        fis_result[col][fld] = np.nan if fld not in ['leak_flag', 'model_correction'] else False
                else:
                    fis_result[col] = pd.concat(
                        [fis_result[col], today_fis[~today_fis.index.isin(fis_result[col].index)]])
                    for fld in ['leak_cum', 'neg_cum' ,'leak_days', 'leak_flag', 'model_correction',
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
                        s['leak_cum'] = max(0, s['leak_cum']+day_result.loc[current_day, col])
                        s['leak_days'] += 1
                    s['recent_noise_days'] = 0  # 重置噪声计数

                elif curr == 'Noise':
                    # 若已处于渐进状态且渐进天数>4
                    if s['in_leak'] and s['leak_days'] > 4:
                        s['recent_noise_days'] += 1
                        # 若噪声未连续3天以内，继续保持渐进状态
                        if s['recent_noise_days'] <= noise_tolerance_days:
                            s['leak_cum'] = max(0, s['leak_cum']+day_result.loc[current_day, col])
                            fis_result[col].at[current_day, 'State'] = 'Leak'
                        else:
                            # 超出容忍天数，认为渐进结束
                            # 若此前已满足阈值或天数，则标记模型校正区间
                            if (s['leak_cum'] < leak_thresh):
                                s.update({
                                    'in_leak': False,
                                    'leak_start': None,
                                    'leak_cum': 0.0,
                                    'leak_days': 0,
                                    'recent_noise_days': 0,
                                    'is_leak': False
                                })
                    else:
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


# ==========================================================
# 收集泄漏检测时间
# ==========================================================
def collect_leak_detection_times(fis_result):
    leak_records = []
    detect_records = []

    for sensor, df in fis_result.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            continue  # 跳过无时间索引的结果

        if 'leak_flag' in df.columns:
            leak_times = df.index[df['leak_flag'] == True]
            leak_records.extend([(t, sensor) for t in leak_times])

        if 'detection_flag' in df.columns:
            detect_times = df.index[df['detection_flag'] == True]
            detect_records.extend([(t, sensor) for t in detect_times])

    # 转换为DataFrame
    leak_df = pd.DataFrame(leak_records, columns=['time', 'sensor'])
    detect_df = pd.DataFrame(detect_records, columns=['time', 'sensor'])

    # 按时间聚合：每个时间点对应传感器列表
    leak_flag_times = (
        leak_df.groupby('time')['sensor']
        .apply(list)
        .reset_index()
        .rename(columns={'sensor': 'sensors'})
        .sort_values('time')
        .reset_index(drop=True)
    )

    detection_flag_times = (
        detect_df.groupby('time')['sensor']
        .apply(list)
        .reset_index()
        .rename(columns={'sensor': 'sensors'})
        .sort_values('time')
        .reset_index(drop=True)
    )

    sensor_leak_dict = {}
    for _, row in leak_flag_times.iterrows():
        t = row['time']
        for s in row['sensors']:
            sensor_leak_dict.setdefault(s, []).append(t)

    # 对每个传感器的时间列表进行排序
    for s in sensor_leak_dict:
        sensor_leak_dict[s] = sorted(sensor_leak_dict[s])

    # 增加列 max_leak_time
    max_leak_times = []
    for _, row in detection_flag_times.iterrows():
        det_time = row['time']
        sensors = row['sensors']
        max_times = []

        for s in sensors:
            if s in sensor_leak_dict:
                # 只保留小于等于当前 detection_time 的 leak_time
                valid_times = [t for t in sensor_leak_dict[s] if t <= det_time]
                if valid_times:
                    max_times.append(max(valid_times))

        if max_times:
            max_leak_times.append(max(max_times))
        else:
            max_leak_times.append(pd.NaT)

    detection_flag_times['max_leak_time'] = max_leak_times

    return leak_flag_times, detection_flag_times

# ==========================================================
# 使用软硬分类进行候选管道概率划分
# ==========================================================
def compute_leak_probability(
    sensor_values: pd.Series,
    n_s: int = 1,
    k: int = 10,
    tau_s: float = 0.995,
    eps: float = 1e-8,
    matrix_path: str = f"{File_PATH}/sensor_sort.pickle",
):
    """
    候选管道概率划分：局部硬划分 + 全局软相似度

    参数
    ----
    sensor_values : pd.Series
        当前时刻所有传感器的修正压降观测值 v(t)，
        index 为传感器 ID（需能与敏感度矩阵行索引对应）。
    n_s : int
        局部硬划分时，Top-n_s 高响应传感器数量。
    k : int
        软相似度时，观测与模板各自取 Top-k 传感器。
    tau_s : float
        相似度阈值 τ_s，用于筛选候选管道。
    eps : float
        数值稳定项。
    matrix_path : str
        敏感度矩阵（或旧 membership 矩阵）pickle 文件路径，
        形状为 [n_sensors, n_pipes]，行：传感器，列：管道。

    返回
    ----
    pi : pd.Series
        最终泄漏概率分布 π_j(t)，index 为管道 ID。
    H : pd.Series
        局部硬划分指标 H_j(t) ∈ {0,1}。
    S : pd.Series
        软相似度 S_j(t)（未做候选截断前的原始相似度）。
    """
    # === 读取敏感度矩阵 M（行：传感器，列：管道） ===
    with open(matrix_path, "rb") as f:
        M: pd.DataFrame = pickle.load(f)

    # 对齐传感器索引：只保留 M 中存在的传感器
    common_sensors = M.index.intersection(sensor_values.index)
    if len(common_sensors) == 0:
        raise ValueError("传感器索引在观测值与敏感度矩阵中没有交集，请检查索引名称。")

    v = sensor_values.loc[common_sensors].astype(float)  # 观测向量 v(t)

    # 结果容器
    S_list = []   # 软相似度 S_j(t)
    H_list = []   # 硬划分 H_j(t)

    # ---------- (1) 局部硬划分：高响应传感器集合 ----------
    # Top-n_s 观测值最大传感器集合 S_top
    S_top = list(set(v.nlargest(n_s).index) & set(common_sensors))  # S_top = Top_{n_s}(v(t))
    # ---------- (2) 遍历每条管道，计算 H_j(t) 和 S_j(t) ----------
    for pipe in M.columns:
        pipe_profile = M[pipe].astype(float)  # 敏感度模板 m_j

        # --- (2.1) H_j(t)：硬划分 ---
        # argmax_s m_{sj}
        s_star = pipe_profile.idxmax()
        H_j = 1 if s_star in S_top else 0
        H_list.append(H_j)

        # --- (2.2) S_j(t)：全局软相似度 ---
        # Top-k 观测传感器集合 T_v(t)
        T_v = v.nlargest(k).index
        # Top-k 模板敏感传感器集合 T_m(j)
        T_m = pipe_profile.nlargest(k).index

        # 有效子集 S_j^{sub}(t) = (T_v ∪ T_m) ∩ S
        sub_sensors = list((set(T_v) | set(T_m)) & set(common_sensors))
        if len(sub_sensors) == 0:
            S_list.append(0.0)
            continue

        v_sub = v.loc[sub_sensors]
        m_sub = pipe_profile.loc[sub_sensors]
        # 对齐（理论上索引已一致，这里只是保险）
        v_sub, m_sub = v_sub.align(m_sub, join="inner")

        num = float(np.dot(v_sub.values, m_sub.values))
        denom = float(
            np.linalg.norm(v_sub.values) * np.linalg.norm(m_sub.values) + eps
        )
        S_j = num / denom
        if S_j < 0:  # 负相关不视为泄漏贡献
            S_j = 0.0
        S_list.append(S_j)

    # ---------- (3) 组装为 Series ----------
    S = pd.Series(S_list, index=M.columns)  # 软相似度 S_j(t)
    H = pd.Series(H_list, index=M.columns, dtype=float)  # 硬划分指标 H_j(t)

    # ---------- (4) 候选集合 C(t) ----------
    # C(t) = { j | S_j(t) > tau_s } ∪ { j | H_j(t) = 1 }
    C_soft = set(S[S > tau_s].index)
    C_hard = set(H[H > 0].index)
    C = C_soft | C_hard
    # 初始概率 \hat{\pi}_j(t)
    hat_pi = pd.Series(0.0, index=M.columns)
    if len(C) > 0:
        hat_pi.loc[list(C)] = S.loc[list(C)]

    # ---------- (5) 概率归一化 ----------
    total = hat_pi.sum() + eps
    pi = hat_pi / total  # π_j(t)

    return pi, H, S

# ==========================================================
# 计算每个时刻的泄漏重心(只有一个泄漏重心)
# ==========================================================
def compute_leakage_center_single(
    pi: pd.Series,
    pipe_coords: Union[pd.DataFrame, Dict],
    eta_c: float = 0.05,
    delta_d: float = 300.0,
) -> Tuple[Optional[np.ndarray], Iterable]:
    """
    在给定泄漏概率分布 π_j(t) 和管道坐标的前提下，计算“泄漏重心”。

    参数
    ----
    pi : pd.Series
        泄漏概率分布 π_j(t)，index 为管道 ID，值 ∈ [0,1]，建议已归一化。
    pipe_coords : DataFrame 或 dict
        管道几何中心坐标：
        - 若为 DataFrame：index 为管道 ID，列名为 'x','y' 或 'X','Y'；
        - 若为 dict：键为管道 ID，值为 (x, y) 元组。
    eta_c : float
        累积概率阈值 η_c，用于选取高置信管道子集（0~1）。
    delta_d : float
        聚集性阈值 δ_d，控制重心周围最大欧氏距离上限（单位与坐标一致）。

    返回
    ----
    center : np.ndarray or None
        泄漏重心坐标 [x_center, y_center]；若无法计算则为 None。
    selected_pipes : list
        参与重心计算的管道 ID 列表（贪心筛选后的 S）。
    """

    # 1) 预处理：只保留有坐标的管道
    if isinstance(pipe_coords, pd.DataFrame):
        # 兼容 'x','y' 或 'X','Y'
        if {"x", "y"}.issubset(pipe_coords.columns):
            coord_df = pipe_coords[["x", "y"]]
        elif {"X", "Y"}.issubset(pipe_coords.columns):
            coord_df = pipe_coords[["X", "Y"]].rename(columns={"X": "x", "Y": "y"})
        else:
            raise ValueError("pipe_coords DataFrame 中未找到 'x','y' 或 'X','Y' 列")
    else:
        # dict -> DataFrame
        coord_df = pd.DataFrame.from_dict(pipe_coords, orient="index", columns=["x", "y"])

    # 对齐：只保留同时在 pi 和 coord_df 中存在的管道
    common_pipes = coord_df.index.intersection(pi.index)
    if len(common_pipes) == 0:
        return None, []

    pi = pi.loc[common_pipes].astype(float)
    coord_df = coord_df.loc[common_pipes].astype(float)

    # 若所有概率几乎为 0，则无法可靠定位
    if pi.sum() <= 0:
        return None, []

    # 2) 按概率从大到小排序
    pi_sorted = pi.sort_values(ascending=False)
    pipes_sorted = pi_sorted.index.tolist()
    probs_sorted = pi_sorted.values

    # 3) 高置信子集 \hat{C}(t)：累积概率 <= eta_c 的前缀
    cum_prob = 0.0
    high_conf_pipes = []
    for pid, p_val in zip(pipes_sorted, probs_sorted):
        if cum_prob + p_val <= eta_c:
            high_conf_pipes.append(pid)
            cum_prob += p_val
        else:
            break

    # 如果 eta_c 很小或概率很集中，可能只选到 1 条
    if not high_conf_pipes:
        # 回退：至少选取概率最大的 1 条管道
        high_conf_pipes = [pipes_sorted[0]]

    # 4) 贪心聚类：在高置信集合内构造空间聚集子集 S
    # 初始化 S 为概率最大的一条管道
    S = [pipes_sorted[0]]
    coords_S = coord_df.loc[S].values
    probs_S = pi.loc[S].values

    # 依次尝试将 high_conf_pipes \ S 中的管道加入
    for pid in high_conf_pipes:
        if pid in S:
            continue

        # 尝试加入后的新集合 S'
        S_candidate = S + [pid]
        coords_candidate = coord_df.loc[S_candidate].values
        probs_candidate = pi.loc[S_candidate].values

        # 加权重心 C_{S ∪ {ℓ}}(t)
        weight_sum = probs_candidate.sum()
        center_candidate = np.sum(coords_candidate * probs_candidate[:, None], axis=0) / weight_sum

        # 计算所有管道到新重心的最大距离
        dists = np.linalg.norm(coords_candidate - center_candidate[None, :], axis=1)
        if dists.max() <= delta_d:
            # 满足聚集性约束：接纳该管道
            S = S_candidate
            coords_S = coords_candidate
            probs_S = probs_candidate
        # 否则跳过该管道

    # 5) 最终重心（基于 S）
    if len(S) == 0:
        return None, []

    weight_sum = probs_S.sum()
    center = np.sum(coords_S * probs_S[:, None], axis=0) / (weight_sum if weight_sum > 0 else 1.0)

    return center, S


# ==========================================================
# 获取当前时间 t 处于泄漏状态的泄漏事件
# ==========================================================
def get_active_leaks(leak_info: pd.DataFrame, current_time: pd.Timestamp) -> pd.DataFrame:
    """
    给定当前时间，返回当前处于泄漏状态的泄漏事件（index=linkID）。
    """
    info = leak_info.copy()
    info["startTime"] = pd.to_datetime(info["startTime"])
    info["endTime"] = pd.to_datetime(info["endTime"])
    current_time += pd.Timedelta(minutes=60*24-1)
    mask = (info["startTime"] <= current_time) & (current_time <= info["endTime"])
    active = info.loc[mask].set_index("linkID")
    return active

# ==========================================================
# 计算每个时刻的泄漏重心(只有一个泄漏重心)
# ==========================================================
def get_leakage_center(
    center_df: pd.DataFrame,
    pipe_coords: pd.DataFrame,
    leak_info: pd.DataFrame,
    distance_thresh: float = 300.0,
) -> pd.DataFrame:
    """
    已经有每个时刻的泄漏重心 (x_center, y_center)，
    参数
    ----
    center_df : DataFrame
        行索引为时间（DatetimeIndex 或可以转为 datetime 的字符串），
        至少包含列: 'x_center', 'y_center'。
    pipe_coords : DataFrame
        index 为管道 ID（与 leak_info['linkID'] 同一编号体系），
        列包含 'x', 'y'。
    leak_info : DataFrame
        泄漏事件信息，至少包含:
        ['linkID', 'x', 'y', 'startTime', 'endTime']。
    distance_thresh : float
        认为“重心覆盖到泄漏管道”的最大距离阈值。

    返回
    ----
    leakage_center : DataFrame
        index 为时间，列为:
        ['x_center', 'y_center', 'distance', 'pipe', 'min_pipe'] + 每条真实泄漏管道的列
    """

    results = []
    leak_names = leak_info["linkID"].tolist()

    # 方便按 linkID 取坐标
    leak_info_idx = leak_info.set_index("linkID")

    for time, row in center_df.iterrows():
        current_time = pd.to_datetime(time)

        x_center_final = row.get("x_center", np.nan)
        y_center_final = row.get("y_center", np.nan)

        # 如果该时刻没有重心（NaN），直接填空
        if pd.isna(x_center_final) or pd.isna(y_center_final):
            distances = pd.Series(np.nan, index=leak_names)
            results.append(
                [np.nan, np.nan, [], [], np.nan] + distances.tolist()
            )
            continue

        # 当前时刻的真实泄漏集合
        active_leaks = get_active_leaks(leak_info, current_time)

        if active_leaks.empty:
            # 没有真实泄漏，就把所有距离设为 inf
            distances = pd.Series(np.inf, index=leak_names)
        else:
            # 先按 linkID 对齐坐标
            leak_coords = leak_info_idx.loc[leak_names, ["x", "y"]]
            dx = leak_coords["x"].values - x_center_final
            dy = leak_coords["y"].values - y_center_final
            dvals = np.sqrt(dx**2 + dy**2)

            distances = pd.Series(dvals, index=leak_names)

            # 只有当前处于 active 的泄漏才认为“有效”，非 active 的距离设置为 inf
            inactive_mask = ~distances.index.isin(active_leaks.index)
            distances.loc[inactive_mask] = np.inf

        # 距离阈值内的真实泄漏及其 ID
        in_range = distances[distances < distance_thresh]
        in_range_ids = list(in_range.index)
        in_range_vals = list(in_range.values)

        # 找出距离重心最近的管道（用所有管道坐标去匹配）
        try:
            pipe_dx = pipe_coords["x"] - x_center_final
            pipe_dy = pipe_coords["y"] - y_center_final
            pipe_dist = np.sqrt(pipe_dx**2 + pipe_dy**2)
            min_pipe = pipe_dist.idxmin()
        except Exception:
            min_pipe = np.nan

        # 结果一行：
        # [x_center, y_center, 距离列表, 管道ID列表, 最近管道] + 各真实泄漏的距离
        results.append(
            [time, x_center_final, y_center_final,
             in_range_vals, in_range_ids, min_pipe] + distances.tolist()
        )

    columns = (
        ["time", "x_center", "y_center", "distance", "pipe", "min_pipe"]
        + [f"{name}" for name in leak_names]
    )
    leakage_center = pd.DataFrame(results, index=range(len(results)), columns=columns)
    return leakage_center

# ==========================================================
# 生成检测表格，包括 detection_time, start_time, pipe, distance
# ==========================================================
def generate_detection_correction(leakage_center, leak_info):
    """
    生成检测表格，包括 detection_time, start_time, pipe, distance
    """
    records = []

    # 先按 startTime 对 leak_info 排序
    leak_info_sorted = leak_info.sort_values("startTime").reset_index(drop=True)

    # 遍历 detection_flag_times 的每个时间点
    leak_info_remaining = leak_info_sorted.copy()
    for _, row in leakage_center.iterrows():
        det_time = row['time']
        lc_row = leakage_center.iloc[row.name]
        min_pipe = lc_row['pipe']
        min_distance = lc_row['distance']


        # 找出 min_pipe 对应的泄漏事件
        pipe_leaks = leak_info_remaining[leak_info_remaining['linkID'].isin(min_pipe)]
        # 默认误报
        start_time = pd.NaT

        if min_distance != []:
            for idx, leak in pipe_leaks.iterrows():
                # 当前 detection 在该泄漏事件时间段内
                if leak['startTime'] <= det_time+pd.Timedelta(minutes=60*24-1) <= leak['endTime']:
                    start_time = leak['startTime']
                    pipe = leak['linkID']
                    leak_info_remaining = leak_info_remaining.drop(idx)
                    records.append([det_time, start_time, pipe, lc_row[pipe]])
        else:
            records.append([det_time, start_time, np.nan, np.nan])

    # 构造结果表格
    detection_correction = pd.DataFrame(records, columns=['detection_time', 'start_time', 'pipe', 'distance'])
    return detection_correction, leak_info_remaining


# ==========================================================
# 计算经济值
# ==========================================================
def calculate_economic_value(
    detection_correction: pd.DataFrame,
    leakage_center: pd.DataFrame,
    leak_info: pd.DataFrame,
    leak_demands: pd.DataFrame,
    unit_price: float = 0.8,
    unit_volume: float = 28432.0,
    repair_unit_cost: float = 500.0,
):
    """
    计算纠错表格的经济值（加入时间类型统一转换，避免 str 与 Timestamp 混用报错）

    参数
    ----
    detection_correction : DataFrame
        必须包含列 ['detection_time', 'pipe', 'distance']（pipe 可能为 NaN）
    leakage_center : DataFrame
        用于估算维护成本（按时间步数量统计）
    leak_info : DataFrame
        至少包含 ['linkID', 'startTime', 'endTime']，
        startTime / endTime 将会被统一转为 datetime
    leak_demands : DataFrame
        泄漏流量序列：
        - 如果存在 'Timestamp' 列，则使用该列作为时间索引
        - 否则使用现有 index，并统一转为 datetime
        列名形如 'xxx_leaknode' 或 'pipeID'

    其他参数
    --------
    unit_price : float
        单位水价（默认 0.8）
    unit_volume : float
        单位体积经济权重（默认 28432）
    repair_unit_cost : float
        每个时间步的维护成本（默认 500）

    返回
    ----
    dict，包含：
        perfect, S, cost_w, cost_r, tp, fp, total
    """

    # -------- 1. 时间列统一转为 datetime --------
    leak_demands = leak_demands.copy()

    if "Timestamp" in leak_demands.columns:
        leak_demands["Timestamp"] = pd.to_datetime(leak_demands["Timestamp"])
        leak_demands = leak_demands.set_index("Timestamp")
    else:
        leak_demands.index = pd.to_datetime(leak_demands.index)

    leak_info = leak_info.copy()
    leak_info["startTime"] = pd.to_datetime(leak_info["startTime"])
    leak_info["endTime"] = pd.to_datetime(leak_info["endTime"])

    detection_correction = detection_correction.copy()
    detection_correction["detection_time"] = pd.to_datetime(
        detection_correction["detection_time"]
    )

    # -------- 2. 统计量：tp / fp / total --------
    leak_num = len(leak_demands.columns)          # 你当前逻辑：以泄漏流量列数为“总数”
    detection_num = len(detection_correction.dropna(subset=["pipe"]))
    error_num = len(detection_correction) - detection_num

    tp = detection_num
    fp = error_num
    total = leak_num

    # -------- 3. 计算漏水收益 cost_w --------
    cost_w = 0.0

    for idx in detection_correction.index:
        pipe = detection_correction.loc[idx, "pipe"]

        # pipe 为 NaN → 视为误报，不计收益
        if pd.isna(pipe):
            continue

        detection_time = detection_correction.loc[idx, "detection_time"]

        # 查找该 pipe 对应的泄漏时间段
        info_rows = leak_info[leak_info["linkID"] == pipe]
        if info_rows.empty:
            # 找不到该管道对应的泄漏事件，跳过
            # print(f"在 leak_info 中未找到管道 {pipe} 的泄漏记录")
            continue

        end_time = info_rows["endTime"].iloc[0]

        # 如果 detection_time 在该泄漏事件结束之后，也不计收益
        if detection_time > end_time:
            continue

        # 根据列名匹配泄漏流量
        col_leaknode = f"{pipe}_leaknode"
        col_plain = f"{pipe}"

        if col_leaknode in leak_demands.columns:
            col_name = col_leaknode
        elif col_plain in leak_demands.columns:
            col_name = col_plain
        else:
            print(f"管道 {pipe} 没有对应的泄漏节点或流量数据")
            continue

        # 防御性处理：截断时间范围到 leak_demands 的索引范围
        t_min, t_max = leak_demands.index.min(), leak_demands.index.max()
        t_start = max(detection_time, t_min)
        t_end = min(end_time, t_max)

        if t_start > t_end:
            # 时间窗口与流量数据没有交集
            continue

        # 该管道在整个周期内的总泄漏量（用于归一化）
        total_leak = leak_demands[col_name].sum()
        if total_leak <= 0:
            continue

        # 检测之后到结束的泄漏量占比
        leak_segment = leak_demands.loc[t_start:t_end, col_name].sum()
        contrib = (leak_segment / total_leak) * unit_volume

        cost_w += contrib

    # 单位水价
    cost_w *= unit_price

    # -------- 4. 维护成本 cost_r --------
    # 这里沿用你的逻辑：每个时间步一个固定维护成本
    cost_r = repair_unit_cost * len(leakage_center.index)

    # 综合指标
    S = cost_w - cost_r

    # -------- 5. 理想收益 perfect --------
    perfect = 0.0
    for col in leak_demands.columns:
        total_leak = leak_demands[col].sum()
        if total_leak <= 0:
            continue
        # 全部被“完美利用”的情况
        contrib = (total_leak / total_leak) * unit_volume * unit_price
        perfect += contrib

    return {
        "perfect": perfect,
        "S": S,
        "cost_w": cost_w,
        "cost_r": cost_r,
        "tp": tp,
        "fp": fp,
        "total": total,
    }


import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
# =====================
# Group Lasso
# =====================
def group_lasso_admm_scaled(
    v_scaled: np.ndarray,
    M_scaled: np.ndarray,
    groups_local: dict,
    lam: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
):
    N, P = M_scaled.shape
    H = M_scaled
    Ht = H.T
    HtH = Ht @ H
    A = HtH + rho * np.eye(P)
    A_inv = np.linalg.inv(A)

    Hty = Ht @ v_scaled
    x = np.zeros(P)
    z = np.zeros(P)
    u = np.zeros(P)

    for it in range(max_iter):
        rhs = Hty + rho * (z - u)
        x_new = A_inv @ rhs

        z_new = np.zeros_like(z)
        for g_id, idx in groups_local.items():
            v_g = x_new[idx] + u[idx]
            norm_v = np.linalg.norm(v_g, 2)
            if norm_v <= lam / rho:
                z_new[idx] = 0.0
            else:
                factor = 1.0 - (lam / (rho * norm_v))
                z_new[idx] = factor * v_g

        u_new = u + (x_new - z_new)

        r_norm = np.linalg.norm(x_new - z_new)
        x, z, u = x_new, z_new, u_new
        if r_norm < tol:
            break

    v_hat_scaled = H @ z
    residual = np.linalg.norm(v_scaled - v_hat_scaled)
    return z, residual


# =====================
# 稀疏/非稀疏求解器：LS / Ridge / LASSO / SBL / Group Lasso / Elastic Net
# =====================
def solve_sparse_weights_scaled(
    v_scaled: np.ndarray,
    M_scaled: np.ndarray,
    algorithm: str = "lasso",
    lambda_: float = 0.1,
    max_iter: int = 200,
    groups_local: dict = None,   # Group Lasso 需要的 group 映射
):
    """
    稀疏求解器（在“已标准化空间”中工作）：
        给定 v_scaled, M_scaled，求解 v_scaled ≈ M_scaled @ w

    支持算法：
        "ls" / "least_squares" : 最小二乘 (不加正则)
        "ridge"                : L2 正则（岭回归）
        "lasso"                : L1 正则
        "elastic_net"/"enet"   : L1 + L2 弹性网络
        "sbl"                  : ARD-SBL
        "group_lasso"          : Group Lasso (group-wise L2)

    参数
    ----
    v_scaled : (N_sensors,)
    M_scaled : (N_sensors, N_pipes)
    lambda_  : 对 LASSO / Ridge / Elastic Net / Group Lasso 为正则系数
    groups_local : dict 或 None
        algorithm="group_lasso" 时必须提供:
            group_id -> 局部管道索引数组

    返回
    ----
    w : (N_pipes,)
        归一化后的非负权重（可视作相对泄漏概率）
    residual : float
        残差范数 ||v_scaled - M_scaled @ w||
    """
    algo = algorithm.lower()
    y = v_scaled
    Phi = M_scaled  # (N, D)
    N, D = Phi.shape

    # --------- 0) 纯最小二乘 LS ---------
    if algo in ("ls", "least_squares"):
        # 最小二乘：min ||y - Phi w||_2^2
        # 用 lstsq 防止病态
        w_raw, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        v_hat_scaled = Phi @ w_raw
        residual = np.linalg.norm(y - v_hat_scaled)
        w = np.abs(w_raw)

    # --------- 0.5) Ridge（L2 正则）---------
    elif algo == "ridge":
        # min ||y - Phi w||_2^2 + lambda * ||w||_2^2
        A = Phi.T @ Phi + lambda_ * np.eye(D)
        b = Phi.T @ y
        w_raw = np.linalg.solve(A, b)
        v_hat_scaled = Phi @ w_raw
        residual = np.linalg.norm(y - v_hat_scaled)
        w = np.abs(w_raw)

    # --------- 0.75) Elastic Net（L1+L2）---------
    elif algo in ("elastic_net", "enet"):
        # 目标：min 0.5||y - Phi w||^2 + lambda*(alpha||w||_1 + (1-alpha)/2 ||w||_2^2)
        # 这里将 lambda_ 作为整体正则强度，alpha 固定为 0.5（L1:L2 各一半）
        model = ElasticNet(
            alpha=lambda_,
            l1_ratio=0.5,      # L1:L2 比例，后续需要可以外部再做成参数
            fit_intercept=False,
            max_iter=10000
        )
        model.fit(Phi, y)
        v_hat_scaled = model.predict(Phi)
        residual = np.linalg.norm(y - v_hat_scaled)
        w = np.abs(model.coef_)

    # --------- 1) LASSO 情况 ---------
    elif algo == "lasso":
        model = Lasso(alpha=lambda_, positive=False, max_iter=10000)
        model.fit(Phi, y)
        v_hat_scaled = model.predict(Phi)
        residual = np.linalg.norm(y - v_hat_scaled)
        w = np.abs(model.coef_)

    # --------- 2) SBL 情况（简单 ARD-SBL 实现） ---------
    elif algo == "sbl":
        gamma = np.ones(D)   # 各管道先验方差 γ_i
        sigma2 = 1.0         # 噪声方差

        for it in range(max_iter):
            Gamma = np.diag(gamma)
            S_inv = sigma2 * np.eye(N) + Phi @ Gamma @ Phi.T
            S = np.linalg.inv(S_inv)

            Sigma = Gamma - Gamma @ Phi.T @ S @ Phi @ Gamma
            mu = (1.0 / sigma2) * Sigma @ Phi.T @ y

            gamma_new = mu**2 + np.diag(Sigma)
            gamma_new = np.maximum(gamma_new, 1e-12)

            if np.linalg.norm(gamma_new - gamma) < 1e-4 * np.linalg.norm(gamma):
                gamma = gamma_new
                break
            gamma = gamma_new

        v_hat_scaled = Phi @ mu
        residual = np.linalg.norm(y - v_hat_scaled)
        w = np.abs(mu)

    # --------- 3) Group Lasso 情况 ---------
    elif algo == "group_lasso":
        if groups_local is None:
            raise ValueError("algorithm='group_lasso' 时必须提供 groups_local")
        w_raw, residual = group_lasso_admm_scaled(
            v_scaled=y,
            M_scaled=Phi,
            groups_local=groups_local,
            lam=lambda_,
            rho=1.0,
            max_iter=max_iter,
            tol=1e-4,
        )
        w = np.abs(w_raw)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 归一化成“概率权重”
    if w.sum() > 0:
        w = w / w.sum()
    return w, residual


# =====================
#  聚类 + 稀疏/非稀疏反演：按传感器聚类
# =====================
def leak_probability_cluster(
    sensor_values: pd.Series,
    p: int = 10,
    sensors: list = None,
    lambda_: float = 0.1,
    leak_count: int = 1,
    algorithm: str = "lasso",
    membership_path: str = f"{File_PATH}/Sensitivity_Matrix.pickle",
    clusters = None,
    cluster_map = None
):
    # === 1. 加载敏感度矩阵 ===
    with open(membership_path, "rb") as f:
        membership = pickle.load(f)   # index=传感器, columns=管道

    # 对齐传感器
    if sensors is not None:
        sensor_values = sensor_values.loc[sensors]
        membership = membership.loc[sensors]

    # === 4. 标准化 ===
    v = sensor_values.values.astype(float)
    scaler_v = StandardScaler(with_std=True)
    v_scaled = scaler_v.fit_transform(v.reshape(-1, 1)).ravel()

    M_full = membership.values.astype(float)
    scaler_M = StandardScaler(with_std=True)
    M_full_scaled = scaler_M.fit_transform(M_full)

    baseline_residual = np.linalg.norm(v_scaled)
    zero_weights = pd.Series(0.0, index=membership.columns)  # 所有管道

    if leak_count == 0:
        return [], baseline_residual, zero_weights

    # === 5. 内部求解函数（几乎不用动） ===
    def compute_residual_and_weights(candidate_pipes, combo_clusters):
        if len(candidate_pipes) == 0:
            return baseline_residual, zero_weights.copy()

        pipe_idx = [membership.columns.get_loc(p) for p in candidate_pipes]
        M_scaled = M_full_scaled[:, pipe_idx]

        groups_local = None
        if algorithm.lower() == "group_lasso":
            pipe_to_local = {pipe: i for i, pipe in enumerate(candidate_pipes)}
            groups_local = {}
            for c in combo_clusters:
                pipes_c = cluster_map.get(c, [])
                local_idx = [
                    pipe_to_local[p] for p in pipes_c if p in pipe_to_local
                ]
                if len(local_idx) > 0:
                    groups_local[c] = np.array(local_idx, dtype=int)
            if not groups_local:
                return baseline_residual, zero_weights.copy()

        w_sub, residual = solve_sparse_weights_scaled(
            v_scaled=v_scaled,
            M_scaled=M_scaled,
            algorithm=algorithm,
            lambda_=lambda_,
            max_iter=200,
            groups_local=groups_local,
        )

        weights = zero_weights.copy()
        weights.loc[candidate_pipes] = w_sub
        return residual, weights

    # === 6. 枚举 k 个“区域”组合 ===
    best_residual = baseline_residual
    best_combination = []
    best_weights = zero_weights.copy()

    for combo in itertools.combinations(clusters, leak_count):
        candidate_pipes = []
        for c in combo:
            candidate_pipes += cluster_map.get(c, [])

        residual, weights = compute_residual_and_weights(candidate_pipes, combo)

        if residual < best_residual:
            best_residual = residual
            best_combination = list(combo)
            best_weights = weights

    return best_combination, best_residual, best_weights.sort_values(ascending=False)

import numpy as np
import pandas as pd
from collections import defaultdict

def compute_leakage_center(
    max_leak,
    times,
    pipe_coords: pd.DataFrame,
    best_k_series: pd.Series,
    best_weights_df: pd.DataFrame,
    leak_info: pd.DataFrame,
    pipe_cluster_df: pd.DataFrame,
    threshold: float = 300.0,
):
    """
    基于管道聚类结果（pipe_cluster_df['cluster']）计算每个时刻的泄漏重心，
    并统计重心与真实泄漏管段之间的距离。

    参数
    ----
    max_leak : int
        最大泄漏点数（用于重心结果 DataFrame 的行数）
    times : list / Index
        时间序列
    pipe_coords : DataFrame
        管道坐标，index=pipe_id, columns=['x','y'] 或类似二维坐标
    best_k_series : Series
        每个时间对应的最优泄漏点数 K_opt，index=times
    best_weights_df : DataFrame
        每个时间对应的管道泄漏权重 w，columns=times, index=pipe_id
    leak_info : DataFrame
        真实泄漏信息，至少包含 ['linkID','x','y']，以及可选的激活状态信息
    pipe_cluster_df : DataFrame
        管道聚类结果，index=pipe_id，必须包含 'cluster' 列
    threshold : float
        统计“阈值内泄漏管道”的距离阈值

    返回
    ----
    leak_distance_df : DataFrame
        每个时间、每个泄漏中心与真实泄漏管段之间的距离信息。
    """

    # === 0) 基本检查 ===
    if "cluster" not in pipe_cluster_df.columns:
        raise ValueError("pipe_cluster_df 必须包含 'cluster' 列。")

    # 将每条管道的簇标签取出来，index=pipe_id, value=cluster_id
    pipe_clusters = pipe_cluster_df["cluster"]

    # === 1) 存重心的 DataFrame：行=最多 max_leak 个中心，列=时间 ===
    leak_centers_df = pd.DataFrame(index=range(max_leak), columns=times, dtype=object)

    # === 2) 遍历每个时间，计算泄漏重心 ===
    for t in times:
        k_opt = int(best_k_series[t])           # 该时刻的“最优泄漏点数”
        weights = best_weights_df[t].fillna(0.) # 该时刻所有管道的泄漏权重 w (Series, index=pipe_id)

        positive_mask = weights > 0
        pipes_pos = weights[positive_mask].index.tolist()

        if len(pipes_pos) > 0:
            X = pipe_coords.loc[pipes_pos].values  # 候选泄漏管道的坐标
            w = weights[positive_mask].values      # 对应权重
        else:
            X = np.empty((0, 2))
            w = np.array([])

        centers = []

        # ---- 情况1：K_opt <= 1，直接算一个整体加权重心 ----
        if k_opt <= 1:
            if w.sum() > 0:
                center = np.average(X, axis=0, weights=w)
            else:
                # 没有任何正权重，就用所有管道坐标的整体均值兜底
                center = pipe_coords.mean(axis=0).values
            centers.append(center.tolist())

        # ---- 情况2：K_opt >= 2，按簇聚合再算加权重心 ----
        else:
            # 先按 cluster 分组：cluster_id -> [pipe_id,...]
            cluster_groups = defaultdict(list)
            for pipe in pipes_pos:
                if pipe in pipe_clusters.index:
                    cluster_groups[pipe_clusters.loc[pipe]].append(pipe)
                # 若不在 cluster_df 里，可以选择忽略或给 warning

            # 计算每个簇的加权重心
            cluster_centers = {}
            for cluster_id, pipes_in_cluster in cluster_groups.items():
                pipes_w = weights[pipes_in_cluster]
                coords = pipe_coords.loc[pipes_in_cluster].values
                if pipes_w.sum() > 0:
                    c = np.average(coords, axis=0, weights=pipes_w.values)
                else:
                    c = coords.mean(axis=0)
                cluster_centers[cluster_id] = c.tolist()

            # 根据簇内权重总和，选权重最大的 k_opt 个簇
            if len(cluster_centers) > 0:
                cluster_weights = {
                    c: weights[cluster_groups[c]].sum() for c in cluster_centers.keys()
                }
                # 排序取 top k_opt
                top_clusters = sorted(cluster_weights.items(), key=lambda x: -x[1])[:k_opt]
                centers = [cluster_centers[c_id] for c_id, _ in top_clusters]
            else:
                centers = []

            # 如果簇数 < k_opt，用权重最大的单管道位置补足
            if len(centers) < k_opt and len(w) > 0:
                needed = k_opt - len(centers)
                top_idx = np.argsort(-w)[:needed]
                for idx in top_idx:
                    centers.append(X[idx].tolist())

        # 将重心写入 leak_centers_df
        for i, c in enumerate(centers):
            leak_centers_df.at[i, t] = c

    # === 3) 计算每个时间的泄漏中心到真实泄漏管段的距离 ===
    leak_names = leak_info["linkID"].tolist()
    all_leaks_coords = leak_info.set_index("linkID")[["x", "y"]]

    records = []

    for t in times:
        centers = leak_centers_df.loc[:, t].dropna().tolist()  # list of [x, y]
        active_leaks = get_active_leaks(leak_info, t)          # 当前时间活跃泄漏（外部函数）

        for i, c in enumerate(centers):
            c = np.array(c)
            distance_list = []
            pipe_list = []
            per_pipe_dist = {}

            for lid, row in all_leaks_coords.iterrows():
                pipe_coord = row.values
                if lid in active_leaks.index:
                    d = np.linalg.norm(pipe_coord - c)
                else:
                    d = np.inf
                per_pipe_dist[lid] = d
                if d <= threshold:
                    distance_list.append(d)
                    pipe_list.append(lid)

            record = {
                "time": t,
                "center_id": i + 1,
                "x_center": c[0],
                "y_center": c[1],
                "distance": distance_list,
                "pipe": pipe_list,
            }
            # 每个真实泄漏管段单独一列
            for lid in leak_names:
                record[lid] = per_pipe_dist.get(lid, np.inf)

            records.append(record)

    leak_distance_df = pd.DataFrame(records)
    return leak_distance_df

def defuzzify_leak_probability_cluster(sensor_values: pd.Series,
                                       p: int = 10,
                                       sensors: list = DMA_AB_sensors,
                                       lambda_: float = 0.1,
                                       leak_count: int = 1):
    """
    基于稀疏回归和聚类优化的泄漏概率计算（支持任意 leak_count=k）

    参数：
        sensor_values : pd.Series 当前时刻传感器值
        DMA_AB_sensors: list 当前区域传感器列表
        p : int, top-p 传感器
        lambda_ : float, LASSO 正则系数
        leak_count : int, 假设泄漏数量 k

    返回：
        best_combination : list 最优聚类编号组合
        best_residual : float 最小残差
        result : pd.Series 每条管道概率（非候选管道为0）
    """
    # === 1. 加载管道隶属度模板 ===
    with open(f"{File_PATH}/sensor_sort.pickle", "rb") as f:
        membership = pickle.load(f)  # DataFrame: 行=传感器，列=管道

    # === 2. 对齐传感器集合 ===
    sensor_values = sensor_values.loc[sensors]

    # === 3. 选出 top-p 传感器 ===
    top_p_sensors = list(set(sensor_values.nlargest(min(p, len(sensors))).index) & set(sensors))

    # === 4. 生成每条管道对应的聚类（取最大隶属传感器为聚类） ===
    pipe_clusters = pd.Series(index=membership.columns)
    for pipe in membership.columns:
        pipe_clusters.loc[pipe] = membership[pipe].idxmax()

    membership = membership.loc[sensors]
    # === 5. 聚类到管道映射 ===
    clusters = top_p_sensors
    cluster_map = {c: pipe_clusters[pipe_clusters==c].index.tolist() for c in clusters}

    # === 6. LASSO 残差函数 ===
    def lasso_residual(candidate_pipes):
        if len(candidate_pipes) == 0:
            return np.inf, pd.Series(0.0, index=pipe_clusters.index)
        v = sensor_values.values
        M = membership[candidate_pipes].values
        scaler_v = StandardScaler(with_std=True)
        scaler_M = StandardScaler(with_std=True)
        v_scaled = scaler_v.fit_transform(v.reshape(-1,1)).ravel()
        M_scaled = scaler_M.fit_transform(M)
        model = Lasso(alpha=lambda_, positive=False, max_iter=10000)
        model.fit(M_scaled, v_scaled)
        residual = np.linalg.norm(v_scaled - model.predict(M_scaled))
        w = np.abs(model.coef_)
        if w.sum() > 0:
            w /= w.sum()
        weights = pd.Series(0.0, index=pipe_clusters.index)
        weights.loc[candidate_pipes] = w
        return residual, weights

    # === 7. 遍历 k 个聚类组合 ===
    best_residual = np.inf
    best_combination = None
    best_weights = None

    for combo in itertools.combinations(clusters, leak_count):
        # 合并候选管道
        candidate_pipes = []
        for c in combo:
            candidate_pipes += cluster_map.get(c, [])
        residual, w = lasso_residual(candidate_pipes)
        if residual < best_residual:
            best_residual = residual
            best_combination = list(combo)
            best_weights = w

    # === 8. 返回结果 ===
    if best_weights is None:
        best_weights = pd.Series(0.0, index=pipe_clusters.index)
    return best_combination, [best_residual], best_weights.sort_values(ascending=False)

class Main:
    def __init__(self, year, select_sensors, fix_pressure, fis_params, leak_data, leak_demands=None):
        self.year = year
        self.select_sensors = select_sensors
        self.fix_pressure = fix_pressure
        self.fis_params = fis_params
        self.leak_demands = leak_demands

        self.wn = wntr.network.WaterNetworkModel(f"{DATA_PATH}/L-TOWN_v2_Model.inp")
        self.pipe_coords = get_coordinate(self.wn)
        self.leak_data = leak_data
        self.leak_events = pd.DataFrame(leak_data, columns=["linkID", "startTime", "endTime"])
        # 合并坐标
        self.leak_info = self.leak_events.join(self.pipe_coords, on="linkID")
        # 转换时间格式
        self.leak_info["startTime"] = pd.to_datetime(self.leak_info["startTime"]).dt.normalize()
        self.leak_info["endTime"] = pd.to_datetime(self.leak_info["endTime"]).dt.normalize()

    def detection(self):
        # === 1) 检测 ===
        print(f"（1）利用FIS进行泄漏识别")
        fis_result, day_result_step, self.pressure_result = run(self.fix_pressure, self.fis_params, sensors=self.select_sensors)
        self.leak_flag_times, self.detection_flag_times = collect_leak_detection_times(fis_result)
        pressure_detection = self.pressure_result.loc[list(self.detection_flag_times['time']), self.select_sensors].astype(float)

        max_leak = self.fis_params['max_leak']
        times = pressure_detection.index
        leak_range = list(range(1, max_leak + 1))

        # === 2) 计算所有 k 的结果（始终计算 1..5） ===
        print(f"（2）开始计算每个时间点的泄漏数量")
        self.residual_df = pd.DataFrame(index=times, columns=leak_range, dtype=float)
        self.weights_df = pd.DataFrame(index=self.pipe_coords.index, columns=pd.MultiIndex.from_product([times, leak_range]))
        self.combination_df = pd.DataFrame(index=times, columns=leak_range, dtype=object)

        for t in times:
            v = pressure_detection.loc[t, :]  # series of sensors
            for k in leak_range:
                combo, residual, weights = defuzzify_leak_probability_cluster(
                    sensor_values=v,
                    p=10,
                    sensors=self.select_sensors,
                    lambda_=self.fis_params['lasso_lambda'],
                    leak_count=k
                )
                if isinstance(residual, (list, tuple, np.ndarray)):
                    residual_val = float(residual[0])
                else:
                    residual_val = float(residual)
                self.residual_df.at[t, k] = residual_val
                self.weights_df.loc[:, (t, k)] = weights.reindex(self.pipe_coords.index).fillna(0).astype(float)
                self.combination_df.at[t, k] = combo

        # === 3) 根据 20% 阈值选每个时间的最优 k ===
        print(f"（3）根据阈值选择每个时间点的最优泄漏数量")
        threshold = self.fis_params['lasso_threshold']
        self.best_k_series = pd.Series(index=times, dtype=int)
        self.best_weights_df = pd.DataFrame(index=self.pipe_coords.index, columns=times, dtype=float)
        self.best_combination_series = pd.Series(index=times, dtype=object)

        for t in times:
            res_row = self.residual_df.loc[t].fillna(np.inf)
            chosen_k = max_leak
            for k in range(2, max_leak + 1):
                prev, curr = res_row[k - 1], res_row[k]
                if not np.isfinite(prev) or not np.isfinite(curr):
                    continue
                rel_drop = (prev - curr) / prev if prev != 0 else 0.0
                if rel_drop < threshold:
                    chosen_k = k - 1
                    break
            self.best_k_series[t] = chosen_k
            self.best_weights_df.loc[:, t] = self.weights_df.loc[:, (t, chosen_k)]
            self.best_combination_series[t] = self.combination_df.at[t, chosen_k]

        # === 4) 计算每个时间的泄漏重心 ===
        print(f"（4）计算每个时间点的泄漏重心")
        self.leakage_center = compute_leakage_center(max_leak=max_leak,
                                                     times=times,
                                                     pipe_coords=self.pipe_coords,
                                                     best_k_series=self.best_k_series,
                                                     best_weights_df=self.best_weights_df,
                                                     leak_info=self.leak_info)

        # === 5) 计算每个时间的检测结果 ===
        print(f"（5）计算每个时间点的检测结果")
        self.detection_correction, _ = generate_detection_correction(leakage_center=self.leakage_center,
                                                                  leak_info=self.leak_info,)

        # === 6) 计算经济指标 ===
        print(f"（6）计算经济指标")
        self.result = calculate_economic_value(detection_correction=self.detection_correction,
                                               leak_info=self.leak_info,
                                               leakage_center=self.leakage_center,
                                               leak_demands=self.leak_demands,
                                               )

if __name__ == "__main__":

    with open("./file/fix_pressure_2018.pickle", "rb") as f:
        fix_pressure_2018 = pickle.load(f)
    with open("./file/fix_pressure_2019.pickle", "rb") as f:
        fix_pressure_2019 = pickle.load(f)


    detection_2018_AB = Main(year=2018,
                            fix_pressure=fix_pressure_2018,
                            fis_params=init_fis_params('AB'),
                            leak_data=leak_2018_data,
                            select_sensors=DMA_AB_sensors)
    detection_2018_C = Main(year=2018,
                            fix_pressure=fix_pressure_2018,
                            fis_params=init_fis_params('C'),
                            leak_data=leak_2018_data,
                            select_sensors=DMA_C_sensors)
    detection_2019_AB = Main(year=2019,
                            fix_pressure=fix_pressure_2019,
                            fis_params=init_fis_params('AB'),
                            leak_data=leak_2019_data,
                            select_sensors=DMA_AB_sensors)
    detection_2019_C = Main(year=2019,
                            fix_pressure=fix_pressure_2019,
                            fis_params=init_fis_params('C'),
                            leak_data=leak_2019_data,
                            select_sensors=DMA_C_sensors)
