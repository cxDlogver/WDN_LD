""" 对每一个管道，模拟一个泄漏 """
import pandas as pd
import pickle
import os
import yaml
import wntr
import shutil
from math import sqrt
import numpy as np
from multiprocessing import Pool
import time

results_folder = "../Dataset Generator/Sensor Leak Dataset/"
conf_path = "../Dataset Generator/sensor_leak_configuration.yalm"

class Leak_Generatior:
    def __init__(self):
        # 导入管网配置
        with open(conf_path, "r") as f:
            wn_config = yaml.safe_load(f)
        self.start_time = pd.Timestamp(wn_config['times']['StartTime']) # 模拟开始时间
        self.end_time = pd.Timestamp(wn_config['times']['EndTime']) # 模拟结束时间
        self.leak_start_time = pd.Timestamp(wn_config['times']['LeakStartTime']) # 泄漏开始时间
        self.leak_end_time = pd.Timestamp(wn_config['times']['LeakEndTime']) # 泄漏结束时间
        self.leak_peak_time = pd.Timestamp(wn_config['times']['LeakPeakTime']) # 泄漏峰值时间
        self.leak_type = ['incipient']
        self.leak_diameter_range = [0.04]
        self.base_minute_step = 5 # 基础时间步长，单位：分钟
        self.Mode_Simulation = "PDD" # 模拟模式
        self.inp_file = wn_config['Network']['filename'] # 管网inp文件
        self.pressure_sensors = wn_config['pressure_sensors']
        self.amrs = wn_config['amrs']
        self.flow_sensors = wn_config['flow_sensors']
        self.level_sensors = wn_config['level_sensors']

    def init_wn(self):

        """ 初始化管网 """
        wn = wntr.network.WaterNetworkModel(self.inp_file)
        for name, node in wn.junctions():
            node.required_pressure = 25
        wn.options.time.hydraulic_timestep = self.base_minute_step * 60
        wn.options.time.pattern_timestep = self.base_minute_step * 60
        wn.options.time.report_timestep = self.base_minute_step * 60
        wn.options.time.quality_timestep = self.base_minute_step * 60
        wn.options.time.duration = (self.end_time - self.start_time).total_seconds()
        wn.options.hydraulic.demand_model = self.Mode_Simulation

        self.time_step = round(wn.options.time.hydraulic_timestep)
        self.time_stamp = pd.date_range(self.start_time, self.end_time, freq=f"{self.base_minute_step}min")
        self.TIMESTEPS = int(wn.options.time.duration / wn.options.time.hydraulic_timestep)
        return wn

    def create_folder(self, _path_):
        """ 创建文件夹 """
        try:
            if os.path.exists(_path_):
                shutil.rmtree(_path_)
            os.makedirs(_path_)
        except Exception as error:
            pass

    def generate_leak(self, pipe_name, type):
        leak_node = {}
        leak_diameter = {}
        leak_area = {}
        leak_type = {}
        leak_starts = {}
        leak_ends = {}
        leak_peak_time = {}
        leak_param = {}

        """ 生成泄漏 """

        wn = self.init_wn()
        ST = self.time_stamp.get_loc(self.leak_start_time) # 泄漏开始时间点
        ET = self.time_stamp.get_loc(self.leak_end_time) # 泄漏结束时间点
        PT = self.time_stamp.get_loc(self.leak_peak_time) # 泄漏峰值时间点
        node_leak = f"{pipe_name}_leaknode"
        wn = wntr.morph.split_pipe(wn, pipe_name, f'{pipe_name}_bleak', node_leak)

        leak_node[pipe_name] = wn.get_node(node_leak)
        leak_diameter[pipe_name] = np.random.choice(self.leak_diameter_range) # 泄漏直径
        leak_type[pipe_name] = type
        leak_area[pipe_name] = 3.14159 * (leak_diameter[pipe_name] / 2) ** 2
        print(f"generate {pipe_name} leak diameter: {leak_diameter[pipe_name]}")

        # 如果是 incipient 泄漏
        if type == "incipient":
            ET = ET + 1
            PT = PT + 1
            required_pressure = 100

            # incipient 泄漏逐步增加
            leak_param[pipe_name] = 'demand'
            leak_node[pipe_name].required_pressure = required_pressure
            increment_leak_diameter = leak_diameter[pipe_name] / (PT - ST)
            increment_leak_diameter = np.arange(increment_leak_diameter, leak_diameter[pipe_name], increment_leak_diameter)
            increment_leak_area = 0.75 * sqrt(2 / 1000) * 990.27 * 3.14159 * (increment_leak_diameter / 2) ** 2
            leak_magnitude = 0.75 * sqrt(2 / 1000) * 990.27 * leak_area[pipe_name]
            pattern_array = [0] * (ST) + increment_leak_area.tolist() + [leak_magnitude] * (ET - PT + 1) + [0] * (
            self.TIMESTEPS - ET)

            # basedemand
            leak_node[pipe_name].demand_timeseries_list[0]._base = 1
            pattern_name = f'{str(leak_node[pipe_name])}'
            wn.add_pattern(pattern_name, pattern_array)
            leak_node[pipe_name].demand_timeseries_list[0].pattern_name = pattern_name
            leak_node[pipe_name].required_pressure = required_pressure
            leak_node[pipe_name].minimum_pressure = 0

            # save times of leak
            leak_starts[pipe_name] = self.time_stamp[ST]
            leak_starts[pipe_name] = leak_starts[pipe_name]._date_repr + ' ' + leak_starts[pipe_name]._time_repr
            leak_ends[pipe_name] = self.time_stamp[ET - 1]
            leak_ends[pipe_name] = leak_ends[pipe_name]._date_repr + ' ' + leak_ends[pipe_name]._time_repr
            leak_peak_time[pipe_name] = self.time_stamp[PT - 1]._date_repr + ' ' + self.time_stamp[PT - 1]._time_repr

        # 如果是 abrupt
        else:
            leak_param[pipe_name] = 'leak_demand'
            PT = ST

            leak_node[pipe_name]._leak_end_control_name = str(pipe_name) + 'end'
            leak_node[pipe_name]._leak_start_control_name = str(pipe_name) + 'start'

            leak_node[pipe_name].add_leak(wn, discharge_coeff=0.75,
                                       area=leak_area[pipe_name],
                                       start_time=ST * self.time_step,
                                       end_time=(ET + 1) * self.time_step)

            leak_starts[pipe_name] = self.time_stamp[ST]
            leak_starts[pipe_name] = leak_starts[pipe_name]._date_repr + ' ' + leak_starts[pipe_name]._time_repr
            leak_ends[pipe_name] = self.time_stamp[ET]
            leak_ends[pipe_name] = leak_ends[pipe_name]._date_repr + ' ' + leak_ends[pipe_name]._time_repr
            leak_peak_time[pipe_name] = self.time_stamp[PT]._date_repr + ' ' + self.time_stamp[PT]._time_repr

        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
        if results.node["pressure"].empty:
            print("Negative pressures.")
            return -1

        leakage_folder = f"{results_folder}{pipe_name}/"
        self.create_folder(leakage_folder)  # 生成泄漏文件夹
            # Create leak XLS files
        decimal_size = 2
        if results:
            # Create CSV files
            for leak_i in leak_node.keys():
                if 'leaknode' in str(leak_node[leak_i]):
                    NODEID = str(leak_node[leak_i]).split('_')[0]
                totals_info = {'Description': ['Leak Pipe', 'Leak Area', 'Leak Diameter', 'Leak Type', 'Leak Start',
                                               'Leak End', 'Peak Time'],
                               'Value': [NODEID, str(leak_area[leak_i]), str(leak_diameter[leak_i]),
                                         leak_type[leak_i],
                                         str(leak_starts[leak_i]), str(leak_ends[leak_i]), str(leak_peak_time[leak_i])]}

                leaks = results.node[leak_param[leak_i]][str(leak_node[leak_i])].values
                # Convert m^3/s (wntr default units) to m^3/h
                # https://wntr.readthedocs.io/en/latest/units.html#epanet-unit-conventions
                leaks = [elem * 3600 for elem in leaks]
                leaks = [round(elem, decimal_size) for elem in leaks]
                leaks = leaks[:len(self.time_stamp)]

                total_Leaks = {'Timestamp': self.time_stamp}
                total_Leaks[NODEID] = leaks
                df1 = pd.DataFrame(totals_info)
                df2 = pd.DataFrame(total_Leaks)
                file_path = f'{leakage_folder}\\Leak_{NODEID}.xlsx'

                # 使用 with 语句来确保自动关闭
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    df1.to_excel(writer, sheet_name='Info', index=False)
                    df2.to_excel(writer, sheet_name='Demand (m3_h)', index=False)
                    # 不需要手动调用 save 或 close，with 块结束时会自动保存并关闭

            # Create xlsx file with Measurements
            total_pressures = {'Timestamp': self.time_stamp}
            total_demands = {'Timestamp': self.time_stamp}
            total_flows = {'Timestamp': self.time_stamp}
            total_levels = {'Timestamp': self.time_stamp}
            total_leak_demand = {'Timestamp': self.time_stamp}
            for j in range(0, wn.num_nodes):
                node_id = wn.node_name_list[j]
                if node_id in self.pressure_sensors:
                    pres = results.node['pressure'][node_id]
                    pres = pres[:len(self.time_stamp)]
                    pres = [round(elem, decimal_size) for elem in pres]
                    total_pressures[node_id] = pres

                if node_id in self.amrs:
                    dem = results.node['demand'][node_id]
                    dem = dem[:len(self.time_stamp)]
                    dem = [elem * 3600 * 1000 for elem in dem]  # CMH / L/s
                    dem = [round(elem, decimal_size) for elem in dem]  # CMH / L/s
                    total_demands[node_id] = dem

                if node_id in self.level_sensors:
                    level_pres = results.node['pressure'][node_id]
                    level_pres = level_pres[:len(self.time_stamp)]
                    level_pres = [round(elem, decimal_size) for elem in level_pres]
                    total_levels[node_id] = level_pres

            for leak_i in leak_node.keys():
                if 'leaknode' in str(leak_node[leak_i]):
                    if 'incipient' in leak_type[leak_i]:
                        node_id = str(leak_node[leak_i])
                        leak_demand = results.node['demand'][node_id]
                        leak_demand = leak_demand[:len(self.time_stamp)]
                        leak_demand = [elem * 3600 * 1000 for elem in leak_demand]  # CMH / L/s
                        leak_demand = [round(elem, decimal_size) for elem in leak_demand]  # CMH / L/s
                        total_leak_demand[node_id.split("_")[0]] = leak_demand
                    else:
                        node_id = str(leak_node[leak_i])
                        leak_demand = results.node['leak_demand'][node_id]
                        leak_demand = leak_demand[:len(self.time_stamp)]
                        leak_demand = [elem * 3600 * 1000 for elem in leak_demand]  # CMH / L/s
                        leak_demand = [round(elem, decimal_size) for elem in leak_demand]  # CMH / L/s
                        total_leak_demand[node_id.split("_")[0]] = leak_demand

            for j in range(0, wn.num_links):
                link_id = wn.link_name_list[j]

                if link_id not in self.flow_sensors:
                    continue
                flows = results.link['flowrate'][link_id]
                flows = [elem * 3600 for elem in flows]
                flows = [round(elem, decimal_size) for elem in flows]
                flows = flows[:len(self.time_stamp)]
                total_flows[link_id] = flows

            # Create a Pandas dataframe from the data.
            df1 = pd.DataFrame(total_pressures)
            df2 = pd.DataFrame(total_demands)
            df3 = pd.DataFrame(total_flows)
            df4 = pd.DataFrame(total_levels)
            df5 = pd.DataFrame(total_leak_demand)

            # 保存一份pickle
            with open(f'{leakage_folder}Measurements.pickle', 'wb') as f:
                pickle.dump({
                    'Pressures (m)': df1,
                    'Demands (L_h)': df2,
                    'Flows (m3_h)': df3,
                    'Levels (m)': df4,
                    'LeakDemands (L_h)': df5
                }, f)

            # Create a Pandas Excel writer using XlsxWriter as the engine.
            writer = pd.ExcelWriter(f'{leakage_folder}Measurements.xlsx', engine='xlsxwriter')

            # Convert the dataframe to an XlsxWriter Excel object.
            # Pressures (m), Demands (m^3/h), Flows (m^3/h), Levels (m)
            df1.to_excel(writer, sheet_name='Pressures (m)', index=False)
            df2.to_excel(writer, sheet_name='Demands (L_h)', index=False)
            df3.to_excel(writer, sheet_name='Flows (m3_h)', index=False)
            df4.to_excel(writer, sheet_name='Levels (m)', index=False)
            df5.to_excel(writer, sheet_name='LeakDemands (L_h)', index=False)

            # Close the Pandas Excel writer and output the Excel file.
            writer.close()
        else:
            print('Results empty.')
            return -1

def simulate_pipe(list_pipe, leak_generator):
    start = time.time()
    for pipe in list_pipe:
        st = time.time()
        leak_generator.generate_leak(pipe, 'incipient')
        print(f"incipient leak in {pipe} cost time: {time.time() - st}")
        # st = time.time()
        # leak_generator.generate_leak(pipe, 'abrupt')
        # print(f"abrupt leak in {pipe} cost time: {time.time() - st}")
        print("----------")

if __name__ == "__main__":
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    with open(conf_path, "r") as f:
        wn_config = yaml.safe_load(f)
    inp_file = wn_config['Network']['filename']
    wn = wntr.network.WaterNetworkModel(inp_file)

    # 传感器列表（按你提供）
    DMA_A_sensors = ['n54', 'n105', 'n114', 'n163', 'n188', 'n288', 'n296', 'n332', 'n342', 'n410', 'n415', 'n429',
                     'n458', 'n469', 'n495', 'n506', 'n516', 'n519', 'n549', 'n613', 'n636', 'n644', 'n679', 'n722',
                     'n726', 'n740', 'n752', 'n769', 'n229']
    DMA_B_sensors = ['n215']
    DMA_C_sensors = ['n1', 'n4', 'n31']

    all_sensors = DMA_A_sensors + DMA_B_sensors + DMA_C_sensors
    pipe_list = wn.pipe_name_list

    leak_pipe_list = []
    leak_node_list = []
    for pipe in pipe_list:
        if wn.get_link(pipe).start_node.name in all_sensors and wn.get_link(pipe).start_node.name not in leak_node_list:
            leak_pipe_list.append(pipe)
            leak_node_list.append(wn.get_link(pipe).start_node.name)
        elif wn.get_link(pipe).end_node.name in all_sensors and wn.get_link(pipe).end_node.name not in leak_node_list:
            leak_pipe_list.append(pipe)
            leak_node_list.append(wn.get_link(pipe).end_node.name)
    print(f"泄漏管道: {leak_pipe_list}")
    print(f"泄漏管道: {len(leak_pipe_list)}")

    saved_pipe_list = os.listdir(results_folder)
    no_saved_pipe_list = [pipe for pipe in leak_pipe_list if pipe not in saved_pipe_list]
    print(f"已保存的泄漏管道: {saved_pipe_list}")
    print(f"未保存的泄漏管道: {no_saved_pipe_list}")

    mapping = {}
    for item, pipe in enumerate(leak_pipe_list):
        mapping[leak_node_list[item]] = pipe
    import json5
    with open(f'{results_folder}leak_node_mapping.json5', 'w') as f:
        json5.dump(mapping, f, indent=4)
    #

    leak_generator = Leak_Generatior()
    simulate_pipe(no_saved_pipe_list, leak_generator)

    # print("初始化完成")
    # print("----------")
    #
    # # simulate_pipe(['p1'])
    #
    # results = []
    # for idx, sublist in enumerate(slice_pipes):
    #     print(f"Dispatch slice {idx}, size={len(sublist)}")
    #     res = pool.apply_async(simulate_pipe, args=(sublist, leak_generator))
    #     results.append(res)
    #
    # pool.close()
    # pool.join()
