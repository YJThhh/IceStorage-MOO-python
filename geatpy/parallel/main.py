# -*- coding: utf-8 -*-
import geatpy as ea  # import geatpy
import numpy as np

from MyProblem import MyProblem  # 导入自定义问题接口

if __name__ == '__main__':



    def Config_process(config):
        def get_period(config):  # 获得各种期间
            ice_save_period = []  # 蓄冰期间
            non_ice_save_period = []  # 非蓄冰期间
            cooling_capacity_need_period = []  # 有冷负荷期间
            non_cooling_capacity_need_period = []  # 无冷负荷期间
            if config["end"] < config["start"]:
                for i in range(config["end"], config["start"]):
                    non_ice_save_period.append(i)

                    if config["cooling_capacity"][i] > 10:
                        cooling_capacity_need_period.append(i)
                    else:
                        non_cooling_capacity_need_period.append(i)

                for i in range(config["start"], 24):
                    ice_save_period.append(i)
                for i in range(0, config["end"]):
                    ice_save_period.append(i)


            else:
                for i in range(config["end"], 24):
                    non_ice_save_period.append(i)

                    if config["cooling_capacity"][i] > 10:
                        cooling_capacity_need_period.append(i)
                    else:
                        non_cooling_capacity_need_period.append(i)

                for i in range(0, config["start"]):
                    non_ice_save_period.append(i)

                    if config["cooling_capacity"][i] > 10:
                        cooling_capacity_need_period.append(i)
                    else:
                        non_cooling_capacity_need_period.append(i)

                for i in range(config["start"], config["end"]):
                    ice_save_period.append(i)





            return ice_save_period, non_ice_save_period, cooling_capacity_need_period

        config["ice_save_period"], config["non_ice_save_period"], config["cooling_capacity_need_period"] = get_period(config)
        config["ita_num"] = config["cooling_capacity_need_period"].__len__()
        config["K"] = config["Q"].size

        return config


    config = {}
    config["PoolType"] = 'Process'  # Thread 设置采用多线程，若修改为: PoolType = 'Process'，则表示用多进程
    config["Q"] = np.array([250000])  # 冷机额定功率
    config["e"] = e = 10000  # 等式约束的允许误差
    config["cooling_capacity"] = np.array([0, 0, 0, 0, 0, 0,  # 0-5
                                           0, 0, 0, 3, 10, 15,  # 6-11
                                           20, 22, 23, 23, 22, 20,  # 12-17
                                           18, 16, 12, 8, 5, 0]) * 1000  # 16-23 当天每小时所需要的冷量 单位：kw
    config["delta"] = 1  # 蓄冰转化率
    config["start"] = 0
    config["end"] = 7  # [start,end)
    config["h1"] = 15687
    config["h2"] = 0.537

    config = Config_process(config)

    """===============================实例化问题对象============================"""
    problem = MyProblem(config)  # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'  # 编码方式
    NIND = 500 # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders,
                      [10] * len(problem.varTypes))  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.6  # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 4000  # 最大进化代数
    myAlgorithm.logTras = 0  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
