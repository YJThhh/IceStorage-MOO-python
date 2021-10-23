# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, config, M=2):
        name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = config["K"] * 24 + config["ita_num"]  # 初始化Dim（决策变量维数）
        varTypes = np.array([0] * (config["K"] * 24 + config["ita_num"]))  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * (config["K"] * 24 + config["ita_num"])  # 决策变量下界
        ub = [1] * (config["K"] * 24 + config["ita_num"])  # 决策变量上界
        lbin = [1] * (config["K"] * 24 + config["ita_num"])  # 决策变量下边界
        ubin = [1] * (config["K"] * 24 + config["ita_num"])  # 决策变量上边界

        # 调用父类构造方法完成实例化
        self.K = config["Q"].size
        self.Q = config["Q"]
        self.func_Price = config["Func_Price"]
        self.func_COP = config["Func_COP"]
        self.cooling_capacity = config["cooling_capacity"]
        self.e = config["e"]
        self.delta = config["delta"]
        self.start = config["start"]
        self.end = config["end"]
        self.h1 = config["h1"]
        self.h2 = config["h2"]
        self.ita_num = config["ita_num"]

        self.cooling_capacity_total = self.cooling_capacity.sum()
        self.ice_save_period = config["ice_save_period"]
        self.non_ice_save_period = config["non_ice_save_period"]
        self.cooling_capacity_need_period = config["cooling_capacity_need_period"]
        self.itera=0
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin,
                            ubin)

    def P_Chiller(self, PLR, i):  # 冷机功率

        return (self.Q * PLR[:, self.K * i:self.K + self.K * i] / self.func_COP(
            PLR[:, self.K * i:self.K + self.K * i])).sum(axis=1)

    def P_Cooling_capacity_chiller(self, PLR, i):  # 某时刻制冷量的功率总和
        return (self.Q * PLR[:, self.K * i:self.K + self.K * i]).sum(axis=1)

    def W_Cooling_capacity(self, PLR):  # 制冷量
        W = 0
        for i in self.non_ice_save_period:
            W += self.P_Cooling_capacity_chiller(PLR, i)

        for i in self.ice_save_period:
            W += self.P_Cooling_capacity_chiller(PLR, i) * self.delta

        return W

    def W_Chiller(self, PLR):  # 冷机功耗
        W = 0
        for i in range(24):
            W += self.P_Chiller(PLR, i)
        return W

    def Price_Chiller(self, PLR):  # 冷机耗电费用
        W = 0
        for i in range(24):
            W += self.P_Chiller(PLR, i) * self.func_Price(i)
        return W

    def Q_ice(self, PLR):  # 冰槽总蓄冰量转化为蓄冷量
        W = 0
        # assert self.start<=24 & self.end>=0,"蓄冰时间设置错误"
        if self.start > self.end:
            for i in range(self.start, 24):
                W += self.P_Cooling_capacity_chiller(PLR, i) * self.delta
            for i in range(0, self.end):
                W += self.P_Cooling_capacity_chiller(PLR, i) * self.delta
        else:
            for i in range(self.start, self.end):
                W += self.P_Cooling_capacity_chiller(PLR, i) * self.delta
        return W

    def Max_deicing_capacity_ice(self, PLR, ita, t):  # 当前时刻冰槽最大融冰供冷量
        Q_tank = 0
        if self.start > self.end:
            for i in range(self.end, t):
                Q_tank += self.cooling_capacity[i] * ita[i]
        else:
            if t > self.end:
                for i in range(self.end, t):
                    Q_tank += self.cooling_capacity[i] * ita[i]
            else:
                for i in range(self.end, 24):
                    Q_tank += self.cooling_capacity[i] * ita[i]
                for i in range(0, t):
                    Q_tank += self.cooling_capacity[i] * ita[i]
        return self.h1 + self.h2 * (self.Q_ice(PLR) - Q_tank)

    def get_PLR_and_ita(self, x):
        PLR = x[:, 0:self.K * 24]
        ita = np.zeros([x.shape[0], 24])

        for i in range(self.ita_num):
            ita[:, self.cooling_capacity_need_period[i]] = x[:, self.K * 24 + i]

        return PLR, ita

    def aimFunc(self, pop):  # 目标函数
        print(self.itera)
        #import datetime

        #start_t = datetime.datetime.now()
        self.itera+=1
        x = pop.Phen
        PLR, ita = self.get_PLR_and_ita(x)

        f1 = self.W_Chiller(PLR)  # 功耗
        f2 = self.Price_Chiller(PLR)  # 费用

        g1 = (self.W_Cooling_capacity(
            PLR) - self.cooling_capacity_total) ** 2 - self.e  # (冷机运行一天所产生的总制冷量-当天所需要的总制冷量)**2-e<0
        g_cooling_capacity = []  #约束2
        # g_max_deicing_capacity_ice = []  #约束3


        for i in self.cooling_capacity_need_period:

            g_cooling_capacity.append(
                (self.cooling_capacity[i] * (1 - ita[:,i]) - self.P_Cooling_capacity_chiller(PLR,
                                                                                           i)) ** 2 - self.e
            )

            # g_max_deicing_capacity_ice.append(
            #     self.cooling_capacity[i] * ita[:,i] - self.Max_deicing_capacity_ice(PLR, ita[:,i], i))

        g_cooling_capacity=np.array(g_cooling_capacity).transpose()
        #g_max_deicing_capacity_ice=np.array(g_max_deicing_capacity_ice).transpose()
        pop.ObjV = np.hstack([f1.reshape([f1.size, 1]), f2.reshape([f1.size, 1])])

        pop.CV = np.hstack([g1.reshape([f1.size, 1]),g_cooling_capacity])#,g_max_deicing_capacity_ice])
        #print("单核耗时：")
        #end_t0 = datetime.datetime.now()
        #print((end_t0 - start_t).microseconds)