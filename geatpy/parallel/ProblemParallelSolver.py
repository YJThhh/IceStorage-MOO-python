import numpy as np
class ProblemParallelSolver():
    def __init__(self,config):
        # 初始化参数


        # 调用父类构造方法完成实例化
        self.K = config["Q"].size
        self.Q = config["Q"]

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

    def COP(self,PLR):  # 定义COP计算公式
        # res = -12.018 * (PLR ** 2) + 20.627 * PLR
        res = 2
        return res


    def Price(self,t):  # 定义价格计算公式
        if 11 < t < 21:
            return 1.1
        else:
            return 0.5

    def P_Chiller(self, PLR, i):  # 冷机功率

        return (self.Q * PLR[:, self.K * i:self.K + self.K * i] / self.COP(
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
            W += self.P_Chiller(PLR, i) * self.Price(i)
        return W

    def get_PLR_and_ita(self, x):
        PLR = x[:, 0:self.K * 24]
        ita = np.zeros([x.shape[0], 24])

        for i in range(self.ita_num):
            ita[:, self.cooling_capacity_need_period[i]] = x[:, self.K * 24 + i]

        return PLR, ita