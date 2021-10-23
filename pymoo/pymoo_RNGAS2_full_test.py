import numpy as np
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.rnsga2 import RNSGA2
from pymoo.factory import get_problem
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt


class MyProblem(Problem):

    def __init__(self, config):

        super().__init__(
            n_var=config["K"] * 24 + config["ita_num"],
            # ita不在蓄冰的start到end里面的每个小时
            n_obj=2,
            n_constr=1 + config["ita_num"] * 1,
            xl=np.array([0 for i in range(config["Q"].size * 24 + config["ita_num"])]),
            xu=np.array([1 for i in range(config["Q"].size * 24 + config["ita_num"])]),
            elementwise_evaluation=True)
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

    def P_Chiller(self, PLR, i):  # 冷机功率
        return (self.Q * PLR[self.K * i:self.K + self.K * i] / self.func_COP(
            PLR[self.K * i:self.K + self.K * i])).sum()

    def P_Cooling_capacity_chiller(self, PLR, i):  # 某时刻制冷量的功率总和
        return (self.Q * PLR[self.K * i:self.K + self.K * i]).sum()

    def W_Cooling_capacity(self, PLR):  # 总的有效制冷量
        W = 0

        def Is_save_ice(self, i):  # 可优化速度
            if self.start > self.end:
                if self.end > i >= 0:
                    return True
                if 24 > i >= self.start:
                    return True
                else:
                    return False
            else:
                if self.end > i >= self.start:
                    return True
                else:
                    return False

        for i in range(24):#
            if Is_save_ice(self, i):
                W += self.P_Cooling_capacity_chiller(PLR, i) * self.delta
            else:
                W += self.P_Cooling_capacity_chiller(PLR, i)
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

    def Q_ice(self, PLR):  # 冰槽总蓄冰量
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
        PLR = x[0:self.K * 24]
        ita = [0 for i in range(24)]
        if self.end < self.start:
            for i in range(self.end, self.start):
                ita[i] = x[self.K * 24 + i - self.end]
        else:
            for i in range(self.end, 24):
                ita[i] = x[self.K * 24 + i - self.end]
            for i in range(0, self.start):
                ita[i] = x[self.K * 24 + (24 - self.end) + i]
        return PLR, ita

    def plan_printer(self, x):
        PLR, ita = self.get_PLR_and_ita(x)

        f1 = self.W_Chiller(PLR)  # 功耗
        f2 = self.Price_Chiller(PLR)  # 费用

        g1 = (self.W_Cooling_capacity(
            PLR) - self.cooling_capacity_total) ** 2 - self.e  # (冷机运行一天所产生的总制冷量-当天所需要的总制冷量)**2-e<0

        g_max_deicing_capacity_ice = []
        g_cooling_capacity = []
        if self.start > self.end:  # TODO 重复代码段，此处代码可以简洁
            for i in range(self.end, self.start):
                g_max_deicing_capacity_ice.append(
                    self.cooling_capacity[i] * ita[i] - self.Max_deicing_capacity_ice(PLR, ita, i)
                )
                g_cooling_capacity.append(
                    (self.cooling_capacity[i] * (1 - ita[i]) - self.P_Cooling_capacity_chiller(PLR,
                                                                                               i)) ** 2 - self.e
                )
        else:
            for i in range(0, self.start):
                g_max_deicing_capacity_ice.append(
                    self.cooling_capacity[i] * ita[i] - self.Max_deicing_capacity_ice(PLR, ita, i))
                g_cooling_capacity.append(
                    (self.cooling_capacity[i] * (1 - ita[i]) - self.P_Cooling_capacity_chiller(PLR,
                                                                                               i)) ** 2 - self.e
                )
            for i in range(self.end, 24):
                g_max_deicing_capacity_ice.append(
                    self.cooling_capacity[i] * ita[i] - self.Max_deicing_capacity_ice(PLR, ita, i))
                g_cooling_capacity.append(
                    (self.cooling_capacity[i] * (1 - ita[i]) - self.P_Cooling_capacity_chiller(PLR,
                                                                                               i)) ** 2 - self.e
                )
        print("*************************** plan begin ******************************")
        print("功耗:{f1} W ,费用:{f2} 元".format(f1=f1, f2=f2))
        print("总制冷量:{w_Cooling_capacity} W,冷负荷需要总量:{cooling_capacity_total} W, 总蓄冷量:{Q_ice} W".format(
            w_Cooling_capacity=self.W_Cooling_capacity(PLR), cooling_capacity_total=self.cooling_capacity_total,
            Q_ice=self.Q_ice(PLR)))
        print("蓄冰起始时间:{start}:00,蓄冰结束时间:{end}:00 ,蓄冰工作时长:{working_hour} H".format(start=self.start, end=self.end,
                                                                                  working_hour=24 - self.ita_num))
        print("每小时情况：")
        for i in range(24):
            if self.start > self.end:
                if i < self.end or i >= self.start:  # 在蓄冰
                    print("时间:{t},状态:在蓄冰 ,冷机PLR:{PLR}, ".format(t=i, PLR=PLR[i]))
                else:
                    print("时间:{t},状态:冰槽供冷 ,"
                          "冷负荷需求:{cooling_capacity} W, "
                          "冰槽供冷占比:{ita}, "
                          "冰槽最大可供冷量:{g_max_deicing_capacity_ice} W, "
                          "冰槽供冷量:{P_cooling_capacity_ice} W, "
                          "所有冷机供冷量{P_Cooling_capacity_chiller} W"
                        .format(
                        t=i,
                        cooling_capacity=self.cooling_capacity[i],
                        ita=ita[i],
                        g_max_deicing_capacity_ice=self.Max_deicing_capacity_ice(PLR, ita, i),
                        P_cooling_capacity_ice=ita[i] * self.cooling_capacity[i],
                        P_Cooling_capacity_chiller=self.P_Cooling_capacity_chiller(PLR, i)
                    ))
            else:
                if self.start <= i < self.end:  # 在蓄冰
                    print("时间:{t},状态:在蓄冰 ,冷机PLR:{PLR}, ".format(t=i, PLR=PLR[i]))
                else:
                    print("时间:{t},状态:冰槽供冷 ,"
                          "冷负荷需求:{cooling_capacity} W, "
                          "冰槽供冷占比:{ita}, "
                          "冰槽最大可供冷量:{g_max_deicing_capacity_ice} W, "
                          "冰槽供冷量:{P_cooling_capacity_ice} W, "
                          "所有冷机供冷量{P_Cooling_capacity_chiller} W"
                        .format(
                        t=i,
                        cooling_capacity=self.cooling_capacity[i],
                        ita=ita[i],
                        g_max_deicing_capacity_ice=self.Max_deicing_capacity_ice(PLR, ita, i),
                        P_cooling_capacity_ice=ita[i] * self.cooling_capacity[i],
                        P_Cooling_capacity_chiller=self.P_Cooling_capacity_chiller(PLR, i)
                    ))
        print("*************************** plan end ******************************")

    def plan_optimization(self, x):
        x = x.tolist()
        PLR, ita = self.get_PLR_and_ita(x)
        if self.start > self.end:
            for i in range(self.end, self.start):
                if self.cooling_capacity[i] < 10:
                    x[x.index(ita[i])] = 0
        else:
            for i in range(0, self.start):
                if self.cooling_capacity[i] < 10:
                    x[x.index(ita[i])] = 0
            for i in range(self.end, 24):
                if self.cooling_capacity[i] < 10:
                    x[x.index(ita[i])] = 0
        return np.array(x)

    def _evaluate(self, x, out, *args, **kwargs):

        PLR, ita = self.get_PLR_and_ita(x)

        f1 = self.W_Chiller(PLR)  # 功耗
        f2 = self.Price_Chiller(PLR)  # 费用

        g1 = (self.W_Cooling_capacity(
            PLR) - self.cooling_capacity_total) ** 2 - self.e  # (冷机运行一天所产生的总制冷量-当天所需要的总制冷量)**2-e<0

        g_max_deicing_capacity_ice = []
        g_cooling_capacity = []
        if self.start > self.end:  # TODO 重复代码段，此处代码可以简洁
            for i in range(self.end, self.start):
                # g_max_deicing_capacity_ice.append(
                #     self.cooling_capacity[i] * ita[i] - self.Max_deicing_capacity_ice(PLR, ita, i)
                # )
                g_cooling_capacity.append(
                    (self.cooling_capacity[i] * (1 - ita[i]) - self.P_Cooling_capacity_chiller(PLR,
                                                                                               i)) ** 2 - self.e
                )
        else:
            for i in range(0, self.start):
                # g_max_deicing_capacity_ice.append(
                #     self.cooling_capacity[i] * ita[i] - self.Max_deicing_capacity_ice(PLR, ita, i))
                g_cooling_capacity.append(
                    (self.cooling_capacity[i] * (1 - ita[i]) - self.P_Cooling_capacity_chiller(PLR,
                                                                                               i)) ** 2 - self.e
                )
            for i in range(self.end, 24):
                # g_max_deicing_capacity_ice.append(
                #     self.cooling_capacity[i] * ita[i] - self.Max_deicing_capacity_ice(PLR, ita, i))
                g_cooling_capacity.append(
                    (self.cooling_capacity[i] * (1 - ita[i]) - self.P_Cooling_capacity_chiller(PLR, i)) ** 2 - self.e
                )

        out["F"] = [f1, f2]
        out["G"] = [g1] + g_cooling_capacity  # + g_max_deicing_capacity_ice


# 初始化参数
def COP(PLR):  # 定义COP计算公式
    # res = -12.018 * (PLR ** 2) + 20.627 * PLR
    res = 2
    return res


def Price(t):  # 定义价格计算公式
    if 11 < t < 21:
        return 1.1
    else:
        return 0.5


def Config_process(config):
    config["ita_num"] = (config["start"] - config["end"]) if config["start"] > config["end"] else (
            24 - (config["end"] - config["start"]))
    config["K"]=config["Q"].size
    return config


def Visualization(res, problem):
    problem.plan_printer(res.X[0])

    # X_1 = problem.plan_optimization(res.X[0])  # 这个地方要处理一下plan，不能直接画图
    # X_2 = problem.plan_optimization(res.X[-1])
    #
    # start = problem.start
    # end = problem.end
    # K = problem.K
    # ita_1 = [0 for i in range(24)]
    # ita_2 = [0 for i in range(24)]
    # if end < start:
    #     for i in range(end, start):
    #         ita_1[i] = X_1[K * 24 + i - end]
    #         ita_2[i] = X_2[K * 24 + i - end]
    # else:
    #     for i in range(end, 24):
    #         ita_1[i] = X_1[K * 24 + i - end]
    #         ita_2[i] = X_2[K * 24 + i - end]
    #     for i in range(0, start):
    #         ita_1[i] = X_1[K * 24 + (24 - end) + i]
    #         ita_2[i] = X_2[K * 24 + (24 - end) + i]
    #
    # PLR_1 = X_1[0:24 * K].reshape(K, -1)
    # PLR_1 = np.array(PLR_1)
    # ita_1 = np.array(ita_1)
    #
    # PLR_2 = X_2[0:24 * K].reshape(K, -1)
    # PLR_2 = np.array(PLR_2)
    # ita_2 = np.array(ita_2)
    #
    # t = np.arange(0, 24, 1)
    #
    # plt.figure(1)
    # # 第一行第一列图形
    # ax1 = plt.subplot(3, 2, 1)
    # # 第一行第二列图形
    # ax2 = plt.subplot(3, 2, 2)
    #
    # # 第二行第一列图形
    # ax3 = plt.subplot(3, 2, 3)
    # # 第二行第二列图形
    # ax4 = plt.subplot(3, 2, 4)
    #
    # # 第三行
    # ax5 = plt.subplot(3, 2, 5)
    #
    # plt.sca(ax1)
    # plt.plot(t, PLR_1[0], 'r--')
    # plt.title('PLR')
    #
    # plt.sca(ax2)
    # plt.plot(t, ita_1, 'r--')
    # plt.title('ita')
    #
    # plt.sca(ax3)
    # plt.plot(t, PLR_2[0], 'r--')
    # plt.title('PLR')
    #
    # plt.sca(ax4)
    # plt.plot(t, ita_2, 'r--')
    # plt.title('ita')
    #
    # plt.sca(ax5)
    # X = res.F[:, 0]
    # Y = res.F[:, 1]
    # plt.scatter(X, Y, color='red')
    # plt.title('F')
    # plt.show()


config = {}
config["Q"] = np.array([250000])  # 冷机额定功率
config["e"] = e = 10000  # 等式约束的允许误差
config["cooling_capacity"] = np.array([0, 0, 0, 0, 0, 0,  # 0-5
                                       0, 0, 0, 3, 10, 15,  # 6-11
                                       20, 22, 23, 23, 22, 20,  # 12-17
                                       18, 16, 12, 8, 5, 0]) * 1000  # 16-23 当天每小时所需要的冷量 单位：kw
config["delta"] = 0.94  # 蓄冰转化率
config["start"] = 0
config["end"] = 7  # [start,end)
config["h1"] = 15687
config["h2"] = 0.537
config["Func_COP"] = COP
config["Func_Price"] = Price
config = Config_process(config)
ref_points = np.array([[110000, 55000]])
problem = MyProblem(config=config)

algorithm = RNSGA2(
    ref_points=ref_points,
    pop_size=40,
    epsilon=0.01,
    normalization='front',
    extreme_points_as_reference_points=False,
    weights=np.array([0.5, 0.5]))
res = []
for i in range(5):
    res.append(minimize(problem,
                        algorithm,
                        ("n_gen", 2000),
                        verbose=True,
                        seed=i * 25))
for i in range(5):
    Visualization(res[i], problem)
