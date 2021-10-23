# -*- coding: utf-8 -*-

import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

import geatpy as ea
import numpy as np

from ProblemParallelSolver import ProblemParallelSolver
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, config, M=2):
        name = 'DTLZ1'  # 初始化name（函数名称，可以随意设置）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = config["K"] * 24 + config["ita_num"]  # 初始化Dim（决策变量维数）
        varTypes = np.array([0] * (config["K"] * 24 + config["ita_num"]))  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * (config["K"] * 24 + config["ita_num"])  # 决策变量下界
        ub = [1] * (config["K"] * 24 + config["ita_num"])  # 决策变量上界
        lbin = [1] * (config["K"] * 24 + config["ita_num"])  # 决策变量下边界 （0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * (config["K"] * 24 + config["ita_num"])  # 决策变量上边界 （0表示不包含该变量的下边界，1表示包含）
        self.config = config
        self.PoolType = config["PoolType"]
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(4)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小
        self.res=[]
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin,
                            ubin)



    def resCollecter(self,res):
        self.res.append(res)
        #print(res)
        pass
    def aimFunc(self, pop):  # 目标函数
        import datetime


        args = np.array_split(pop.Phen, 4)
        args = list(
            zip([self.config] * args.__len__(), args ))
        for i in range(args.__len__()):
            self.pool.apply_async(subAimFunc, args[i],callback=self.resCollecter)
        self.pool.close()  # 关闭进程池，不再接受新的进程
        self.pool.join()  # 主进程阻塞等待子进程的退出


        res = list(result.get())
        ObjV = res[0][0]
        CV = res[0][1]

        for point in res[1:]:
            ObjV = np.vstack([ObjV, point[0]])
            CV = np.vstack([CV, point[1]])


        pop.ObjV = ObjV
        pop.CV = CV

def subAimFunc(args):
        import datetime

        start_t = datetime.datetime.now()
        config=args[0]
        x = args[1]

        problemParallelSolver=ProblemParallelSolver(config)
        PLR, ita = problemParallelSolver.get_PLR_and_ita(x)

        f1 = problemParallelSolver.W_Chiller(PLR)  # 功耗
        f2 = problemParallelSolver.Price_Chiller(PLR)  # 费用

        g1 = (problemParallelSolver.W_Cooling_capacity(
            PLR) - problemParallelSolver.cooling_capacity_total) ** 2 - config["e"] # (冷机运行一天所产生的总制冷量-当天所需要的总制冷量)**2-e<0

        #g_cooling_capacity = []

        # for i in self.cooling_capacity_need_period:
        #     g_cooling_capacity.append(
        #         (self.cooling_capacity[i] * (1 - ita[ i]) - self.P_Cooling_capacity_chiller(PLR, i)) ** 2 - self.e
        #     )
        # g_cooling_capacity = np.array(g_cooling_capacity).transpose(1, 0)

        ObjV = np.hstack([f1.reshape([f1.size, 1]), f2.reshape([f1.size, 1])])

        CV = g1.reshape([f1.size, 1])
        print("单进程耗时：")
        end_t0 = datetime.datetime.now()
        print((end_t0 - start_t).microseconds)
        return [ObjV, CV]