import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import time

torch.manual_seed(42)
np.random.seed(42)
# 获取开始时间
start = time.perf_counter()
class CPADF:
    def __init__(self, nvec, R):
        self.nmod = len(nvec)# 长度为4，nevc是train数据集前四列，每列最大值+1的向量，为[12, 6, 12, 1461]
        self.R = R
        #self.tau = 1.0
        #initialize with  prior 
        self.mu = [torch.tensor(np.random.rand(nvec[j], self.R), dtype=torch.float32, requires_grad=False) for j in range(self.nmod)] #j=0,1,2,3
        self.v = [torch.tensor(np.ones([nvec[j], self.R]), dtype=torch.float32, requires_grad=False) for j in range(self.nmod)]

    #input is a vector
    def get_logz(self, mu, v, y): #进来的值是mu_n, v_n, y_n
        #nmod x R
        mu = mu.view(self.nmod, -1) #进来的值是mu_n(4*2的0~1随机值), v_n(4*2的1)
        v  =  v.view(self.nmod, -1)
        f_mean = mu[0,:]#mu取第一行
        mm2 = torch.outer(mu[0,:], mu[0,:]) + torch.diag(v[0,:]) #torch.outer算两个（mu取第一行）向量的外积，再+v取第一行值变对角矩阵
        for j in range(1, self.nmod):
            mm2 = mm2*(torch.outer(mu[j,:], mu[j,:]) + torch.diag(v[j,:])) #torch.outer算两个（mu取第2行）向量的外积，再+v取第2行值变对角矩阵（3，4行循环）
            f_mean = f_mean*mu[j,:] #mu取第2，3，4行分别乘以上面的f_mean
        f_mean = torch.sum(f_mean)# j=3时的f_mean元素相加    #这个是miu 
        f_v =  torch.sum(mm2) - f_mean**2 #j=3时mm2元素相加再减去（j=3时的f_mean元素相加）的平方
        y_v = f_v +self.tau #f_v+1                          #这个是v
        logZ = -0.5*torch.log(torch.tensor(2*np.pi)) - 0.5*torch.log(y_v) - 0.5/y_v*(y - f_mean)**2
       #logZ = -torch.lgamma(prior_a) + torch.logsumexp(log_weight + log_f_node,dim=0)
        return logZ
        #r=1 pass #0, 800001 points, rmse = 0.919666  #pass #9, 12000 points, rmse = 0.572033
        #r=2 pass #0, 12000 points, rmse = 0.53164   #pass #9, 12000 points, rmse = 0.409691
        #r=3                                          pass #9, 12000 points, rmse = 0.297
        #r=4                                          pass #9, 12000 points, rmse = 0.605672 中间炸了一次，前面到过0.301
        #r=5 nan
    def go_through(self, ind, y, test_ind, test_y):
        #let fix tau for convenience 
        self.tau = torch.tensor(np.var(y))
        N = ind.shape[0]
        for npass in range(10):
            for n in range(N):
                ind_n = ind[n,:]
                y_n = y[n]
                #nmod * R X 1   4*2*1
                mu_n = torch.hstack([self.mu[j][ ind_n[j] ] for j in range(self.nmod)]).clone().detach().requires_grad_(True)
                                     #self.mu[j][ind_n[j]]是第j个模块的输出张量`self.mu[j]`中选择由`ind_n[j]`索引的子集
                                     #上下两个都是1*8的，mu_n是1*8的0~1随机数，v_n是1*8的1
                v_n  = torch.hstack([self. v[j][ ind_n[j] ] for j in range(self.nmod)]).clone().detach().requires_grad_(True)
                #clone()创建副本张量，使操作不影响原始张量。detach()将张量从计算图分离，使其成为独立的张量，不会进行梯度计算。
                logZ = self.get_logz(mu_n, v_n, y_n)
                dmu = torch.autograd.grad(logZ, mu_n, create_graph=True)[0] #计算关于logZ(标量)的梯度
                # create_graph参数为True表示要为计算创建一个计算图，以便可以对该图再次计算梯度
                # 最终得到的dmu是关于mu_n的梯度值
                dv = torch.autograd.grad(logZ, v_n, create_graph=True)[0]
                mu_star = mu_n + v_n*dmu
                v_star = v_n - (v_n**2)*(dmu**2 -2*dv)#这两行就是(9)式
                mu_star = mu_star.view(self.nmod, -1)
                v_star = v_star.view(self.nmod, -1)
                for j in range(self.nmod):#（更新self.mu和self.v）
                    #self.mu的第j个元素的第ind_n[j]个子元素 =（改成） mu_star第j行
                    self.mu[j][ ind_n[j] ] = mu_star[j,:].clone().detach()
                    self.v[j][ ind_n[j] ] = v_star[j,:].clone().detach()
                if n%1000 == 0 or n==N-1:   #100,...,12000出rmse
                    pred = self.pred(test_ind, test_y) #调用下面的pred函数
                    rmse = torch.sqrt(torch.mean( torch.square(pred - test_y) )) # 计算预测结果pred和真实值test_y之间的均方根误差(RMSE)
                    # 获取结束时间
                    end = time.perf_counter()
                    # 计算运行时间
                    runTime = end - start
                    runTime_ms = runTime * 1000
                    print('pass #%d, %d points, rmse = %g'%(npass, n+1, rmse),"运行时间：", runTime, "秒")

    '''对测试集数据进行预测'''
    #pred`方法根据输入的测试集数据`test_ind`和训练好的模型参数`self.mu`，利用模型进行预测。
    #通过遍历特征索引的列，逐步更新`pred`并得到最终的预测结果。最后将预测结果返回，完成对测试集数据的预测。
    #这些操作基于学术研究和理论，采用了概率因子分解方法，通过模型参数和特征对目标值进行预测。
    def pred(self, test_ind, test_y):
        pred = self.mu[0][test_ind[:,0]] #test_ind[:,0]表示取出test_ind的第一列
        #根据test_ind的第一列索引，来取出self.mu的第一行或者第一组行，并将它们赋给pred
        for j in range(1,self.nmod): #self.nmod=4, j=1~3  上面拿完第一列了，这里拿2~4列
            pred = pred*self.mu[j][test_ind[:,j]]
            #根据test_ind中的每一列索引，从self.mu中取出对应的元素或者一组元素，并与pred相乘，更新pred
        pred = torch.sum(pred, 1)#对pred进行按行求和，torch.sum()函数的第二个参数1示按行求和
        return pred 

def test_beijing():
    ind = []
    y = []
    with open('./data/beijing/beijing_15k_train.txt','r') as f: #以只读模式打开，并将文件对象赋值给变量`f`  with`语句在代码块执行完毕后关闭文件
            for line in f: #遍历文件中的每一行
                items = line.strip().split(',') #strip()对当前行去除首尾空白字符（包括换行符），split(',')按逗号分割成一个列表
                #例如，如果`line`为`"apple,banana,orange"`，那么`items`将变为`["apple",  "banana",  "orange"]`
                y.append(float(items[-1])) #append将当前行中的最后一个元素（目标值）（转换为浮点数）添加到列表`y`中
                ind.append([int(idx) for idx in items[0:-1]]) #将当前行中除最后一个元素外的所有元素（特征向量）转换为整数添加到列表`ind`中
                                #后面部分使用列表推导式，idx遍历items[0:-1]中的每个元素，并将其转换为整数，items[0:-1]获取除最后一个元素外所有元素
            ind = np.array(ind) #将列表`ind`转换为NumPy数组
            y = np.array(y)     #将列表`y`  转换为NumPy数组
    nvec = np.max(ind, 0) + 1# max是选数组中最大值    0是指：沿着数组第一个轴（行）操作
    #这样nvec就包含了每一列的最大值加1的结果
    #若ind = [[5, 2, 4, 0],
            #[3, 2, 6, 0],
            #[5, 2, 0, 0]]，算完这一行先得到[5, 2, 6, 0]（每一列的max值），再加1得到[6, 3, 7, 1]
    test_ind = []
    test_y = []
    with open('./data/beijing/beijing_15k_test.txt','r') as f:
                for line in f:  #操作同上
                    items = line.strip().split(',')
                    test_y.append(float(items[-1]))
                    test_ind.append([int(idx) for idx in items[0:-1]])
                test_ind = np.array(test_ind)
                test_y = np.array(test_y)

    R = 5
    model = CPADF(nvec, R)
    model.go_through(ind, y, test_ind, test_y)

if __name__ == '__main__': #当前文件被直接运行时，执行下面的代码；而当该文件作为模块被导入到其他文件中时，不执行下面的代码。
    test_beijing()         #在当前文件被直接运行时，执行`test_beijing()`函数

