import numpy as np
import pandas as pd
import warnings
import scipy.linalg
from resampled import resampled
from transfor import transfor
warnings.filterwarnings("ignore")
class LMDWNG:
    def fit(self, X, y,n_nbor,distance):
            from sklearn.neighbors import kneighbors_graph, NearestNeighbors
            X,y=resampled(X,y)
            Xeu=np.array(X)
            ss=[]
            l3=[]
            ss = pd.to_numeric(y.iloc[:, 0])
            l=len(ss)
            for j, element in enumerate(ss):
                if element == 0:
                    l3.append(j)
            t=len(l3)
            print("多数类")
            print(t)
            print("少数类")
            print(l-t)
            l_n=l-t#l_n表示少数类数目
            neigh = NearestNeighbors(n_neighbors=n_nbor+1)
            neigh.fit(X)
            arr=neigh.kneighbors()[1]
            Data2=pd.DataFrame([])
            #print(arr)
            arr1=pd.DataFrame(arr)
            #arr1.to_csv(('222.csv'))
            #邻接矩阵
            distances, indices = neigh.kneighbors()
            # 构建邻接矩阵
            adjacency_matrix = np.zeros((len(X), len(X)))
            for i in range(len(X)):
                for j in indices[i][1:]:  # 第一个是自身，跳过
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1
            # print("邻接矩阵：")
            # print(adjacency_matrix)
            ad = pd.DataFrame(adjacency_matrix)
            # ad.to_csv('ad.csv')
            i=-1
            for x in arr:
                i=i+1
                for j in range(0,l):
                    Data2.loc[i, j] = 0
            i=-1
            for x in arr:
                i=i+1
                for z in x:
                    if distance=='Euclidean':
                        dist=np.sqrt(np.sum(np.square(Xeu[x,:]-Xeu[z,:])))
                        Data2.loc[i,z] = dist
                    elif distance == 'Chebyshev':
                        dist = np.abs(Xeu[x,:]-Xeu[z,:]).max()
                        Data2.loc[i, z] = dist
                    elif distance == 'Manhattan':
                        dist = np.sum(np.abs(Xeu[x,:] - Xeu[z,:]))
                        Data2.loc[i, z] = dist
            # print(Data2)
            # 度矩阵
            Data3=pd.DataFrame([])
            i=-1
            for x in arr:
                i=i+1
                for j in range(0,l):
                    Data3.loc[i, j] = 0
            ad.loc["按列求和"] =ad.apply(lambda x:x.sum())
            l2=[]
            l2=ad.loc["按列求和"]
            #print(l2)
            for i, element in enumerate(l2):
                Data3.loc[i,i]=element
            #print(Data3)
            #拉普拉斯矩阵的定义L = D-A 度矩阵-邻接矩阵
            Data_L=Data3-Data2

            for j in range(0,l):
                for i in range(0,l):
                    if(Data_L.loc[i, j] != 0):
                        Data_L.loc[j, i] = Data_L.loc[i, j]


            u,s=scipy.linalg.schur(Data_L)
            u=pd.DataFrame(u)
            # u.to_csv(('u.csv'))
            s=pd.DataFrame(s)
            # s.to_csv(('s.csv'))
            # q=pd.DataFrame(q)
            # q.to_csv(('q.csv'))
            # r=pd.DataFrame(r)
            # r.to_csv(('r.csv'))
            # print(s)
            # print(u)
            l1=[]
            for j in range(0,l):
                for i in range(0,l):
                    if(i==j):
                       l1.append(u.loc[i,j])
            l1=np.array(l1)
            #print(t)
            l_t=[]
            l_i=[]
            for i, element in enumerate(l1):
                if element>=0:
                    l_t.append(element)
                    l_i.append(i)
            #print("大于0的特征值共有%s个"%(len(l_i)))
            #print(len(l_i))
            #print(l_t)
            del1=list(zip(l_i,l_t))
            del1=np.array(del1)
            del1=del1[np.argsort(del1[:,1])]
            #print(del1)
            #print(del1[:,1])
            y=pd.DataFrame(y)
            X=pd.DataFrame(X)
            deldata=del1[:,0]
            #print(deldata)

            deldata2=[]
            for x in deldata:
                if x<t:
                    deldata2.append(x)
            #print(deldata2)
            lk = len(deldata2)  # lk表示要删除的多数类个数
            print("要删除的多数类个数为%s"%(lk))
            l_m=t-lk#l_m表示删除后剩余的多数类个数
            if l_m>=l_n:#剩余的多数类个数大于少数类
                for i in deldata2:
                    X.drop(index=i, inplace=True)
                    y.drop(index=i, inplace=True)
                #print(X)
                #print(y)
                ss2 = pd.to_numeric(y.iloc[:, 0])
                l2i = len(ss2)
                l4i=[]
                for j, element in enumerate(ss2):
                    if element == 0:
                        l4i.append(j)
                t1 = len(l4i)
                print("删除后多数类和少数类个数")
                print(t1)
                print(l2i-t1)
                return (X, y)
            else:#剩余的多数类小于少数类
                deldata2=deldata2[0:t-(l-t)]
                for i in deldata2:
                    X.drop(index=i, inplace=True)
                    y.drop(index=i, inplace=True)
                ss2 = pd.to_numeric(y.iloc[:, 0])
                l2i = len(ss2)
                l4i = []
                for j, element in enumerate(ss2):
                    if element == 0:
                        l4i.append(j)
                t1 = len(l4i)
                print("删除后多数类和少数类个数")
                print(t1)
                print(l2i - t1)
                return (X, y)