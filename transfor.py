import numpy as np
import pandas as pd
def transfor(y):
    ss=list(y)
    s0=[]
    s1=[]
    s2=[]
    s_1=[]
    l=[]
    ss0=[]
    ss1=[]
    for i, element in enumerate(ss):
        if element==0:
            s0.append(i)
        if element==1:
            s1.append(i)
        if element==2:
            s2.append(i)
        if element==-1:
            s_1.append(i)
    a=0
    for sx in (s0,s1,s2,s_1):
        if len(sx)>a:
            ss0=sx
            a=len(sx)
    b=0
    for sx2 in (s0,s1,s2,s_1):
        if len(sx2)>b and len(sx2)!=len(ss0):
            ss1=sx2
            b=len(sx2)
    for i in ss0:
        ss[i]=0
    for j in ss1:
        ss[j]=1
    y_t=np.array(ss)
    return y_t