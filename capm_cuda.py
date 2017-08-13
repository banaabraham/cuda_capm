# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:52:39 2017

@author: bana
"""
import pandas as pd
import urllib
import numpy as np
import threading
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit


dow=['goog','aapl','ge']

SP500 = pd.read_csv("GSPC.csv")
sp = list(SP500['Close'][:250])
sp_new = np.array(sp, dtype=np.float32)
sp_gpu = gpuarray.to_gpu(sp_new.astype(np.float32))
sp_gpu_avg = sum(sp_gpu)/len(sp_gpu)
X = [(sp_gpu[i])**2 for i in range(len(sp_gpu))]
sp_gpu_var = (sum(X)-(sp_gpu_avg**2))/len(sp_gpu)
sp_return = (sp_gpu[0]-sp_gpu[-1])/sp_gpu[-1]

rate = float(input("Input risk free rate (%): "))

def get_data(ticker,stock_dict):
    stock = ticker+".csv"
    try:
        d = pd.read_csv(stock)
    except:
        try:
            url="https://www.google.com/finance/historical?output=csv&q="+ticker
            urllib.request.urlretrieve(url,stock)
            d = pd.read_csv(stock)
        except Exception as e:
            print (e)
    stock_dict[ticker] = list(d['Close'][:250])        

stock_dict ={}
threads = []
for i in range(len(dow)):
    t = threading.Thread(target=get_data, args=(dow[i],stock_dict,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
    
def cuda_capm(arr):     
    X_new = np.array(arr, dtype=np.float32)
    arrc = gpuarray.to_gpu(X_new.astype(np.float32))
    #mth_return = [((arrc[i]-arrc[i-30])/arrc[i-30]).get() for i in range(len(arrc)-1,0,-30)]
    average = sum(arrc)/len(arrc)
    W = [(arrc[i]-average)*(sp_gpu[i]-sp_gpu_avg) for i in range(len(arrc))]
    cov = sum(W)/(len(arrc)-1)
    beta = cov/sp_gpu_var
    capm = rate + (sp_return-rate)*beta
    return capm.get()

hasil  = []
for i in stock_dict.values():
    hasil.append(cuda_capm(i))
print ("CAPM return: \n")

for i in range(len(dow)):
    print("%s : %.5f%%" %(dow[i],hasil[i]))    


   
    
    