import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Vepp_lib import freq_lib as fl

def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array

number1=1
number2=2
number3=3

number4=2
number5=3
number6=4
size=5000
############################################################################################################
"импортирование эксперементальных данных и подбор 1-го оборота"


dataExp1=np.load('../data_Vepp/tbt_2021_12_06_19_01_04.npy','r')
dataExp2=np.load('../data_Vepp/tbt_2021_12_06_19_02_36.npy','r')
dataExp3=np.load('../data_Vepp/tbt_2021_12_06_19_03_49.npy','r')
dataExp4=np.load('../data_Vepp/tbt_2021_12_06_19_04_46.npy','r')
dataExp5=np.load('../data_Vepp/tbt_2021_12_06_19_05_30.npy','r')
dataExp6=np.load('../data_Vepp/tbt_2021_12_06_19_06_31.npy','r')
dataExp7=np.load('../data_Vepp/tbt_2021_12_06_19_08_23.npy','r')
dataExp8=np.load('../data_Vepp/tbt_2021_12_06_19_09_10.npy','r')
dataExp9=np.load('../data_Vepp/tbt_2021_12_06_19_09_59.npy','r')
dataExp10=np.load('../data_Vepp/tbt_2021_12_06_19_10_35.npy','r')
dataExp11=np.load('../data_Vepp/tbt_2021_12_06_19_11_44.npy','r')
dataExp12=np.load('../data_Vepp/tbt_2021_12_06_19_12_26.npy','r')
dataExp13=np.load('../data_Vepp/tbt_2021_12_06_19_13_32.npy','r')
dataExp14=np.load('../data_Vepp/tbt_2021_12_06_19_14_21.npy','r')
dataExp15=np.load('../data_Vepp/tbt_2021_12_06_19_15_12.npy','r')
dataExp16=np.load('../data_Vepp/tbt_2021_12_06_19_16_22.npy','r')
dataExp17=np.load('../data_Vepp/tbt_2021_12_06_19_17_09.npy','r')
dataExp18=np.load('../data_Vepp/tbt_2021_12_06_19_19_03.npy','r')
dataExp19=np.load('../data_Vepp/tbt_2021_12_06_19_20_00.npy','r')
dataExp20=np.load('../data_Vepp/tbt_2021_12_06_19_20_36.npy','r')

dataExp=np.array([dataExp1,dataExp2,dataExp3,dataExp4,dataExp5,dataExp6,dataExp7,dataExp8])
name_data=open("../data_Vepp/name.dat","r")


def importExpData(number1,number2,number3,first):

    dataCoordMassive1 = np.array([dataExp1[0][number1][first[number1]:size+first[number1]], dataExp2[0][number1][first[number1]:size+first[number1]],
                                  dataExp3[0][number1][first[number1]:size+first[number1]], dataExp4[0][number1][first[number1]:size+first[number1]],
                                  dataExp5[0][number1][first[number1]:size+first[number1]], dataExp6[0][number1][first[number1]:size+first[number1]],
                                  dataExp7[0][number1][first[number1]:size+first[number1]], dataExp8[0][number1][first[number1]:size+first[number1]],
                                  dataExp9[0][number1][first[number1]:size+first[number1]],dataExp10[0][number1][first[number1]:size+first[number1]],
                                  dataExp11[0][number1][first[number1]:size+first[number1]],dataExp12[0][number1][first[number1]:size+first[number1]],
                                  dataExp13[0][number1][first[number1]:size+first[number1]],dataExp14[0][number1][first[number1]:size+first[number1]],
                                  dataExp15[0][number1][first[number1]:size+first[number1]],dataExp16[0][number1][first[number1]:size+first[number1]],
                                  dataExp17[0][number1][first[number1]:size+first[number1]],dataExp18[0][number1][first[number1]:size+first[number1]],
                                  dataExp19[0][number1][first[number1]:size+first[number1]],dataExp20[0][number1][first[number1]:size+first[number1]]])

    dataCoordMassive2 = np.array([dataExp1[0][number2][first[number2]:size+first[number2]], dataExp2[0][number2][first[number2]:size+first[number2]],
                                  dataExp3[0][number2][first[number2]:size+first[number2]], dataExp4[0][number2][first[number2]:size+first[number2]],
                                  dataExp5[0][number2][first[number2]:size+first[number2]], dataExp6[0][number2][first[number2]:size+first[number2]],
                                  dataExp7[0][number2][first[number2]:size+first[number2]], dataExp8[0][number2][first[number2]:size+first[number2]],
                                  dataExp9[0][number2][first[number2]:size+first[number2]],dataExp10[0][number2][first[number2]:size+first[number2]],
                                  dataExp11[0][number2][first[number2]:size+first[number2]],dataExp12[0][number2][first[number2]:size+first[number2]],
                                  dataExp13[0][number2][first[number2]:size+first[number2]],dataExp14[0][number2][first[number2]:size+first[number2]],
                                  dataExp15[0][number2][first[number2]:size+first[number2]],dataExp16[0][number2][first[number2]:size+first[number2]],
                                  dataExp17[0][number2][first[number2]:size+first[number2]],dataExp18[0][number2][first[number2]:size+first[number2]],
                                  dataExp19[0][number2][first[number2]:size+first[number2]],dataExp20[0][number2][first[number2]:size+first[number2]]])

    dataCoordMassive3 = np.array([dataExp1[0][number3][first[number3]:size+first[number3]], dataExp2[0][number3][first[number3]:size+first[number3]],
                                  dataExp3[0][number3][first[number3]:size+first[number2]], dataExp4[0][number3][first[number3]:size+first[number3]],
                                  dataExp5[0][number3][first[number3]:size+first[number3]], dataExp6[0][number3][first[number3]:size+first[number3]],
                                  dataExp7[0][number3][first[number3]:size+first[number3]], dataExp8[0][number3][first[number3]:size+first[number3]],
                                  dataExp9[0][number3][first[number3]:size+first[number3]],dataExp10[0][number3][first[number3]:size+first[number3]],
                                  dataExp11[0][number3][first[number3]:size+first[number3]],dataExp12[0][number3][first[number3]:size+first[number3]],
                                  dataExp13[0][number3][first[number3]:size+first[number3]],dataExp14[0][number3][first[number3]:size+first[number3]],
                                  dataExp15[0][number3][first[number3]:size+first[number3]],dataExp16[0][number3][first[number3]:size+first[number3]],
                                  dataExp17[0][number3][first[number3]:size+first[number3]],dataExp18[0][number3][first[number3]:size+first[number3]],
                                  dataExp19[0][number3][first[number3]:size+first[number3]],dataExp20[0][number3][first[number3]:size+first[number3]]])
    return [dataCoordMassive1,dataCoordMassive2,dataCoordMassive3]

dt_1=np.dtype([('name','U4'),('f_name','int')])
first=np.genfromtxt(name_data,dtype=dt_1)['f_name']
print(first)
############################################################################################################
all_data_massive=[]
for i in range(54):
    all_data_massive.append(dataExp2[0][i][first[i]:size+first[i]])
print(len(all_data_massive))
############################################################################################################

"импортирование модельных данных"
"фаза и бета"
data=import_DATA()
delt_phase=np.empty(0)
phase=np.empty(0)
for i in np.arange(len(data)).tolist():
    phase=np.append(phase,data[i][4])

for i in np.arange(1,len(data)).tolist():
    delt_phase=np.append(delt_phase,data[i][4]-data[i-1][4])

model_beta = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_beta=np.append(model_beta,data[i][2])

model_alf = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_alf=np.append(model_alf,data[i][3])

############################################################################################################
'''
len1-сколько точек используется для определения частоты
len2-сколько точек используется для определения ampl
'''
len1=300
len2=124
exp_data=importExpData(number1,number2,number3,first)
freq_class_massive=np.empty(0)
freq_massive=np.empty(0)
ampl_massive=np.empty(0)
for i in np.arange(int(np.shape(exp_data)[1])):
    freq_class_massive=np.append(freq_class_massive,fl.Calculate_Freq(exp_data[0][i][:len1],N_pad=0))
    freq_massive=np.append(freq_massive,freq_class_massive[i].calculate_freq)
    ampl_massive=np.append(ampl_massive,freq_class_massive[i].calculate_ampl_and_phase(len2)[0])

plt.scatter(ampl_massive,freq_massive)
plt.show()