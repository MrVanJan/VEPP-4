import numpy as np
import pandas as pd
from Vepp_lib import twiss_parametrs_exp as tpe
from Vepp_lib import freq_lib as fl
import matplotlib.pyplot as plt


'''
data=import_DATA-импортирование модельных данных
'''
def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array

data=import_DATA()

'''
из data получаем массивы модельных значений оптических функций и фаз
'''
model_phase=np.empty(0)
for i in np.arange(len(data)).tolist():
    model_phase=np.append(model_phase,data[i][4])

model_delt_phase=np.empty(0)
for i in np.arange(1,len(data)).tolist():
    model_delt_phase=np.append(model_delt_phase,data[i][4]-data[i-1][4])

model_beta = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_beta=np.append(model_beta,data[i][2])

model_alf = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_alf=np.append(model_alf,data[i][3])

'''
name.dat-файл в котором находятся номера первых оборотов для различных пикапов
'''
name_data=open("../data_Vepp/name.dat","r")
dt_1=np.dtype([('name','U4'),('f_name','int')])
first=np.genfromtxt(name_data,dtype=dt_1)['f_name']

'''
импортирую экспериментальные данные
'''
dataExp1=np.load('../data_Vepp/tbt_2021_12_06_19_02_17.npy','r')[0]
'''
size-колличество оборотов
numberi-номера пикапов для определения оптики методом 3х пикапов
'''
size=200
number1=0
number2=1
number3=2

'''
создание массива с экспериментальными данными
'''
dataExp_massive=[]
for i in range(54):
    dataExp_massive.append(dataExp1[i][first[i]:size+first[i]])
print(np.shape(dataExp_massive))
'''
calculate_ampl_and_phase- метод класса Calculate_Freq
classMassive(self,exp_ampl_and_phase,model_delt_bpm_phase,parametr)-метод класса Twiss_data
parametr-число показывающее сколько последующих набегов фаз нужно подсчитать(см библиотеку twiss_parametrs_exp)
'''
classMassive=tpe.Twiss_data(type="t",bpm_numbers=np.array([number1,number2,number3]),x_coordinats1=dataExp_massive[number1],
                            x_coordinats2=dataExp_massive[number2],x_coordinats3=dataExp_massive[number3],model_phase=model_phase,
                            model_beta=model_beta,model_alf=model_alf,len=size)
ampl_phase_Massive=np.empty(0)
for i in np.arange(54):
    ampl_phase_Massive=np.append(ampl_phase_Massive,fl.Calculate_Freq(dataExp_massive[i],0).calculate_ampl_and_phase(size))
ampl_phase_Massive=np.reshape(ampl_phase_Massive,(-1,2))
#print(ampl_phase_Massive)
phase=classMassive.exp_phase(ampl_phase_Massive,model_delt_phase,53)

'''
построение графиков cot фазы для модели и эксперимента
'''
x=np.array([i for i in np.arange(53)])

plt.scatter(x,1/np.tan(phase))
plt.scatter(x,1/np.tan(model_delt_phase[0:53]))
plt.show()
'''
подсчет оптических функций методом 3-х пикапов
class Twiss_data библиотеки twiss_parametrs
beta_expF_massive-массив beta_from_phase
beta_expA_massive-массив beta from ampl
'''
len=124
beta_class_massive=np.empty(0)
beta_expF_massive=np.empty(0)
beta_expA_massive=np.empty(0)
for i in range(52):
    beta_class_massive=np.append(beta_class_massive,tpe.Twiss_data(type='None', bpm_numbers=np.array([i, i+1, i+2]),
                                                                   x_coordinats1=dataExp_massive[i],x_coordinats2=dataExp_massive[i+1], x_coordinats3=dataExp_massive[i+2],
                                                                   model_phase=model_phase, model_beta=model_beta, model_alf=model_alf, len=len))
    beta_expF_massive=np.append(beta_expF_massive,beta_class_massive[i].beta_from_phase_3(dataExp_massive[i],dataExp_massive[i+1],dataExp_massive[i+2],np.array([i,i+1,i+2]),1))
    beta_expA_massive=np.append(beta_expA_massive,beta_class_massive[i].beta_from_ampl(i,dataExp_massive,len))
print(beta_expF_massive)
plt.scatter(x[:52],beta_expF_massive)#синий
plt.scatter(x[:52],beta_expA_massive)#ор
plt.scatter(x[:52],model_beta[:52])#зел
plt.show()