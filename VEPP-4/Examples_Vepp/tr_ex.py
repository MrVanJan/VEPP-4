import numpy as np
import matplotlib.pyplot as plt
from Vepp_lib import tarajectory_methods_vepp as tmv
from Vepp_lib import twiss_parametrs_exp as tpe
from Vepp_lib import model_structure_vepp as msv
import pandas as pd

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


exp_data1=importExpData(number1,number2,number3,first)
exp_data2=importExpData(number4,number5,number6,first)


############################################################################################################
# step=100
# len=100
#
# cl_massive=np.empty(0)
# freq=np.empty(0)
# ampl=np.empty(0)
# for i in np.arange(size/len):
#     cl_massive=np.append(cl_massive,tv.Calculate_Freq(exp_data1[0][0][int(i*step):int((i+1)*step)]))
#     freq = np.append(freq,cl_massive[int(i)].calculate_freq)
#     ampl=np.append(ampl,cl_massive[int(i)].calculate_ampl_and_phase(len)[0])
# print(np.shape(cl_massive))
# print(freq)
# print(ampl)
# plt.scatter(ampl,freq)
# plt.show()
#
# cl_massive=np.empty(0)
# freq=np.empty(0)
# ampl=np.empty(0)
# for i in np.arange(size/len):
#     cl_massive=np.append(cl_massive,tv.Calculate_Freq(exp_data1[0][0][:int((i+1)*step)]))
#     freq = np.append(freq,cl_massive[int(i)].calculate_freq)
#     ampl=np.append(ampl,cl_massive[int(i)].calculate_ampl_and_phase(len)[0])
# plt.scatter(ampl,freq)
# plt.show()
############################################################################################################

class_massive=np.empty(0)
for i in np.arange(exp_data1[1].shape[0]):
    class_massive=np.append(class_massive,tmv.Method_two_pickup(x_massive1=exp_data1[1][i]-np.mean(exp_data1[1][i]),x_massive2=exp_data2[1][i]-np.mean(exp_data1[2][i]),beta1=model_beta[number2],beta2=model_beta[number5],delt_phase=data[number5][4]-data[number2][4],alph1=model_alf[number2]))

coordinatsModel=[]
for i in range(class_massive.shape[0]):
    coordinatsModel.append(class_massive[i].px_massive())

coordinatsModelLin=[]
for i in range(class_massive.shape[0]):
    coordinatsModelLin.append(class_massive[i].liniarization_trajectory(model_beta[number2],model_alf[number2]))

############################################################################################################
ffff1=tpe.Twiss_data(type='f',bpm_numbers=np.array([number1,number2,number3]),x_coordinats1=exp_data1[0][1],x_coordinats2=exp_data1[1][1],x_coordinats3=exp_data1[2][1],
                   model_phase=phase,model_beta=model_beta,model_alf=model_alf,len=200)
ffff2=tpe.Twiss_data(type='f',bpm_numbers=np.array([number4,number5,number6]),x_coordinats1=exp_data2[0][1],x_coordinats2=exp_data2[1][1],x_coordinats3=exp_data2[2][1],
                   model_phase=phase,model_beta=model_beta,model_alf=model_alf,len=200)

ffff3=tpe.Twiss_data(type='f',bpm_numbers=np.array([number2,number5,number6]),x_coordinats1=exp_data1[0][1],x_coordinats2=exp_data1[1][1],x_coordinats3=exp_data1[2][1],
                   model_phase=phase,model_beta=model_beta,model_alf=model_alf,len=200)

beta2=ffff1.beta_from_phase_3(exp_data1[0][1],exp_data1[1][1],exp_data1[2][1],np.array([number1,number2,number3]),1)
beta5=ffff2.beta_from_phase_3(exp_data2[0][1],exp_data2[1][1],exp_data2[2][1],np.array([number4,number5,number6]),1)
alph2=ffff1.beta_from_phase_3(exp_data1[0][1],exp_data1[1][1],exp_data1[2][1],np.array([number1,number2,number3]),11)

beta_all=np.empty(0)
delt_phase_all=np.empty(0)

for i in range(1,53):

    adds1 = msv.Trajectory(init_cond=np.array([[1, 0]]), sext_location=3, strenght=0.0, number=i-1, power=1024, data=data).calculate_trajectory()
    adds2 = msv.Trajectory(init_cond=np.array([[1, 0]]), sext_location=3, strenght=0.0, number=i, power=1024, data=data).calculate_trajectory()
    adds3 = msv.Trajectory(init_cond=np.array([[1, 0]]), sext_location=3, strenght=0.0, number=i+1, power=1024, data=data).calculate_trajectory()
    ffff = tpe.Twiss_data(type='f', bpm_numbers=np.array([i-1, i, i+1]), x_coordinats1=adds1[0],
                          x_coordinats2=adds2[0], x_coordinats3=adds3[0],
                          model_phase=phase, model_beta=model_beta, model_alf=model_alf, len=124)
    beta_all=np.append(beta_all,ffff.beta_from_phase_3(adds1[0],adds2[0],adds3[0],np.array([i-1,i,i+1]),1))
    delt_phase_all=np.append(delt_phase_all,ffff.exp_delt_phase[0])
############################################################################################################
beta_all1=np.empty(0)
beta_from_ampl1=np.empty(0)
delt_phase_all1=np.empty(0)


for i in range(1,53):
    exp_data=importExpData(i-1,i,i+1,first)

    ffff1 = tpe.Twiss_data(type='f', bpm_numbers=np.array([i-1, i, i+1]), x_coordinats1=exp_data[0][2],
                          x_coordinats2=exp_data[1][2], x_coordinats3=exp_data[2][2],
                          model_phase=phase, model_beta=model_beta, model_alf=model_alf, len=124)
    beta_all1=np.append(beta_all1,ffff1.beta_from_phase_3(exp_data[0][2],exp_data[1][2],exp_data[2][2],np.array([i-1,i,i+1]),1))
    beta_from_ampl1=np.append(beta_from_ampl1,ffff1.beta_from_ampl(i,all_data_massive,124))
    delt_phase_all1=np.append(delt_phase_all1,ffff1.exp_delt_phase[0])




xcoord=np.arange(52)
#plt.scatter(xcoord,beta_all)
plt.scatter(xcoord,model_beta[1:53]-beta_all)
plt.show()
plt.scatter(xcoord,1/np.tan(delt_phase_all)-1/np.tan(delt_phase[:52]))

plt.show()

plt.scatter(xcoord,model_beta[1:53]-beta_all1)
plt.show()
plt.scatter(xcoord,1/np.tan(delt_phase_all1)-1/np.tan(delt_phase[:52]))

plt.show()

plt.scatter(xcoord,model_beta[1:53]-beta_from_ampl1)
plt.show()

ffff3.beta_from_phase_3(exp_data1[1][1],exp_data2[1][1],exp_data2[2][1],np.array([number2,number5,number6]),1)
print(beta2,beta5,alph2)
print(model_beta[number2],model_beta[number5],model_alf[number2])
exp_delt_phase25=ffff3.exp_delt_phase[0]
############################################################################################################








############################################################################################################
class_massive1=np.empty(0)
for i in np.arange(exp_data1[1].shape[0]):
    class_massive1=np.append(class_massive1,tmv.Method_two_pickup(x_massive1=exp_data1[1][i]-np.mean(exp_data1[1][i]),x_massive2=exp_data2[1][i]-np.mean(exp_data1[2][i]),beta1=beta2,beta2=beta5,delt_phase=exp_delt_phase25,alph1=alph2))

coordinatsModel1=[]
for i in range(class_massive1.shape[0]):
    coordinatsModel1.append(class_massive1[i].px_massive())

coordinatsModelLin1=[]
for i in range(class_massive1.shape[0]):
    coordinatsModelLin1.append(class_massive1[i].liniarization_trajectory(model_beta[number2],model_alf[number2]))


############################################################################################################
len=19

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
for i in range(len):
    plt.scatter(coordinatsModel[i][0], coordinatsModel[i][1])

plt.subplot(2, 2, 2)
for i in range(len):
    plt.scatter(coordinatsModelLin[i][0], coordinatsModelLin[i][1])

plt.subplot(2, 2, 3)
for i in range(len):
    plt.scatter(coordinatsModel1[i][0], coordinatsModel1[i][1])

plt.subplot(2, 2, 4)
for i in range(len):
    plt.scatter(coordinatsModelLin1[i][0], coordinatsModelLin1[i][1])
plt.show()