import numpy as np
from Vepp_lib import model_structure_vepp as msv
import pandas as pd
from Vepp_lib import tarajectory_methods_vepp as tmv
import matplotlib.pyplot as plt

def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array
data=import_DATA()

delt_phase=np.empty([0])
for i in np.arange(1,len(data)).tolist():
    delt_phase=np.append(delt_phase,data[i][4]-data[i-1][4])

model_beta = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_beta=np.append(model_beta,data[i][2])

model_alf = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_alf=np.append(model_alf,data[i][3])

init=np.array([[0.002,pow(10,-4)]])
#numbers=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
numbers=np.array([i for i in range(10)])
power=100
cl_massive=np.array([msv.Trajectory(init_cond=init,sext_location=0,strenght=0,number=i,power=power,data=data) for i in range(54)])
x_tr_massive=np.array([cl_massive[i].calculate_trajectory()[0] for i in np.arange(len(cl_massive))])
px_tr_massive=np.array([cl_massive[i].calculate_trajectory()[1] for i in np.arange(len(cl_massive))])

cl1=tmv.Matrix_N_coord_pickUp(numbers=numbers,beta_mass=model_beta,alf_mass=model_alf,delt_phase_m=delt_phase,x_massive=x_tr_massive)
cl1.matrix
coordinats=cl1.coord(type="z")
imp=cl1.coord(type="pz")

x=np.arange(len(x_tr_massive[0]))

"""
создание шума ~40-110 микрон
"""
noize=75*pow(10,-6)+25*np.random.normal(0,0.5,power)*pow(10,-6)
noize_data=x_tr_massive+noize
cl2=tmv.Matrix_N_coord_pickUp(numbers=numbers,beta_mass=model_beta,alf_mass=model_alf,delt_phase_m=delt_phase,x_massive=noize_data)
cl2.matrix
coordinats_noize=cl2.coord(type="z")
imp_noize=cl2.coord(type="pz")


"""
Графики
"""
plt.figure(figsize=(12, 12))
plt.subplot(1,2,1)
plt.scatter(x,x_tr_massive[0]-coordinats)
plt.title("модельное значение-найденное значение(N-ПИКАПОВ)(x-номер оборота)")
plt.subplot(1,2,2)
plt.scatter(x,x_tr_massive[0]-coordinats_noize)
plt.title("тоже самое для данных с шумом")
plt.show()

plt.figure(figsize=(12, 12))
plt.subplot(1,2,1)
plt.scatter(x,px_tr_massive[0]-imp)
plt.title("модельное значение-найденное значение(N-ПИКАПОВ)(x-номер оборота)")
plt.subplot(1,2,2)
plt.scatter(x,px_tr_massive[0]-imp_noize)
plt.title("тоже самое для данных с шумом")
plt.show()

"""
исследование точности от кол-ва датчиков
"""
cl_massive1=np.array([msv.Trajectory(init_cond=init,sext_location=0,strenght=0,number=i,power=1,data=data) for i in range(54)])
x_tr_massive1=np.array([cl_massive1[i].calculate_trajectory()[0] for i in np.arange(len(cl_massive))])
numbers_mass=np.arange(54)
noize_massive=np.array([75*pow(10,-6)+25*np.random.normal(0,0.5,1)*pow(10,-6) for i in range(54)])
x_tr_massive1_noize=x_tr_massive1+noize_massive
print(np.shape(x_tr_massive1))
print(np.shape(x_tr_massive1_noize))
coord3=np.empty([0])
imp3=np.empty([0])
coord3_noize=np.empty([0])
imp3_noize=np.empty([0])
for i in np.arange(2,54):
    numbers=numbers_mass[0:i]
    cl3 = tmv.Matrix_N_coord_pickUp(numbers=numbers, beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
                                    x_massive=x_tr_massive1)
    cl3_noize = tmv.Matrix_N_coord_pickUp(numbers=numbers, beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
                                    x_massive=x_tr_massive1_noize)
    cl3.matrix
    cl3_noize.matrix
    coord3=np.append(coord3,cl3.coord(type="z"))
    imp3=np.append(imp3,cl3.coord(type="pz"))
    coord3_noize = np.append(coord3_noize, cl3_noize.coord(type="z"))
    imp3_noize = np.append(imp3_noize, cl3_noize.coord(type="pz"))

###############################
"""
2-pickUp
"""
two_pickUp_massive=np.empty([0])
for i in np.arange(1,54):
    pick2_class=tmv.Method_two_pickup(x_massive1=x_tr_massive1[0],x_massive2=x_tr_massive1[i],beta1=model_beta[0],
                                      beta2=model_beta[i],delt_phase=np.sum(delt_phase[:(i-1)]),alph1=model_alf[0])
    two_pickUp_massive=np.append(two_pickUp_massive,pick2_class.px_massive()[1])
print(two_pickUp_massive)
##############################
plt.figure(figsize=(12, 12))
plt.subplot(2,2,1)
plt.scatter(np.arange(len(coord3)),coord3)
plt.plot(np.arange(np.shape(coord3)[0]),np.array([x_tr_massive[0][0]for i in np.arange(np.shape(coord3)[0])] ),'r')
plt.title("X")

plt.subplot(2,2,3)
plt.title("Px")
plt.scatter(np.arange(len(imp3)),imp3)
plt.plot(np.arange(np.shape(imp3)[0]),np.array([px_tr_massive[0][0]for i in np.arange(np.shape(imp3)[0])] ),'r')

plt.subplot(2,2,2)
plt.scatter(np.arange(len(coord3_noize)),coord3_noize)
plt.plot(np.arange(np.shape(coord3_noize)[0]),np.array([init[0][0]for i in np.arange(np.shape(coord3_noize)[0])] ),'r')
plt.title("X с шумом")

plt.subplot(2,2,4)
plt.scatter(np.arange(len(imp3_noize)),imp3_noize)
plt.plot(np.arange(np.shape(imp3_noize)[0]),np.array([init[0][1]for i in np.arange(np.shape(imp3_noize)[0])] ),'r')
#plt.scatter(np.arange(1,54),two_pickUp_massive,'g')
plt.title("Px с шумом")
plt.show()