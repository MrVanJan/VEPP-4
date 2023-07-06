from Vepp_lib import model_structure_vepp as msv
from Vepp_lib import tarajectory_methods_vepp as tmv
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import matrix_power
import  matplotlib.pyplot as plt


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
model_phase=np.empty(0)
for i in np.arange(len(data)).tolist():
    model_phase=np.append(model_phase,data[i][4])
#########################################################################################################################################
number1=0
number2=1
power=1000
init=np.array([[0.002,pow(10,-4)],[0.004,pow(10,-4)],[0.005,pow(10,-4)],[0.006,pow(10,-4)],[0.01,pow(10,-4)],[0.012,pow(10,-4)]])
cl_massive=np.array([msv.Trajectory(init_cond=init,sext_location=53,strenght=0,number=i,power=power,data=data) for i in range(54)])
x_tr_massive=np.array([cl_massive[i].calculate_trajectory()[0] for i in np.arange(len(cl_massive))])
#########################################################################################################################################

x_mass_1=np.empty(0)
numbers=np.array([0,1,2,3,4,5,6,7])
sigma_massive=np.random.random(np.shape(numbers)[0])/10000
weight=1/(sigma_massive**2)
print(sigma_massive)
for i in np.arange(np.shape(numbers)[0]):
    cl_massive[numbers[i]].init_condBPM_error(nu=0, sigma=1/1000000)
    cl_massive[numbers[i]].calculate_trajectory_error()
    x_mass_1=np.append(x_mass_1,cl_massive[numbers[i]].noize_trajectory_error(sigma=sigma_massive[i],nu=0))

x_mass_1=np.reshape(x_mass_1,(np.shape(numbers)[0],-1))
print(np.shape(x_mass_1))
metod_2_pick_Up_tragectory_coordinates=tmv.Method_two_pickup(x_massive1=x_mass_1[0],x_massive2=x_mass_1[1],beta1=model_beta[number1],
                                                             beta2=model_beta[number2],alph1=model_alf[number1],delt_phase=model_phase[number2]-model_phase[number1]).px_massive()

cl_N_mass=tmv.N_pickUp_Matr_sm(numbers=numbers,beta_mass=model_beta,alf_mass=model_alf,delt_phase_m=delt_phase,x_massive=x_mass_1)
cl_N_mass.matrix
resN=cl_N_mass.squad_method(N=np.shape(init)[0]*power,type="O",weight=weight)
resN_mass=cl_N_mass.squad_method(N=np.shape(init)[0]*power,type="W",weight=weight)
print(np.shape(resN))
print(np.shape(metod_2_pick_Up_tragectory_coordinates))
plt.scatter(metod_2_pick_Up_tragectory_coordinates[0],metod_2_pick_Up_tragectory_coordinates[1])
plt.scatter(resN[0],resN[1])
plt.scatter(resN_mass[0],resN_mass[1])
plt.show()

#########################################################################################################################################



# numbers=np.array([i for i in np.arange(54)])
# cl=N_pickUp_Matr(numbers=numbers,beta_mass=model_beta,alf_mass=model_alf,delt_phase_m=delt_phase,x_massive=x_tr_massive1)
# cl.matrix
# cl.squad_method(N=0,type="O",weight=np.array([0,1,2]))
#
# class_massive=np.array([msv.Structure(name="1",type="1",begin=i,end=2,data=data) for i in np.arange(54)])
# turn_matr_massive=np.array([class_massive[i].one_turn_matrix_error(nu=0,sigma=0.5,num=i) for i in np.arange(54)])
# sigma_massive_noize=3*np.random.random(54)*(10**6)
# init_error_coord=np.array([class_massive[i].init_error(init_cond=inition,num=i,nu=0,sigma=0.5) for i in np.arange(54)])
# power=1
# coord=np.empty(0)
# for i in np.arange(54):
#     x=np.empty([0])
#     for j in np.arange(power):
#         x=np.append(x,np.dot(matrix_power(turn_matr_massive[i],j),init_error_coord[i])[0])
#     coord=np.append(coord,x)
# coord=np.reshape(coord,(54,-1))
#
#
# #sigma=0.5
# nu=0
# noize_massive=np.array([np.random.normal(nu,1/sigma_massive_noize[i],1) for i in range(54)])
# print(noize_massive)
# coord_noize=coord+noize_massive
#
# x_dat=np.empty(0)
# p_dat=np.empty(0)
# for i in np.arange(2,54):
#     class1 = N_pickUp_Matr(numbers=numbers[:i], beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
#                            x_massive=coord_noize)
#     class1.matrix
#     x_dat=np.append(x_dat,class1.squad_method(N=0,type="O",weight=np.array([0,1,2]))[0])
#     p_dat=np.append(p_dat,class1.squad_method(N=0,type="O",weight=np.array([0,1,2]))[1])
# x_dat=np.reshape(x_dat,(-1,3))
# p_dat=np.reshape(p_dat,(-1,3))
#
# x_dat1=np.empty(0)
# p_dat1=np.empty(0)
#
# for i in np.arange(2,54):
#     class1 = N_pickUp_Matr(numbers=numbers[:i], beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
#                            x_massive=coord_noize)
#     class1.matrix
#     weight=np.array([1/sigma_massive_noize[j]**2 for j in np.arange(len(numbers[:i])-1)])
#     x_dat1=np.append(x_dat1,class1.squad_method(N=0,type="W",weight=weight)[0])
#     p_dat1=np.append(p_dat1,class1.squad_method(N=0,type="W",weight=weight)[1])
# x_dat1=np.reshape(x_dat1,(-1,3))
# p_dat1=np.reshape(p_dat1,(-1,3))
#
#
#
#
#
#
# plt.scatter(np.array([[i,i,i] for i in np.arange(len(x_dat))]),x_dat)
# plt.scatter(np.array([[i,i,i] for i in np.arange(len(x_dat))]),x_dat1)
# plt.plot(np.arange(np.shape(x_dat)[0]),np.array([init[0][0]for i in np.arange(np.shape(x_dat)[0])] ),'r')
# plt.plot(np.arange(np.shape(x_dat)[0]),np.array([init[0][1]for i in np.arange(np.shape(x_dat)[0])] ),'g')
# plt.scatter(np.array([[i,i,i] for i in np.arange(len(x_dat))]),p_dat)
# plt.scatter(np.array([[i,i,i] for i in np.arange(len(x_dat))]),p_dat1)
# plt.show()