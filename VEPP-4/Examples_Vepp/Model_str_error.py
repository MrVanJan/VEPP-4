import numpy as np
import pandas as pd
from Vepp_lib import model_structure_vepp as msv
from numpy.linalg import matrix_power
from Vepp_lib import tarajectory_methods_vepp as tmv
import matplotlib.pyplot as plt
from Vepp_lib import tarajectory_methods_vepp as tmv
def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array
data=import_DATA()
init_coordinates=np.array([0.002,pow(10,-4)])

delt_phase=np.empty([0])
for i in np.arange(1,len(data)).tolist():
    delt_phase=np.append(delt_phase,data[i][4]-data[i-1][4])

model_beta = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_beta=np.append(model_beta,data[i][2])

model_alf = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_alf=np.append(model_alf,data[i][3])

class_massive=np.array([msv.Structure(name="1",type="1",begin=i,end=2,data=data) for i in np.arange(54)])
turn_matr_massive=np.array([class_massive[i].one_turn_matrix_error(nu=0,sigma=0.5,num=i) for i in np.arange(54)])
init_error_coord=np.array([class_massive[i].init_error(init_cond=init_coordinates,num=i,nu=0,sigma=0.5) for i in np.arange(54)])
freq=np.arccos((turn_matr_massive[0][0,0]+turn_matr_massive[0][1,1])/2)
bet=np.abs(turn_matr_massive[0][0,1]/np.sin(freq))
print((bet-model_beta[0])/model_beta[0]*100)
##########################################################################
power=1
coord=np.empty([0])
for i in np.arange(54):
    x=np.empty([0])
    for j in np.arange(power):
        x=np.append(x,np.dot(matrix_power(turn_matr_massive[i],j),init_error_coord[i])[0])
    coord=np.append(coord,x)
coord=np.reshape(coord,(54,-1))
print(coord)
noize_massive=np.array([75*pow(10,-6)+25*np.random.normal(0,0.5,1)*pow(10,-6) for i in range(54)])
coord_noize=coord+noize_massive
numbers_mass=np.arange(54)
coord3=np.empty([0])
imp3=np.empty([0])
for i in np.arange(2,54):
    numbers=numbers_mass[0:i]
    cl3 = tmv.Matrix_N_coord_pickUp(numbers=numbers, beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
                                    x_massive=coord)

    cl3.matrix
    coord3=np.append(coord3,cl3.coord(type="z"))
    imp3=np.append(imp3,cl3.coord(type="pz"))

plt.figure(figsize=(12, 12))
plt.subplot(2,1,1)
plt.scatter(np.arange(len(coord3)), coord3)
plt.plot(np.arange(np.shape(coord3)[0]), np.array([coord[0][0] for i in np.arange(np.shape(coord3)[0])]), 'r')
plt.title("X")
plt.subplot(2,1,2)
plt.title("Px")
plt.scatter(np.arange(len(imp3)),imp3)
plt.plot(np.arange(np.shape(imp3)[0]),np.array([pow(10,-4)for i in np.arange(np.shape(imp3)[0])] ),'r')
plt.show()
coord4=np.empty([0])
imp4=np.empty([0])
for i in np.arange(2,54):
    numbers=numbers_mass[0:i]
    cl4 = tmv.Matrix_N_coord_pickUp(numbers=numbers, beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
                                    x_massive=coord_noize)

    cl4.matrix
    coord4=np.append(coord4,cl4.coord(type="z"))
    imp4=np.append(imp4,cl4.coord(type="pz"))

plt.figure(figsize=(12, 12))
plt.subplot(2,1,1)
plt.scatter(np.arange(len(coord4)), coord4)
plt.scatter(np.arange(len(coord3)), coord3)
plt.plot(np.arange(np.shape(coord4)[0]), np.array([coord[0][0] for i in np.arange(np.shape(coord4)[0])]), 'r')
plt.title("X")
plt.subplot(2,1,2)
plt.title("Px")
plt.scatter(np.arange(len(imp4)),imp4)
plt.scatter(np.arange(len(imp3)),imp3)
plt.plot(np.arange(np.shape(imp4)[0]),np.array([pow(10,-4)for i in np.arange(np.shape(imp4)[0])] ),'r')


plt.show()



# dataExp2=np.load('../data_Vepp/tbt_2021_12_06_19_02_36.npy','r')
# print(dataExp2)
# name_data=open("../data_Vepp/name.dat","r")
# dt_1=np.dtype([('name','U4'),('f_name','int')])
# first=np.genfromtxt(name_data,dtype=dt_1)['f_name']
# size=1
# all_data_massive=[]
# for i in range(54):
#     all_data_massive.append(dataExp2[0][i][first[i]:size+first[i]])
#
# coord4=np.empty([0])
# imp4=np.empty([0])
# for i in np.arange(2,54):
#     numbers=numbers_mass[0:i]
#     cl4 = npl.Matrix_N_coord_pickUp(numbers=numbers, beta_mass=model_beta, alf_mass=model_alf, delt_phase_m=delt_phase,
#                                     x_massive=all_data_massive)
#
#     cl4.matrix
#     coord4=np.append(coord4,cl4.coord(type="z"))
#     imp4=np.append(imp4,cl4.coord(type="pz"))
# clk=tmv.Method_two_pickup(all_data_massive[0],all_data_massive[1],model_beta[0],model_beta[1],delt_phase[0],model_alf[0])
# print(clk.px_massive())
# plt.figure(figsize=(12, 12))
# plt.subplot(2,1,1)
# plt.scatter(np.arange(len(coord4)), coord4)
# plt.plot(np.arange(np.shape(coord4)[0]), np.array([clk.px_massive()[0] for i in np.arange(np.shape(coord4)[0])]), 'r')
# plt.title("X")
# plt.subplot(2,1,2)
# plt.title("Px")
# plt.scatter(np.arange(len(imp4)),imp4)
# plt.plot(np.arange(np.shape(imp4)[0]),np.array([clk.px_massive()[1]for i in np.arange(np.shape(imp4)[0])] ),'r')
# plt.show()