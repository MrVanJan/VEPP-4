import numpy as np
from Vepp_lib import twiss_parametrs_exp as tpe
import pandas as pd

def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array

dataExp2=np.load('../data_Vepp/tbt_2021_12_06_19_02_36.npy','r')
print(dataExp2)
name_data=open("../data_Vepp/name.dat","r")
dt_1=np.dtype([('name','U4'),('f_name','int')])
first=np.genfromtxt(name_data,dtype=dt_1)['f_name']
size=200
all_data_massive=[]
for i in range(54):
    all_data_massive.append(dataExp2[0][i][first[i]:size+first[i]])

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


class_mass=tpe.Twiss_data(type="1",bpm_numbers=np.array([1,2,3]),x_coordinats1=all_data_massive[0],x_coordinats2=all_data_massive[1],x_coordinats3=all_data_massive[2],model_phase=phase,model_beta=model_beta,model_alf=model_alf,len=size)
beta_exp_massive=class_mass.beta_from_phase_N(N=4,number=2,data=all_data_massive,len=size)
print(beta_exp_massive)
print(model_beta[9])
