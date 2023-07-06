import numpy as np
from Vepp_lib import model_structure_vepp as msv
import pandas as pd
from Vepp_lib import twiss_parametrs_exp as tpe
import matplotlib.pyplot as plt

def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array
############################################################################################################

"импортирование модельных данных"
"фаза и бета"
data=import_DATA()
delt_model_phase=np.empty(0)
model_phase=np.empty(0)
for i in np.arange(len(data)).tolist():
    model_phase=np.append(model_phase,data[i][4])

for i in np.arange(1,len(data)).tolist():
    delt_model_phase=np.append(delt_model_phase,data[i][4]-data[i-1][4])

model_beta = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_beta=np.append(model_beta,data[i][2])

model_alf = np.empty(0)
for i in np.arange(len(data)).tolist():
    model_alf=np.append(model_alf,data[i][3])

############################################################################################################
dataExp2=np.load('../data_Vepp/tbt_2021_12_06_19_02_36.npy','r')

name_data=open("../data_Vepp/name.dat","r")
dt_1=np.dtype([('name','U4'),('f_name','int')])
first=np.genfromtxt(name_data,dtype=dt_1)['f_name']
size=10
all_data_exp_massive=[]
for i in range(54):
    all_data_exp_massive.append(dataExp2[0][i][first[i]:size+first[i]])
print(np.shape(all_data_exp_massive))
class_massive=np.empty(0)
bpm_numb_massive=np.append(np.insert(np.array([i for i in np.arange(54)]),0,53),0)
beta_exp_massive=np.empty(0)
alf_exp_massive=np.empty(0)
exp_delt_phase=np.array(0)

for i in np.arange(1,55):
    class_massive=np.append(class_massive,
                            tpe.Twiss_data(type=str(bpm_numb_massive[i]),bpm_numbers=np.array([bpm_numb_massive[i-1],bpm_numb_massive[i],bpm_numb_massive[i+1]]),x_coordinats1=all_data_exp_massive[bpm_numb_massive[i-1]],x_coordinats2=all_data_exp_massive[bpm_numb_massive[i]],x_coordinats3=all_data_exp_massive[bpm_numb_massive[i+1]],
                                           model_phase=model_phase,model_beta=model_beta,model_alf=model_alf,len=size))
    beta_exp_massive = np.append(beta_exp_massive, class_massive[i-1].beta_from_phase_3(x_coordinats1=all_data_exp_massive[bpm_numb_massive[i-1]],
                                                                                   x_coordinats2=all_data_exp_massive[bpm_numb_massive[i]],
                                                                                   x_coordinats3=all_data_exp_massive[bpm_numb_massive[i+1]],bpm_numbers=np.array([bpm_numb_massive[i-1],bpm_numb_massive[i],bpm_numb_massive[i+1]]),type=1))
    alf_exp_massive = np.append(alf_exp_massive, class_massive[i - 1].beta_from_phase_3(
        x_coordinats1=all_data_exp_massive[bpm_numb_massive[i - 1]],
        x_coordinats2=all_data_exp_massive[bpm_numb_massive[i]],
        x_coordinats3=all_data_exp_massive[bpm_numb_massive[i + 1]],
        bpm_numbers=np.array([bpm_numb_massive[i - 1], bpm_numb_massive[i], bpm_numb_massive[i + 1]]), type=10))
    exp_delt_phase=np.append(exp_delt_phase,class_massive[i-1].dphase()[0])
exp_delt_phase=exp_delt_phase[1:]
print(len(exp_delt_phase))
print(len(alf_exp_massive))
print(len(beta_exp_massive))
############################################################################################################
class Matrix_N_coord_pickUp():
    def __init__(self,numbers:np.array,beta_mass:np.array,alf_mass:np.array,delt_phase_m:np.array,x_massive:np.array):
        self.numbers=numbers
        self.beta_mass=beta_mass
        self.alf_mass = alf_mass
        self.delt_phase_m=delt_phase_m
        self.x_massive=x_massive
    @property

    def matrix(self)->np.array:
        self.matr_massive=np.empty(0)
        for i in np.arange(1,np.shape(self.numbers)[0]):

            self.beta0=self.beta_mass[self.numbers[0]]
            self.beta1=self.beta_mass[self.numbers[i]]
            self.alph0=self.alf_mass[self.numbers[0]]
            self.alph1 = self.alf_mass[self.numbers[i]]
            self.delt_phase=np.sum(self.delt_phase_m[self.numbers[0]:self.numbers[i]])

            self.a11 = np.sqrt(self.beta1 / self.beta0) * (np.cos(self.delt_phase) + self.alph0* np.sin(self.delt_phase))

            self.a12 = np.sqrt(self.beta1*self.beta0) * np.sin(self.delt_phase)

            self.a21 = -((1 + self.alph1*self.alph0) / np.sqrt(self.beta1*self.beta0)) * np.sin(self.delt_phase) + ((self.alph0-self.alph1) / np.sqrt(self.beta1*self.beta0)) * np.cos(self.delt_phase)


            self.a22 = np.sqrt(self.beta0/self.beta1)* (np.cos(self.delt_phase) - self.alph1 * np.sin(self.delt_phase))

            self.matr_massive=np.append(self.matr_massive,np.array([[self.a11, self.a12], [self.a21, self.a22]]))
            self.matr_massive = np.reshape(self.matr_massive, (-1, 4))

        return self.matr_massive

    @property
    def coord(self):
        S11=np.sum(np.array([self.matr_massive[i][0]**2 for i in np.arange(np.shape(self.matr_massive)[0])]))
        S12 = np.sum(np.array([self.matr_massive[i][0]*self.matr_massive[i][1] for i in np.arange(np.shape(self.matr_massive)[0])]))
        S22 = np.sum(np.array([self.matr_massive[i][1]**2 for i in np.arange(np.shape(self.matr_massive)[0])]))
        S=S11*S22-S12**2
        S1z=np.empty(0)
        S2z = np.empty(0)
        z=np.empty(0)
        pz=np.empty(0)
        for j in np.arange(np.shape(self.x_massive)[1]):
            S1z=np.append(S1z,np.sum(np.array([self.matr_massive[i][0]*self.x_massive[self.numbers[1:][i]][j] for i in np.arange(np.shape(self.matr_massive)[0])])))
            S2z = np.append(S2z, np.sum(np.array([self.matr_massive[i][1] * self.x_massive[self.numbers[1:][i]][j] for i in np.arange(np.shape(self.matr_massive)[0])])))
        for i in np.arange(np.shape(S1z)[0]):
            z=np.append(z,(S22*S1z[i]-S11*S2z[i])/S)
            #z=np.reshape(z,(np.shape(self.x_massive)[1],-1))
        return z
            

numbers=np.array([0,1,2,3])

print(np.array([i**2 for i in np.array([1,2,3,4])]))
cl=Matrix_N_coord_pickUp(numbers=numbers,beta_mass=beta_exp_massive,alf_mass=alf_exp_massive,delt_phase_m=exp_delt_phase,x_massive=all_data_exp_massive)

print(cl.matrix)
print(cl.coord)
print(all_data_exp_massive[0])