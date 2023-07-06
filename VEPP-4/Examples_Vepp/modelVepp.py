import numpy as np
from Vepp_lib import model_structure_vepp as msv
import pandas as pd
import matplotlib.pyplot as plt

def import_DATA():
    data = pd.read_csv(r"../data_Vepp/vepp4m-inj.dat", sep="\t", header=None)
    array = []
    for i in range(len(data)):
        array.append(data.loc[i])
    return array
data=import_DATA()

init=np.array([[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0]])

cl_massive=msv.Trajectory(init_cond=init,sext_location=1,strenght=0.1,number=0,power=1000,data=data)
tr=cl_massive.calculate_trajectory()
tr_lin=cl_massive.liniarization_trajectory()
plt.figure(figsize=(24, 12))
plt.subplot(1,2,1)
plt.scatter(tr[0],tr[1])
plt.subplot(1,2,2)
plt.scatter(tr_lin[0],tr_lin[1])
plt.show()