import numpy as np
import statsmodels.api as sm

class Method_two_pickup():
    '''
    класс Method_two_pickup
    x_massive1,x_massive2 - массивы координат с двух различных датчиков
    beta1,beta2,alph1 - значения опт.ф-ий на двух различных датчиках
    delt_phase - набек между двумя датчиками

    self.x_massive1[0].shape[0] - число оборотов
    liniarization_trajectory - метод линеаризации траектории,не стал делать как отдельный класс т.к линеаризация нужна только после нахождения не линеаризованной траектории.
    '''

    def __init__(self,x_massive1:np.array,x_massive2:np.array,beta1:float,beta2:float,delt_phase,alph1:float):
        self.x_massive1=x_massive1
        self.x_massive2 = x_massive2
        self.beta1=beta1
        self.beta2 = beta2
        self.delt_phase=delt_phase
        self.alph1=alph1
    def px_massive(self):
        self.coordinats_Array=np.empty(0)
        for j in np.arange(self.x_massive1.shape[0]).tolist():
            self.x=self.x_massive1[j]
            self.p=(self.x_massive2[j]-np.sqrt(self.beta2/self.beta1)*(np.cos(self.delt_phase)+self.alph1*np.sin(self.delt_phase))*self.x_massive1[j])/(np.sqrt(self.beta2*self.beta1)*np.sin(self.delt_phase))
            #self.p=(np.sqrt(self.beta1/self.beta2)*self.x_massive2[j]-self.x_massive1[j]*np.cos(self.delt_phase))/np.sin(self.delt_phase)
            self.coordinats_Array=np.append(self.coordinats_Array,np.array([self.x,self.p]))
        self.vector_massive = np.reshape(self.coordinats_Array, (self.x_massive1.shape[0], 2))
        return np.reshape(self.coordinats_Array,(self.x_massive1.shape[0],2)).T

    def liniarization_trajectory(self,beta1_model,alf1_model):
        self.normalization_matrix = np.array([[1 / np.sqrt(beta1_model), 0],[alf1_model/ np.sqrt(beta1_model),np.sqrt(beta1_model)]])
        self.norm_coordinats_Array = np.dot(self.normalization_matrix, self.vector_massive.T).T
        return np.reshape(self.norm_coordinats_Array, (self.x_massive1.shape[0], 2)).T

class N_pickUp_Matr_sm():
    '''
    класс N_pickUp_Matr_sm
    ИСПОЛЬЗУЕТСЯ БИБЛИОТЕКА statsmodels
    numbers-номера используемых датчиков,координаты считаются на 1м датчике в этом массиве используя остальные
    x_massive-массив координат размерностью(len(numbers),N)
    N-кол-во оборотов
    def matrix-нахождение матриц перехода между пикапом с номером numbers[0] и остальными в списке
    '''
    def __init__(self,numbers:np.array,beta_mass:np.array,alf_mass:np.array,delt_phase_m:np.array,
                 x_massive:np.array):
        self.numbers=numbers
        self.beta_mass=beta_mass
        self.alf_mass=alf_mass
        self.delt_phase_m=delt_phase_m
        self.x_massive=x_massive

    @property
    def matrix(self) -> np.array:
        self.matr_massive = np.empty(0)
        for i in np.arange(1, np.shape(self.numbers)[0]):
            beta0 = self.beta_mass[self.numbers[0]]
            beta1 = self.beta_mass[self.numbers[i]]
            alph0 = self.alf_mass[self.numbers[0]]
            alph1 = self.alf_mass[self.numbers[i]]
            delt_phase = np.sum(self.delt_phase_m[self.numbers[0]:self.numbers[i]])

            a11 = np.sqrt(beta1 / beta0) * (np.cos(delt_phase) + alph0 * np.sin(delt_phase))

            a12 = np.sqrt(beta1 * beta0) * np.sin(delt_phase)

            a21 = -((1 + alph1 * alph0) / np.sqrt(beta1 * beta0)) * np.sin(delt_phase) + (
                        (alph0 - alph1) / np.sqrt(beta1 * beta0)) * np.cos(delt_phase)

            a22 = np.sqrt(beta0 / beta1) * (np.cos(delt_phase) - alph1 * np.sin(delt_phase))

            self.matr_massive = np.append(self.matr_massive, np.array([[a11, a12], [a21, a22]]))
            self.matr_massive = np.reshape(self.matr_massive, (-1, 4))

        return self.matr_massive

    def squad_method(self,N:int,type:str,weight:np.array):
        """
        type(W)-WLS
        type(O)-OLS
        N-кол-во оборотов
        """
        self.X_matr=np.empty(0)
        for i in np.arange(np.shape(self.numbers)[0]-1):
            self.X_matr=np.append(self.X_matr,np.array([self.matr_massive[i][0],self.matr_massive[i][1]]))
        self.X_matr=np.reshape(self.X_matr,(-1,2))
        y_vect=np.empty(0)
        for i in np.arange(N):
            for j in np.arange(len(self.numbers) - 1):
                y_vect = np.append(y_vect, self.x_massive[self.numbers[1:][j]][i])
        y_vect=np.reshape(y_vect,(N,-1))
        x=np.empty(0)
        px=np.empty(0)
        if type == "O":
            for i in np.arange(np.shape(y_vect)[0]):
                x_px_cl=sm.OLS(y_vect[i],self.X_matr).fit()
                x=np.append(x,x_px_cl.params[0])
                px=np.append(px,x_px_cl.params[1])
        if type == "W":
            for i in np.arange(np.shape(y_vect)[0]):
                x_px_cl=sm.WLS(y_vect[i],self.X_matr,weights=weight).fit()
                x = np.append(x, x_px_cl.params[0])
                px = np.append(px, x_px_cl.params[1])

        return np.array([x,px])

class Matrix_N_coord_pickUp():
    '''
    класс Matrix_N_coord_pickUp
    метод N пикапов для нахождения координат
    def matrix-нахождение матриц перехода между пикапом с номером numbers[0] и остальными в списке
    '''
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

            beta0=self.beta_mass[self.numbers[0]]
            beta1=self.beta_mass[self.numbers[i]]
            alph0=self.alf_mass[self.numbers[0]]
            alph1 = self.alf_mass[self.numbers[i]]
            delt_phase=np.sum(self.delt_phase_m[self.numbers[0]:self.numbers[i]])

            a11 = np.sqrt(beta1 / beta0) * (np.cos(delt_phase) + alph0* np.sin(delt_phase))

            a12 = np.sqrt(beta1*beta0) * np.sin(delt_phase)

            a21 = -((1 + alph1*alph0) / np.sqrt(beta1*beta0)) * np.sin(delt_phase) + ((alph0-alph1) / np.sqrt(beta1*beta0)) * np.cos(delt_phase)

            a22 = np.sqrt(beta0/beta1)* (np.cos(delt_phase) - alph1 * np.sin(delt_phase))

            self.matr_massive=np.append(self.matr_massive,np.array([[a11, a12], [a21, a22]]))
            self.matr_massive = np.reshape(self.matr_massive, (-1, 4))

        return self.matr_massive


    def coord(self,type:str):
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
            z=np.append(z,(-S12*S2z[i]+S22*S1z[i])/S)
            pz=np.append(pz,(S11*S2z[i]-S12*S1z[i])/S)
        if type=="z":
            return z
        if type=="pz":
            return pz