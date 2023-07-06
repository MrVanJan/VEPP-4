import numpy as np
from prettytable import PrettyTable
import copy

class Element():

    def __init__(self, name:str, type:str,length:float) -> None:
        self.name = name
        self.type = type
        self.length=length

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1.0, self.length], [0.0, 1.0]])

    def forward(self, state:np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, state.T).T



class Structure(Element):
    '''
    Класс структура,свойства класса:
    1)переопределил свойство matrix,теперь при создании экземпляра класса типа transport matrix,вызывается метод нахождения
    транспортной матрицы.
    begin и end -номера BPM's между которыми находится транспортная матрица
    при создании экземпляра класса типа turn matrix,вызывается метод нахождения
    оборотной матрицы.
    begin - номер BPM для которого находится матрица оборота
    2)метод information возращает таблицу с информацией (тип,матрица и тд) о матрице в зависимости от типа
    3)метод twiss_parametrs-возвращает частоту и параметры твисса
    '''

    def __init__(self,name:str,type:str,begin:int,end:int,data):
        super().__init__( name, type,0)
        self.begin=begin
        self.end=end
        self.data = data
        self.type=type
        self.number=54

        if self.type == 'transport matrix':
            self.name_type = type + '' + str(self.data[self.begin][0]) + '-' + str(self.data[self.end][0])

        elif self.type == 'turn matrix':
            self.name_type = type + '' + str(self.data[self.begin][0])

    @property
    def matrix(self) -> np.ndarray:
        if self.type =='transport matrix':
            return self.calculate_TM()
        elif self.type == 'turn matrix':
            return self.one_turn_matrix()

    @property
    def information(self):
        information = PrettyTable()
        information.field_names = ["type", "number", "Matrix"]
        if self.type == 'transport matrix':
            information.add_row([self.name_type, str(self.begin)+'-'+str(self.end), self.calculate_TM()])
        elif self.type == 'turn matrix':
            information.add_row([self.name_type, self.begin, self.one_turn_matrix()])

        return information


    def twiss_parametrs(self):
        if self.type == 'turn matrix':
            self.freq = np.arccos((self.matrix[0, 0] + self.matrix[1, 1]) / 2)
            self.beta = np.abs(self.matrix[0, 1] / np.sin(self.freq))
            self.alf = -(self.matrix[0, 0] - self.matrix[1, 1]) / (2 * np.sin(self.freq))
            return self.freq, self.beta, self.alf
        else:
            print('the function is not supported by this type of matrix ')

    def calculate_TM(self)->np.ndarray:

        a11 = np.sqrt(self.data[self.end][2] / self.data[self.begin][2]) * (
                    np.cos(self.data[self.end][4] - self.data[self.begin][4]) + self.data[self.begin][3]
                    * np.sin(self.data[self.end][4] - self.data[self.begin][4]))

        a12 = np.sqrt(self.data[self.end][2] * self.data[self.begin][2]) * np.sin(
            self.data[self.end][4] - self.data[self.begin][4])

        a21 = -((1 + self.data[self.begin][3] * self.data[self.end][3]) / np.sqrt(
            self.data[self.end][2] * self.data[self.begin][2])) * np.sin(
            self.data[self.end][4] - self.data[self.begin][4]) + (
                               (self.data[self.begin][3] - self.data[self.end][3]) / np.sqrt(
                           self.data[self.end][2] * self.data[self.begin][2])) * np.cos(
            self.data[self.end][4] - self.data[self.begin][4])

        a22 = np.sqrt(self.data[self.begin][2] / self.data[self.end][2]) * (
                    np.cos(self.data[self.end][4] - self.data[self.begin][4]) - self.data[self.end][
                3] * np.sin(self.data[self.end][4] - self.data[self.begin][4]))

        return np.array([[a11,a12],[a21,a22]])


    def one_turn_matrix(self):

        if self.begin==0:
            self.turn_matrix=Structure('none','none',self.begin,self.number,self.data).calculate_TM()
        else:
            self.turn_matrix=np.dot(Structure('none','none',0,self.begin,self.data).calculate_TM(),Structure('none','none',self.begin,self.number,self.data).calculate_TM())

        return self.turn_matrix

    def one_turn_matrix_error(self,sigma,nu,num):
        k = np.random.normal(nu, sigma, 54) * pow(10, -3)
        turn_matr_error = np.array([[1, 0], [0, 1]])
        if num==0:
            for i in np.arange(54):
                if i == 53:
                    j = 54
                else:
                    j = i + 1
                k_matr = np.array([[1, 0], [k[i], 1]])
                turn_matr_error = np.dot(Structure('none', 'none', i, j, self.data).calculate_TM(),
                                         np.dot(k_matr, turn_matr_error))
        else:
            for i in np.arange(num,54):
                if i == 53:
                    j = 54
                else:
                    j = i + 1
                k_matr = np.array([[1, 0], [k[i], 1]])
                turn_matr_error = np.dot(Structure('none', 'none', i, j, self.data).calculate_TM(),np.dot(k_matr, turn_matr_error))
            for i in np.arange(0,num):
                j = i + 1
                k_matr = np.array([[1, 0], [k[i], 1]])
                turn_matr_error = np.dot(Structure('none', 'none', i, j, self.data).calculate_TM(),np.dot(k_matr, turn_matr_error))

        return turn_matr_error

    def init_error(self,init_cond:np.array,num:int,nu:float,sigma:float):
        if num==0:
            init_error_coord=init_cond
        else:
            transp_matr=np.array([[1,0],[0,1]])
            k = np.random.normal(nu, sigma, 54) * pow(10, -4)
            for i in np.arange(0,num):
                k_matr = np.array([[1, 0], [k[i], 1]])
                j=i+1
                transp_matr=np.dot(Structure('none', 'none', i, j, self.data).calculate_TM(),np.dot(k_matr, transp_matr))
            init_error_coord=np.dot(transp_matr,init_cond)
        return init_error_coord
class Sextupole(Element):

    def __init__(self, name:str=None, strength:float=0.0) -> None:
        super().__init__(name, self.__class__.__name__, 0.0)
        self.strength = strength

    def forward(self, state:np.ndarray) -> np.ndarray:
        q, p = state.T

        return np.array([q, p + self.strength*q**2]).T

class Trajectory():
    '''
    класс траектория
    методы:
    1)init_condBPM-находит начальные условия для bpm под номером number(входной массив n начальный условий на 0-ом bpm)
    calculate_trajectory-считает траекторию для n начальных условий
    если sext_location>=number:
    coord=T0-i.Ts-0.S.Ti-s.Init_new
    если sext_location<number:
    coord=Ts-i.S.T0-s.Ti-0.Init_new
    у класса есть свой метод линеаризации траектории- liniarization_trajectory (удобно)
    у метода 2х пикапов своя линеаризация

    '''

    def __init__(self,init_cond,sext_location:int,strenght:float,number:int,power:int,data):
        self.data = data
        self.sext_location = sext_location
        self.strenght = strenght
        self.init_cond = init_cond
        self.number=number
        self.end = 54
        self.power=power

    def init_condBPM(self):

        if self.sext_location >= self.number:
            self.initcond_i_bpm=np.dot(Structure('none','none',0,self.number,self.data).calculate_TM(),self.init_cond.T).T
        else:
            self.initcond_i_bpm = np.dot(Structure('none', 'none', 0, self.sext_location, self.data).calculate_TM(),self.init_cond.T).T
            self.initcond_i_bpm=np.array([Sextupole(name="none",strength=self.strenght).forward(self.initcond_i_bpm[i]) for i in range(self.initcond_i_bpm.shape[0])])
            self.initcond_i_bpm=np.dot(Structure('none','none',self.sext_location,self.number,self.data).calculate_TM(),self.initcond_i_bpm.T).T

        return self.initcond_i_bpm

    def init_condBPM_error(self,nu,sigma):
        self.initcond_i_bpm_error = copy.deepcopy(self.init_cond)
        self.sigma=sigma
        self.nu=nu
        if self.sext_location >= self.number:
            for i in np.arange(self.number):
                thin_Q=np.array([[1,0],[np.random.normal(self.nu,self.sigma,1)[0],1]])
                self.initcond_i_bpm_error=np.dot(Structure('none','none',i,i+1,self.data).calculate_TM(), self.initcond_i_bpm_error.T).T
                self.initcond_i_bpm_error = np.dot(thin_Q,self.initcond_i_bpm_error.T).T

        else:
            for i in np.arange(self.sext_location):
                thin_Q = np.array([[1, 0], [np.random.normal(self.nu,self.sigma,1)[0], 1]])
                self.initcond_i_bpm_error = np.dot(Structure('none', 'none', i, i+1, self.data).calculate_TM(), self.initcond_i_bpm_error.T).T
                self.initcond_i_bpm_error=np.dot(thin_Q,self.initcond_i_bpm_error.T).T

            self.initcond_i_bpm_error = np.array([Sextupole(name="none", strength=self.strenght).forward(self.initcond_i_bpm_error[i]) for i in range(self.initcond_i_bpm_error.shape[0])])
            for i in np.arange(self.number-self.sext_location):
                thin_Q = np.array([[1, 0], [np.random.normal(self.nu,self.sigma,1)[0], 1]])
                self.initcond_i_bpm_error = np.dot(Structure('none', 'none', self.sext_location+i,  self.sext_location+i + 1, self.data).calculate_TM(),self.initcond_i_bpm_error.T).T
                self.initcond_i_bpm_error=np.dot(thin_Q,self.initcond_i_bpm_error.T).T

        return self.initcond_i_bpm_error

    def calculate_trajectory_error(self):
        self.init_condBPM_error(self.nu,self.sigma)
        self.coordinats=copy.deepcopy(self.init_condBPM_error(self.nu,self.sigma))
        self.coordinats_Array_error=np.empty(0)
        self.coordinats_Array_error=np.append(self.coordinats_Array_error,self.initcond_i_bpm_error)
        if self.sext_location >= self.number:

            for j in np.arange(self.power-1).tolist():

                for i in np.arange(self.sext_location-self.number):
                    thin_Q = np.array([[1, 0], [np.random.normal(self.nu, self.sigma, 1)[0], 1]])
                    self.coordinats = np.dot(Structure('none', 'none', self.number+i, self.number+i+1, self.data).calculate_TM(),self.coordinats.T).T
                    self.coordinats=np.dot(thin_Q,self.coordinats.T).T

                self.coordinats = np.array(
                    [Sextupole(name="none", strength=self.strenght).forward(self.coordinats[k]) for k in
                     range(self.initcond_i_bpm.shape[0])])


                for i in np.arange(self.end-self.sext_location):
                    thin_Q = np.array([[1, 0], [np.random.normal(self.nu, self.sigma, 1)[0], 1]])
                    self.coordinats = np.dot(Structure('none', 'none', self.sext_location+i, self.sext_location+i+1, self.data).calculate_TM(),self.coordinats.T).T
                    self.coordinats=np.dot(thin_Q,self.coordinats.T).T

                for i in np.arange(self.number):
                    thin_Q = np.array([[1, 0], [np.random.normal(self.nu, self.sigma, 1)[0], 1]])
                    self.coordinats = np.dot(Structure('none', 'none', i, i+1, self.data).calculate_TM(),self.coordinats.T).T
                    self.coordinats = np.dot(thin_Q,self.coordinats.T).T
                self.coordinats_Array_error=np.append(self.coordinats_Array_error,self.coordinats)

        else:

            for j in np.arange(self.power-1).tolist():

                for i in np.arange(self.end-self.number):
                    thin_Q = np.array([[1, 0], [np.random.normal(self.nu, self.sigma, 1)[0], 1]])
                    self.coordinats =np.dot(Structure('none', 'none', self.number+i, self.number+i+1, self.data).calculate_TM(),self.coordinats.T).T
                    self.coordinats = np.dot(thin_Q,self.coordinats.T).T

                for i in np.arange(self.sext_location):
                    thin_Q = np.array([[1, 0], [np.random.normal(self.nu, self.sigma, 1)[0], 1]])
                    self.coordinats = np.dot(Structure('none', 'none', i, i + 1, self.data).calculate_TM(), self.coordinats.T).T
                    self.coordinats = np.dot(thin_Q, self.coordinats.T).T

                self.coordinats = np.array(
                    [Sextupole(name="none", strength=self.strenght).forward(self.coordinats[k]) for k in
                     range(self.initcond_i_bpm.shape[0])])

                for i in np.arange(self.number-self.sext_location):
                    thin_Q = np.array([[1, 0], [np.random.normal(self.nu, self.sigma, 1)[0], 1]])
                    self.coordinats = np.dot(Structure('none', 'none', self.sext_location+i, self.sext_location+i + 1, self.data).calculate_TM(),self.coordinats.T).T
                    self.coordinats = np.dot(thin_Q, self.coordinats.T).T

                self.coordinats_Array_error=np.append(self.coordinats_Array_error,self.coordinats)

        self.vector_massive=np.reshape(self.coordinats_Array_error,(self.power*self.initcond_i_bpm.shape[0],2))
        self.res=np.reshape(self.coordinats_Array_error,(self.initcond_i_bpm.shape[0]*self.power,2)).T
        return self.res

    def noize_trajectory_error(self,sigma,nu):
        self.noize_error_X_coordinat=self.res[0]+np.random.normal(nu,sigma,np.shape(self.res[0])[0])
        return self.noize_error_X_coordinat

    def calculate_trajectory(self):
        self.init_condBPM()
        self.coordinats=copy.deepcopy(self.init_condBPM())
        self.coordinats_Array=np.empty(0)
        self.coordinats_Array=np.append(self.coordinats_Array,self.initcond_i_bpm)
        if self.sext_location >= self.number:

            for i in np.arange(self.power-1).tolist():

                self.coordinats = np.dot(
                    Structure('none', 'none', self.number, self.sext_location, self.data).calculate_TM(),
                    self.coordinats.T).T
                self.coordinats = np.array(
                    [Sextupole(name="none", strength=self.strenght).forward(self.coordinats[j]) for j in
                     range(self.initcond_i_bpm.shape[0])])
                self.coordinats = np.dot(
                    Structure('none', 'none', self.sext_location, self.end, self.data).calculate_TM(),
                    self.coordinats.T).T
                self.coordinats = np.dot(Structure('none', 'none', 0, self.number, self.data).calculate_TM(),
                                         self.coordinats.T).T

                self.coordinats_Array=np.append(self.coordinats_Array,self.coordinats)

        else:

            for i in np.arange(self.power-1).tolist():

                self.coordinats = np.dot(
                    Structure('none', 'none', self.number, self.end, self.data).calculate_TM(),
                    self.coordinats.T).T
                self.coordinats = np.dot(
                    Structure('none', 'none', 0, self.sext_location, self.data).calculate_TM(),
                    self.coordinats.T).T
                self.coordinats = np.array(
                    [Sextupole(name="none", strength=self.strenght).forward(self.coordinats[j]) for j in
                     range(self.initcond_i_bpm.shape[0])])
                self.coordinats = np.dot(
                    Structure('none', 'none', self.sext_location, self.number, self.data).calculate_TM(),
                    self.coordinats.T).T

                self.coordinats_Array=np.append(self.coordinats_Array,self.coordinats)

        self.vector_massive=np.reshape(self.coordinats_Array,(self.power*self.initcond_i_bpm.shape[0],2))

        return np.reshape(self.coordinats_Array,(self.initcond_i_bpm.shape[0]*self.power,2)).T


    def liniarization_trajectory(self):
        self.twiss_parametrs=Structure(name='Turn',type='turn matrix',begin=self.number,end=0,data=self.data).twiss_parametrs()
        normalization_matrix=np.array([[1/np.sqrt(self.twiss_parametrs[1]),0],[self.twiss_parametrs[2]/np.sqrt(self.twiss_parametrs[1]),np.sqrt(self.twiss_parametrs[1])]])
        self.norm_coordinats_Array=np.dot(normalization_matrix,self.vector_massive.T).T
        return np.reshape(self.norm_coordinats_Array,(self.power*self.initcond_i_bpm.shape[0],2)).T