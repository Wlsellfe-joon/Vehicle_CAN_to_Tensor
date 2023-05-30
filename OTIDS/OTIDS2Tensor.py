import pandas as pd
import numpy as np
import tensorflow as tf
import imageio
from collections import Counter

# Transform raw data to pandas dataframe (OTIDS Dataset case)
class OTIDS_to_matrix:
    def __init__(self, path):
        self.row_data = pd.read_csv(path, sep="\t", header=None, encoding="cp949")
        self.Data_1 = pd.DataFrame(columns=['TimeStamp', 'ID'])
        self.Data_2 = pd.DataFrame(columns=['Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8'])
        self.Data = pd.DataFrame(
            columns=['TimeStamp', 'ID', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8'])

    def to_fr(self):
        self.row_data.rename(columns={0: 'All'}, inplace=True)
        self.row_data = self.row_data.All.str.split(expand=True)

    def to_matrix(self):
        self.Data_1['TimeStamp'] = self.row_data[1]
        self.Data_1['ID'] = self.row_data[3]

        self.Data_2['Data1'] = self.row_data[7]
        self.Data_2['Data2'] = self.row_data[8]
        self.Data_2['Data3'] = self.row_data[9]
        self.Data_2['Data4'] = self.row_data[10]
        self.Data_2['Data5'] = self.row_data[11]
        self.Data_2['Data6'] = self.row_data[12]
        self.Data_2['Data7'] = self.row_data[13]
        self.Data_2['Data8'] = self.row_data[14]

        # Fill in the blank(data section) of dataframe as '0x00' data if there are no data
        temp = self.Data_2.isna()
        for i in (self.Data_2['Data8'].isna().loc[self.Data_2['Data8'].isna() == True].index):
            count = Counter(temp.loc[i] == True)[True]
            self.Data_2.loc[i] = self.Data_2.loc[i].shift(count, fill_value='00')

        # Concat after preprocessing (Total dataframe = concat(dataframe1, dataframe2))
        self.Data = pd.concat([self.Data_1, self.Data_2], axis=1)

    # Time interval function (Calc frequency if two packets have same ID & concat frequency frame and dataframe)
    def add_frequency_field(self):
        frequency = pd.DataFrame(columns=['Time_interval'])
        for i in range(int(len(self.Data))):
            select = 1
            count = i - 1
            if count < 0:
                select = 0
            else:
                while not (self.Data.ID[i] == self.Data.ID[count]):
                    count -= 1
                    if count < 0:
                        select = 0
                        break
            if select == 0:
                frequency.loc[i] = 0
            elif select == 1:
                data = float(self.Data.TimeStamp[i]) - float(self.Data.TimeStamp[count])
                frequency.loc[i] = data
            if i % 10000 == 0:
                print(i, 'freq computation is completed')
        self.Data = pd.concat([frequency, self.Data], axis=1)

# Transform raw data to pandas dataframe (Survival Dataset case)
class Survival_to_matrix:
    def __init__(self, path):
        self.row_data = pd.read_csv(path, sep=',', header=None, encoding="cp949")
        self.Data_1 = pd.DataFrame(columns=['TimeStamp', 'ID'])
        self.Data_2 = pd.DataFrame(columns=['Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8'])
        self.Data = Data = pd.DataFrame(
            columns=['TimeStamp', 'ID', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8', 'R/T'])

    def to_matrix(self):
        self.Data_1['TimeStamp'] = self.row_data[0]
        self.Data_1['ID'] = self.row_data[1]

        self.Data_2['Data1'] = self.row_data[3]
        self.Data_2['Data2'] = self.row_data[4]
        self.Data_2['Data3'] = self.row_data[5]
        self.Data_2['Data4'] = self.row_data[6]
        self.Data_2['Data5'] = self.row_data[7]
        self.Data_2['Data6'] = self.row_data[8]
        self.Data_2['Data7'] = self.row_data[9]
        self.Data_2['Data8'] = self.row_data[10]
        self.Data_2['R/T'] = self.row_data[11]

        temp = self.Data_2.isna()
        for i in (self.Data_2['R/T'].isna().loc[self.Data_2['R/T'].isna() == True].index):
            count = Counter(temp.loc[i] == True)[True]
            self.Data_2.loc[i] = self.Data_2.loc[i].shift(count, fill_value='00')

        self.Data = pd.concat([self.Data_1, self.Data_2], axis=1)

    # Normal class data formation is different from other, So I need this func
    def to_matrix_normal(self):
        self.Data_1['TimeStamp'] = self.row_data[0]
        self.Data_1['ID'] = '0' + self.row_data[1]

        Data_2_temp = pd.DataFrame(columns=['Data'])
        Data_2_temp['Data'] = self.row_data[3]
        Data_2_temp = Data_2_temp.Data.str.split(' ')

        self.Data_2['Data1'] = Data_2_temp.str.get(0)
        self.Data_2['Data2'] = Data_2_temp.str.get(1)
        self.Data_2['Data3'] = Data_2_temp.str.get(2)
        self.Data_2['Data4'] = Data_2_temp.str.get(3)
        self.Data_2['Data5'] = Data_2_temp.str.get(4)
        self.Data_2['Data6'] = Data_2_temp.str.get(5)
        self.Data_2['Data7'] = Data_2_temp.str.get(6)
        self.Data_2['Data8'] = Data_2_temp.str.get(7)

        # Fill in the blank(data section) of dataframe as '0x00' data if there are no data
        temp = self.Data_2.isna()
        for i in (self.Data_2['Data8'].isna().loc[self.Data_2['Data8'].isna() == True].index):
            count = Counter(temp.loc[i] == True)[True]
            self.Data_2.loc[i] = self.Data_2.loc[i].shift(count, fill_value='00')

        # Concat after preprocessing (Total dataframe = concat(dataframe1, dataframe2))
        self.Data = pd.concat([self.Data_1, self.Data_2], axis=1)

    # Time interval function (Calc frequency if two packets have same ID & concat frequency frame and dataframe)
    def add_frequency_field(self):
        frequency = pd.DataFrame(columns=['Time_interval'])
        for i in range(int(len(self.Data))):
            select = 1
            count = i - 1
            if count < 0:
                select = 0
            else:
                while (not (self.Data.ID[i] == self.Data.ID[count])):
                    count -= 1
                    if count < 0:
                        select = 0
                        break
            if select == 0:
                frequency.loc[i] = 0
            elif select == 1:
                data = float(self.Data.TimeStamp[i]) - float(self.Data.TimeStamp[count])
                frequency.loc[i] = data
            if (i % 10000 == 0):
                print(i, 'freq computation is completed')
        self.Data = pd.concat([frequency, self.Data], axis=1)

#  Transform dataframe to Tensor(RGBA Image)
class mk_img:
    # Define dictionary for dataframe matching
    id_dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12,'d': 13, 'e': 14, 'f': 15}
    ID_dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12,'D': 13, 'E': 14, 'F': 15}

    def __init__(self, size, start_num, end_num, data, path_to_w, dic):
        self.path = path_to_w
        self.start_num = start_num
        self.end_num = end_num
        self.Data = data
        self.size = size
        self.ch_1 = []
        self.ch_2 = []
        self.ch_3 = []
        self.ch_4 = []
        self.img_data = np.array([]).reshape(-1, 80, 4)
        self.Dic = dic

    def init_ch(self):
        self.ch_1 = []
        self.ch_2 = []
        self.ch_3 = []
        self.ch_4 = []

    # Define image shape
    def init_img(self):
        self.img_data = np.array([]).reshape(-1, 80, 4)

    def to_img_row(self, i):
        # Frequency Matching function
        def encoder(data):
            if (data >= 0) and (data < 0.001):
                return 0
            elif (data >= 0.001) and (data < 0.002):
                return 1
            elif (data >= 0.002) and (data < 0.004):
                return 2
            elif (data >= 0.004) and (data < 0.008):
                return 3
            elif (data >= 0.008) and (data < 0.016):
                return 4
            elif (data >= 0.016) and (data < 0.032):
                return 5
            elif (data >= 0.032) and (data < 0.064):
                return 6
            elif (data >= 0.064) and (data < 0.128):
                return 7
            elif (data >= 0.128) and (data < 0.256):
                return 8
            elif (data >= 0.256) and (data < 0.512):
                return 9
            elif (data >= 0.512) and (data < 1.024):
                return 10
            elif (data >= 1.024) and (data < 2.048):
                return 11
            elif (data >= 2.048) and (data < 4.096):
                return 12
            elif (data >= 4.096) and (data < 8.192):
                return 13
            elif (data >= 8.192) and (data < 16.384):
                return 14
            elif (data >= 16.384) and (data < 32.768):
                return 15
            else:
                return 15

        # Data one-hot encoding and push to array(Ch1~4)
        if (self.Dic == "Large"):
            self.ch_1.extend(tf.one_hot(encoder(self.Data['Time_interval'][i]), 16))
            self.ch_1.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data1'][i][0]], 16))
            self.ch_1.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data1'][i][1]], 16))
            self.ch_1.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data2'][i][0]], 16))
            self.ch_1.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data2'][i][1]], 16))

            self.ch_2.extend(tf.one_hot(mk_img.ID_dic[self.Data['ID'][i][1]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data3'][i][0]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data3'][i][1]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data4'][i][0]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data4'][i][1]], 16))

            self.ch_3.extend(tf.one_hot(mk_img.ID_dic[self.Data['ID'][i][2]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data5'][i][0]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data5'][i][1]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data6'][i][0]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data6'][i][1]], 16))

            self.ch_4.extend(tf.one_hot(mk_img.ID_dic[self.Data['ID'][i][3]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data7'][i][0]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data7'][i][1]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data8'][i][0]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.ID_dic[self.Data['Data8'][i][1]], 16))
            temp = (np.stack((self.ch_1, self.ch_2, self.ch_3, self.ch_4), axis=-1) * 255).reshape(1, 80, 4)
            return temp

        # Data one-hot encoding and push to array(Ch1~4)
        elif (self.Dic == "Small"):
            self.ch_1.extend(tf.one_hot(encoder(self.Data['Time_interval'][i]), 16))
            self.ch_1.extend(tf.one_hot(mk_img.id_dic[self.Data['Data1'][i][0]], 16))
            self.ch_1.extend(tf.one_hot(mk_img.id_dic[self.Data['Data1'][i][1]], 16))
            self.ch_1.extend(tf.one_hot(mk_img.id_dic[self.Data['Data2'][i][0]], 16))
            self.ch_1.extend(tf.one_hot(mk_img.id_dic[self.Data['Data2'][i][1]], 16))

            self.ch_2.extend(tf.one_hot(mk_img.id_dic[self.Data['ID'][i][1]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.id_dic[self.Data['Data3'][i][0]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.id_dic[self.Data['Data3'][i][1]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.id_dic[self.Data['Data4'][i][0]], 16))
            self.ch_2.extend(tf.one_hot(mk_img.id_dic[self.Data['Data4'][i][1]], 16))

            self.ch_3.extend(tf.one_hot(mk_img.id_dic[self.Data['ID'][i][2]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.id_dic[self.Data['Data5'][i][0]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.id_dic[self.Data['Data5'][i][1]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.id_dic[self.Data['Data6'][i][0]], 16))
            self.ch_3.extend(tf.one_hot(mk_img.id_dic[self.Data['Data6'][i][1]], 16))

            self.ch_4.extend(tf.one_hot(mk_img.id_dic[self.Data['ID'][i][3]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.id_dic[self.Data['Data7'][i][0]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.id_dic[self.Data['Data7'][i][1]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.id_dic[self.Data['Data8'][i][0]], 16))
            self.ch_4.extend(tf.one_hot(mk_img.id_dic[self.Data['Data8'][i][1]], 16))
            temp = (np.stack((self.ch_1, self.ch_2, self.ch_3, self.ch_4), axis=-1) * 255).reshape(1, 80, 4)
            return temp

    def to_An_img(self, k):
        self.init_img() #Initializing an image
        for i in range(self.size * k, self.size * (k + 1)):
            self.init_ch() #Initializing a channel
            self.img_data = np.concatenate((self.img_data, self.to_img_row(i))) # Image generation using temp

        # Warning: self.Path!!
        imageio.imwrite(self.path + self.path.split('/')[-2] + '_' + str(k+1) + '.png', np.uint8(self.img_data))
        print(k+1, "of image is completed")

    # Generating many images
    def to_img_set(self):
        for i in range(self.start_num, self.end_num):
            self.to_An_img(i)

def main():
    # OTIDS Data Set 사용
    # Data path
    data_path = "C:/~/CAN-intrusion-dataset(OTIDS)"
    Normal_path = data_path + "/Attack_free_dataset.txt"
    Fuzzy_path = data_path + "/Fuzzy_attack_dataset.txt"
    Imp_path = data_path + "/Impersonation_attack_dataset.txt"
    DoS_path = data_path + "/DoS_attack_dataset.txt"

    OTIDS_N = OTIDS_to_matrix(Normal_path) # 2369800 > 23600장 생성
    OTIDS_N.to_fr()
    OTIDS_N.to_matrix()
    OTIDS_N.add_frequency_field()

    OTIDS_F = OTIDS_to_matrix(Fuzzy_path)
    OTIDS_F.to_fr()
    OTIDS_F.to_matrix()
    OTIDS_F.add_frequency_field()

    OTIDS_I = OTIDS_to_matrix(Imp_path)
    OTIDS_I.to_fr()
    OTIDS_I.to_matrix()
    OTIDS_I.Data = OTIDS_I.Data[515438:].reset_index()
    OTIDS_I.add_frequency_field()

    OTIDS_D = OTIDS_to_matrix(DoS_path)
    OTIDS_D.to_fr()
    OTIDS_D.to_matrix()
    OTIDS_D.add_frequency_field()

    # Path to write
    base = "C:/~/Result/OTIDS"
    Normal_path_w = base + "/Normal/"
    Fuzzy_path_w = base + "/Fuzzy/"
    Imp_path_w = base + "/Imp/"
    DoS_path_w = base + "/DoS/"

    # Define params
    size = 100 # Image rows Number
    start = 0 # Default
    end = 4800 # The number of images to generate

    # Image generation
    Image_data = mk_img(size, start, end, OTIDS_I.Data, Imp_path_w, "Small") # Generating impersonation attack image
    Image_data.to_img_set()

###################################################################################
# main func
main()