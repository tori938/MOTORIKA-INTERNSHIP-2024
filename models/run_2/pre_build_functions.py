import pandas as pd, numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn import ensemble
from sklearn import svm
from sklearn import metrics

import scipy.signal


#write a function to read the palm data file
def read_omg_csv(path_palm_data: str, 
                 n_omg_channels: int, 
                 n_acc_channels: int = 0, 
                 n_gyr_channels: int = 0, 
                 n_mag_channels: int = 0, 
                 n_enc_channels: int = 0,
                 button_ch: bool = True, 
                 sync_ch: bool = True, 
                 timestamp_ch: bool = True) -> pd.DataFrame:
    
    '''
    Reads CSV data for OMG data
    NB: data must be separated by " " separator

        Parameters:
                path_palm_data  (str): path to csv data file
                n_omg_channels  (int): Number of OMG channels
                n_acc_channels  (int): Number of Accelerometer channels, default = 0
                n_gyr_channels  (int): Number of Gyroscope channels, default = 0
                n_mag_channels  (int): Number of Magnetometer channels, default = 0
                n_enc_channels  (int): Number of Encoder channels, default = 0
                button_ch      (bool): If button channel is present, default = True
                sync_ch        (bool): If synchronization channel is present, default = True
                timestamp_ch   (bool): If timestamp channel is present, default = True

        Returns:
                df_raw (pd.DataFrame): Parsed pandas Dataframe with OMG data
    '''
    
    df_raw = pd.read_csv(path_palm_data, sep=' ', 
                         header=None, 
                         skipfooter=1, 
                         skiprows=1, 
                         engine='python')
    columns = np.arange(n_omg_channels).astype('str').tolist()
    
    for label, label_count in zip(['ACC', 'GYR', 'MAG', 'ENC'], 
                                  [n_acc_channels, n_gyr_channels, n_mag_channels, n_enc_channels]):
        columns = columns + ['{}{}'.format(label, i) for i in range(label_count)]
        
    if button_ch:
        columns = columns + ['BUTTON']
        
    if sync_ch:
        columns = columns + ['SYNC']
        
    if timestamp_ch:
        columns = columns + ['ts']
        
    df_raw.columns = columns
    
    return df_raw


#create a generator for the number of files in the directory
def length_of_file_number(length):
    while True:
        try:
            for n in range(0, length):
                yield n
        except StopIteration:
            break
        

#write a file to return the relevant palm file for the relevant pilot
def read_pilot(data: pd.DataFrame,
               file_number: int):
    
    '''
    Reads meta data to return the relevant palm file for the pilot

        Parameters:
                data (DataFrame): meta information dataframe, dataframe with each pilot's palm file
                file_number (int): palm file, i.e. vary in number depending on the pilot
                
        Returns:
                merged (pd.DataFrame): parsed Dataframe with the relevant - single - palm file for the relevant pilot
                current file (str): name of the palm file
                palm file (str): name of the palm file and its folder extension (path)  
    '''
    
    #return the palm files as a list
    files = data['montage'].tolist()
    
    #create an empty list to store each file
    dfs = []

    #create an empty list to store sliced file names
    lst_of_fx = []

    #save the current file name
    current_file = ''
    palm_file = ''

    #loop through the list of file names
    for f in files:
        current_file = files[file_number]
        palm_file = './data_palm/' + current_file
        #check for correct extension
        lst_of_fx.append('./data_palm/' + f)

    #return the right file
    for l in range(0, len(lst_of_fx)):
        if l == file_number:
            read_file = lst_of_fx[l]
            df = read_omg_csv(read_file,
                              n_omg_channels=50,
                              n_acc_channels=3,
                              n_gyr_channels=3,
                              n_enc_channels=6,
                              n_mag_channels=0
                          )
            dfs.append(df)
            
    #merge the files into one file
    merged = pd.concat(dfs,
                       ignore_index=True)
    
    #return the relevant variables
    return merged, current_file, palm_file


#create the encoder function
def encoder(palm_file, gest_protocol):
    le = LabelEncoder()

    #FIT
    le.fit(
        gest_protocol[[
            'Thumb', 'Index', 'Middle', 'Ring', 'Pinky',
            'Thumb_stretch', 'Index_stretch', 'Middle_stretch', 'Ring_stretch', 'Pinky_stretch'
        ]]
        .apply(lambda row: str(tuple(row)), axis=1)
    )

    #TRANSFORM
    gest_protocol['gesture'] = le.transform(
        gest_protocol[[
            'Thumb', 'Index', 'Middle', 'Ring', 'Pinky',
            'Thumb_stretch', 'Index_stretch', 'Middle_stretch', 'Ring_stretch', 'Pinky_stretch'
        ]]
        .apply(lambda row: str(tuple(row)), axis=1)
    )
    return gest_protocol


#create a shift function
def get_naive_centering(
    X_arr, y_arr, gap=500, inter=1000,
    window=20, use_m=True, model=svm.SVC(), 
    return_metrics=False):
    """Функция для устранения глобального лага между сигналами датчиков и таргетом.

    Args:
        X_arr (ndarray): Массив данных.
        y_arr (ndarray): Вектор целевого признака.
        gap (int, optional): Размеры концевых отступов. Defaults to 500.
        inter (int, optional): Величина концевых выборок. Defaults to 1000.
        window (int, optional): Величина окна поиска оптимального сдвига. Defaults to 20.
        use_m (bool, optional): Использование модели для поиска оптимального сдвига.
            Defaults to True. False: поиск сдвига по корреляции таргета с вектором
            суммы модулей дифференциалов векторов признаков массива данных.
        model (_type_, optional): Алгоритм scikit-learn. Defaults to svm.SVC().
        return_metrics (bool, optional): Взвращение значений метрик

    Returns:
        tuple():
            ndarray: Вектор сдвинутого таргета.
            float: метрика на начальном участке.
            float: метрика на конечном участке.
        tuple():
            ndarray: Вектор сдвинутого таргета.
            list: Строки отчета по проделанным операциям.
    """
    # part of the data from the beginning
    X_part1 = X_arr[gap:gap+inter]
    y_part1 = y_arr[gap:gap+inter]
    # part of the data from the end
    X_part2 = X_arr[-gap-inter:-gap]
    y_part2 = y_arr[-gap-inter:-gap]
    
    # Функция для сдвига таргета
    def shifter(y_arr, shift=1):
        first_element = y_arr[0]
        prefix = np.full(shift, first_element)
        y_arr_shifted = np.concatenate((prefix, y_arr))[:-shift]
    
        return y_arr_shifted
    
    # Функция для расчета точности модели
    def get_score(X, y, model=model):
        model = model
        model.fit(X, y)
        preds = model.predict(X)
        
        return metrics.accuracy_score(y, preds)
    
    # Функция для расчета корреляции
    def get_corr(X, y):
        x_diff = pd.DataFrame(X).diff().abs().sum(axis=1)
        correlation = np.corrcoef(x_diff, y)[0, 1]
        
        return abs(correlation)
    
    
    max_score1, current_score1 = 0, 0
    max_score2, current_score2 = 0, 0
    s1, s2 = 1, 1
    
    for i in range(1, window+1):
        y_a = shifter(y_part1, shift=i)
        y_b = shifter(y_part2, shift=i)
        
        if use_m:
            current_score1 = get_score(X_part1, y_a)
            current_score2 = get_score(X_part2, y_b)
        else:
            current_score1 = get_corr(X_part1, y_a)
            current_score2 = get_corr(X_part2, y_b)
        
        if current_score1 > max_score1:
            max_score1, current_score1 = current_score1, max_score1
            s1 = i
        
        if current_score2 > max_score2:
            max_score2, current_score2 = current_score2, max_score2
            s2 = i
    
    optimal_shift = round((s1+s2)/2)
    y_arr_shifted = shifter(y_arr, shift=optimal_shift)
    summary = [
        f'Оптимальные свдиги для концевых выборок:   {s1} и {s2}\n',
        f'Accuracy/correlation на концевых выборках: {max_score1}; {max_score2}\n',
        f'Размер оптимального сдвига (как среднего): {optimal_shift}'
    ]
    
    if return_metrics:
        return y_arr_shifted, max_score1, max_score2
    
    return y_arr_shifted, summary


#filtration function
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def filter_signal(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y