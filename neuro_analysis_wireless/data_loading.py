import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer



def load_datasets_Jacopo_v2(folders, labelsdict, removemodifiers=False):
    dflist = []
    labellist = []
    labellistkey = []
    modifiers = []
    modifiers_temp = []
    session_temp = []
    session = []
    condition=[]
    condition_temp=[]
    times_temp=[]
    times=[]
    for folder in folders: # loop over electrode arrays
        electrodearraydf = []
        for category in list(labelsdict.keys()): # loop over categories
            print(folder, '---', category)
            for filename in os.listdir(folder):
                #if category.lower() in filename.lower():   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! substituted for the next line
                if category.lower() == os.path.splitext(filename.lower())[0]:
                    f = os.path.join(folder, filename)
                    # Determine the file extension
                    _, file_extension = os.path.splitext(filename)
                    if file_extension == '.xlsx':
                        # Use openpyxl engine for .xlsx files
                        df = pd.read_excel(f, engine='openpyxl')
                    elif file_extension == '.csv':
                        # Use default engine for .csv files
                        df = pd.read_csv(f)
                    else:
                        print(f"Unsupported file format: {file_extension}")
                        continue # Skip this file if it's not .xlsx or .csv

                    # Check if 'Modifiers' or 'Behavior' column exists, if not, raise error
                    if 'Modifiers' in df.columns:
                        modifiers_temp = list(df['Modifiers'])
                        df.drop('Modifiers', axis=1, inplace=True)
                    elif 'Behavior' in df.columns:
                        modifiers_temp = list(df['Behavior'])
                        df.drop('Behavior', axis=1, inplace=True)
                    #else:
                    #    raise ValueError("Neither 'Modifiers' nor 'Behavior' columns are present in the DataFrame.")
                    if 'Session' in df.columns:
                        session_temp = list(df['Session'])
                        df.drop('Session', axis=1, inplace=True)
                    if 'Condition' in df.columns:
                        condition_temp=list(df['Condition'])
                        df.drop('Condition', axis=1, inplace=True)
                        
                    if 'Time' in df.columns:
                        times_temp = list(df['Time'])
                        df.drop('Time', axis=1, inplace=True)
                        
                    '''    
                    if removemodifiers:
                        # Check if 'Modifiers' or 'Behavior' column exists, if not, raise error
                        if 'Modifiers' in df.columns:
                            modifiers_temp = list(df['Modifiers'])
                            df.drop('Modifiers', axis=1, inplace=True)
                        elif 'Behavior' in df.columns:
                            modifiers_temp = list(df['Behavior'])
                            df.drop('Behavior', axis=1, inplace=True)
                        else:
                            raise ValueError("Neither 'Modifiers' nor 'Behavior' columns are present in the DataFrame.")
                        if 'Session' in df.columns:
                            session_temp = list(df['Session'])
                            df.drop('Session', axis=1, inplace=True)
                        if 'Condition' in df.columns:
                            condition_temp=list(df['Condition'])
                            df.drop('Condition', axis=1, inplace=True)
                    '''
                    
                    df = df.add_suffix('_' + folder.split('_')[-1]) # add name of the channel to each column of the dataframe
                    electrodearraydf.append(df)
                    if folder == folders[-1]: # make the labels vector during the last loop, i.e., the last folder
                        labelkey = category
                        labelkeyvec = [labelkey] * len(df)
                        labelvec = np.full(len(df), fill_value=labelsdict[labelkey])
                        labellist.append(labelvec.copy())
                        labellistkey.append(labelkeyvec)
                        modifiers.append(modifiers_temp)
                        session.append(session_temp)
                        condition.append(condition_temp)
                        times.append(times_temp)
        dflist.append(pd.concat(electrodearraydf, axis=0)) # make electrode array df
    return dflist, labellist, labellistkey, modifiers, session, condition, times



def load_and_normalize_datasets(folders, labelsdict, removemodifiers=False, normtype='standardscaler'):
    """
    Load and normalize datasets.

    Args:
        folders (list): List of folders containing the datasets.
        labelsdict (dict): Dictionary mapping labels to their keys.
        removemodifiers (bool, optional): Whether to remove modifiers. Defaults to False.
        normtype (str, optional): Normalization type, either 'maxabsscaler' or 'standardscaler'. Defaults to 'maxabsscaler'.

    Returns:
        df (pandas.DataFrame): Normalized dataframe.
        labels (numpy.ndarray): Array of labels.
        labelskeys (list): List of label keys.
        modifiers (list): List of modifiers.
        sessions (numpy.ndarray): Array of sessions.
        conditions (list): List of conditions.
    """
    dflist, labellist, labellistkey, modifierslist, sessionlist, conditionlist, timelist = load_datasets_Jacopo_v2(
        folders, labelsdict, removemodifiers=removemodifiers)
    df = pd.concat(dflist, axis=1)
    df.reset_index(drop=True, inplace=True)
    labels = [item for sublist in labellist for item in sublist]
    labels = np.array(labels)
    labelskeys = [item for sublist in labellistkey for item in sublist]
    modifiers = [item for sublist in modifierslist for item in sublist]
    sessions = [item for sublist in sessionlist for item in sublist]
    sessions = np.array(sessions)
    conditions = [item for sublist in conditionlist for item in sublist]
    col_names=list(df.columns)
    times = [item for sublist in timelist for item in sublist]

    if normtype == 'maxabsscaler':
        maxabsscaler = preprocessing.MaxAbsScaler()
        df_norm = maxabsscaler.fit_transform(df)
    elif normtype == 'standardscaler':
        standardscaler = preprocessing.StandardScaler()
        df_norm = standardscaler.fit_transform(df)
    elif normtype == None:
        df_norm=df
    else :
        raise ValueError('normtype not valid')
        
    return df_norm, labels, labelskeys, modifiers, sessions, conditions, col_names, times



def intra_session_normalization_and_modifiers(folders,labelsdict, normtype='standardscaler'):
    '''
    VERSION 2 of this function (21/01/2025)
    '''
    dflist, labellist, labellistkey, modifierslist, sessionlist, conditionlist, timelist =load_datasets_Jacopo_v2(folders,labelsdict,removemodifiers=True)
    df=pd.concat(dflist, axis=1)
    df.reset_index(drop=True, inplace=True)
    labels = [item for sublist in labellist for item in sublist]
    labels=np.array(labels)
    labelskeys= [item for sublist in labellistkey for item in sublist]
    modifiers = [item for sublist in modifierslist for item in sublist]
    sessions = [item for sublist in sessionlist for item in sublist]
    sessions=np.array(sessions) #to avoid some  problems regarding the type
    conditions=[item for sublist in conditionlist for item in sublist]
    col_names=list(df.columns)
    times = [item for sublist in timelist for item in sublist]

    df_session_list=[]
    labels2_list=[]
    labelskeys2_list=[]
    modifiers2_list=[]
    sessions2_list=[]
    conditions2_list=[]
    times2_list=[]
    
    
    if normtype=='globalnorm': 
        maxabsscaler=preprocessing.MaxAbsScaler()
        df_normalized=maxabsscaler.fit_transform(df)
        df_normalized=pd.DataFrame(data=df_normalized, columns=df.columns)
        modifiers=['None' if type(x)==float else x for x in modifiers]
        modifiers_multilabel_lists = [x.split(',') for x in modifiers]
        mlb = MultiLabelBinarizer()
        mlb.fit(modifiers_multilabel_lists)
        one_hot_array = mlb.transform(modifiers_multilabel_lists)
        df_modifiers = pd.DataFrame(one_hot_array, columns=mlb.classes_)
        return df_normalized, labels, labelskeys, df_modifiers, sessions, conditions, col_names, times


    for i in np.unique(sessions):#loop over sessions
        indexvec=sessions==i #get the indices of the elements from the relevant session
        df_session=df[indexvec]
        sessions2=sessions[indexvec]
        labels2=labels[indexvec]
        labelskeys2=[i for (i, v) in zip(labelskeys, indexvec) if v]
        modifiers2=[i for (i, v) in zip(modifiers, indexvec) if v]
        conditions2=[i for (i, v) in zip(conditions, indexvec) if v]

        times2=[i for (i, v) in zip(times, indexvec) if v]

        if normtype not in ['maxabsscaler','standardscaler','robustscaler', 'globalnorm', None]:
            raise ValueError("normalization type invalid" )
        if normtype=='maxabsscaler':
            #apply maxabsscaler to each session separately
            maxabsscaler=preprocessing.MaxAbsScaler()
            df_session=maxabsscaler.fit_transform(df_session)
        elif normtype=='standardscaler':
            standardscaler=preprocessing.StandardScaler()
            df_session=standardscaler.fit_transform(df_session)
        elif normtype=='robustscaler':
            robustscaler=preprocessing.RobustScaler()
            df_session=robustscaler.fit_transform(df_session)
        #append session to list
        df_session_list.append(df_session.copy())
        #modifiers and others variables
        labels2_list.extend(labels2.copy())
        labelskeys2_list.extend(labelskeys2.copy())
        modifiers2_list.extend(modifiers2.copy())
        sessions2_list.extend(sessions2.copy())
        conditions2_list.extend(conditions2.copy())
        times2_list.extend(times2.copy())
    
    #deal with the modifiers and create a binary dataframe
    modifiers3=['None' if type(x)==float else x for x in modifiers2_list]
    modifiers_multilabel_lists = [x.split(',') for x in modifiers3]
    mlb = MultiLabelBinarizer()
    mlb.fit(modifiers_multilabel_lists)
    one_hot_array = mlb.transform(modifiers_multilabel_lists)
    df_modifiers = pd.DataFrame(one_hot_array, columns=mlb.classes_)

    
    df_intrasessionnorm=np.concatenate(df_session_list, axis=0)
    df_intrasessionnorm=pd.DataFrame(data=df_intrasessionnorm, columns=df.columns)

    return df_intrasessionnorm, np.array(labels2_list), labelskeys2_list, df_modifiers, sessions2_list, conditions2_list, col_names, times2_list



def transform_column_names(input_string):
    '''
    transform column names
    FROM ---> TO
    U_133_Ch_119_activity\45a -> A_45a_U_133_Ch_119
    U_134_Ch_120_activity\45a -> A_45a_U_134_Ch_120
    U_53_Ch_67_activity\46v -> A_46v_U_53_Ch_67
    U_59_Ch_71_activity\46v -> A_46v_U_59_Ch_71
    U_110_Ch_23_activity\F5hand -> A_F5hand_U_110_Ch_23
    U_144_Ch_3_activity\F5hand -> A_F5hand_U_144_Ch_3
    U_31_Ch_35_activity\F5mouth -> A_F5mouth_U_31_Ch_35
    U_34_Ch_40_activity\F5mouth -> A_F5mouth_U_34_Ch_40
    '''
    # Match the pattern and extract relevant groups
    match = re.match(r'(U_\d+_Ch_\d+)_activity\\([a-zA-Z0-9]+)', input_string)
    if match:
        part1 = match.group(1)  # U_... part
        part2 = match.group(2)  # The part after activity\
        return f"A_{part2}_{part1}"
    else:
        raise ValueError(f"Input string does not match the expected format: {input_string}")
