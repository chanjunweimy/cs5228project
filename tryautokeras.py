import pandas as pd
import numpy as np
import autokeras as ak

def load_dataframe(id):
    train_data = np.load("train/train/{}.npy".format(id))
    return pd.DataFrame(data=train_data)

def get_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def clear_missing_data(df, missing_indices):
    df = df.drop(missing_indices,1)
    # for col in df_temp.columns:
    #    df = df.drop(df.loc[df[col].isnull()].index)
    for col in df.columns:
        count = df[col].isnull().sum().max()
        df[col] = df[col].fillna(np.mean(df[col])) 
        df[col] = df[col].fillna(0) # if np mean is NA, then replace 0 
    return df

def pad_data(dfs):
    data = []
    ROW_SIZE = 200
    for i in range(len(dfs)):
        df = dfs[i]
        diff = ROW_SIZE - df.shape[0]
        if diff > 0:
            halfDiff = int(diff / 2)
            df = np.pad(df, [(0,diff), (0,0)], 'constant')
        else:
            df = df[:ROW_SIZE]
        data.append(df)
    data = np.stack(data)
    return data

df_train = pd.read_csv('train_kaggle.csv')
y = df_train['label']
y = y.values

df8 = load_dataframe(8)
missing_df8 = get_missing_data(df8)
missing_indices = []
missing_indices = missing_df8[missing_df8['Percent'] > 0.8].index # .append(pd.Index([11,33,35]))

df_train_all = np.load('df_train_all.npy',allow_pickle=True)
XTrain_pad = pad_data(df_train_all)
XTrain = np.delete(XTrain_pad, missing_indices, 2)

# Build model and train.
automodel = ak.AutoModel(
   inputs=[ak.ImageInput()],
   outputs=[ak.ClassificationHead(loss='binary_crossentropy',
                                  metrics=['accuracy'])])
automodel.fit([XTrain],
              [y],
              validation_split=0.2)