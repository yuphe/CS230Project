import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def sampleCodes(fpath,CClist):
    try:
        DAData = pd.read_csv(fpath + 'Defects.csv')
    except FileNotFoundError:
        print("Defects.csv does not exist")

    DAData = DAData[['Id','ClassCode_Manual']]

    for key,value in CClist.items():
        CCData = DAData.loc[DAData['ClassCode_Manual'].isin(value)]
        CCData.loc[CCData.ClassCode_Manual !=np.int64(key),"ClassCode_Manual" ]=np.int64(key)
        if key == 1:
            pdData = CCData
        else:
            pdData = pd.concat([pdData,CCData],axis=0,ignore_index=True).sort_values(by=['Id'])

    return pdData


def get_File(id, ch, files):
    for f in files:
        if str(id) in f and ch in f:
            return f


def get_image(path):
    patch_image = cv2.imread(path,flags=-1)
    norm_image = cv2.normalize(patch_image, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_16UC1)
    return norm_image


def mapImage(fpath, ch, DAdata):
    files = os.listdir(fpath)
    for c in ch:
        col1 = c + 'File'
        col2 = c + 'Image'
        DAdata[col1] = DAdata.Id.map(lambda id: get_File(id, c, files))
        DAdata[col2] = DAdata[col1].map(lambda f: get_image(fpath + f))
    return DAdata


def convertImage(imgIN,imgCH):
    DAimg = imgIN[['Id','ClassCode_Manual']].reset_index(drop=True)
    for c in imgCH:
        if c == 'Ref':
            DAimg = pd.concat([DAimg, imgIN['ReferenceImage'].reset_index(drop=True)], axis=1)
            DAimg.rename(columns={"ReferenceImage": "Ref"}, inplace=True)
        elif c == 'Def':
            DAimg = pd.concat([DAimg, imgIN['DefectiveImage'].reset_index(drop=True)], axis=1)
            DAimg.rename(columns={"DefectiveImage": "Def"}, inplace=True)
        elif c == 'Diff':
            diff = imgIN['DefectiveImage']-imgIN['ReferenceImage']
            DAdiff = pd.DataFrame(diff.reset_index(drop=True),columns=['Diff'])
            DAimg = pd.concat([DAimg, DAdiff], axis=1)
        else:
            print("No image channel!")
    return DAimg

def stratifiedSplit(Data,frac):

    CC = list(set(Data["ClassCode_Manual"]))

    for i in range(len(CC)):
        CCData = Data.loc[Data['ClassCode_Manual'].isin([CC[i]])]
        train, test = train_test_split(CCData, test_size=frac)
        if i == 0:
            pdTrain = train
            pdTest = test
        else:
            pdTrain= pd.concat([pdTrain,train],axis=0,ignore_index=True).sort_values(by=['Id'])
            pdTest = pd.concat([pdTest, test], axis=0, ignore_index=True).sort_values(by=['Id'])

    return pdTrain, pdTest

def XYsplit(data1,data2):
    Xdata1 = data1.loc[:, ~data1.columns.isin(["Id","ClassCode_Manual"])]
    Ydata1 = data1.loc[:, data1.columns.isin(["ClassCode_Manual"])]
    # convert pd to tensor
    XdataTensor1, YdataTensor1 = toTensor(Xdata1,Ydata1)

    if len(data2) == 0:
        return XdataTensor1, YdataTensor1

    else:
        Xdata2 = data2.loc[:, ~data2.columns.isin(["Id","ClassCode_Manual"])]
        Ydata2 = data2.loc[:, data2.columns.isin(["ClassCode_Manual"])]
        # convert pd to tensor
        XdataTensor2, YdataTensor2 = toTensor(Xdata2,Ydata2)

        return XdataTensor1,YdataTensor1,XdataTensor2,YdataTensor2


def toTensor(pdX,pdY):
    num_samples = pdX.shape[0]
    num_ch = pdX.shape[1]
    ch = list(pdX)
    image_shape = (32, 32, num_ch)
    images = np.zeros((num_samples,) + image_shape).astype(np.float32)

    classList = np.array(sorted(pdY.ClassCode_Manual.unique()))
    num_class = classList.shape[0]
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = classList.reshape(num_class, 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    label = pdY["ClassCode_Manual"].values
    labelVector = np.zeros((num_samples, num_class)).astype(np.float32)

    for i in range(num_samples):
        for j in range(num_ch):
            images[i, :, :, j] = pdX[ch[j]][i]

        for k in range(num_class):
            if label[i] == classList[k]:
                labelVector[i] = onehot_encoded[k]

    return images, labelVector




