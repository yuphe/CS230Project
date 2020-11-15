import math
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import prettytable
from io import StringIO

def GetBPPROC(test_label, pred, pos_label=1, type=1):
    fpr, capR, thresholds = metrics.roc_curve(test_label, pred, pos_label=pos_label,drop_intermediate=False)
    totDOI = np.sum(test_label)
    totNUI =len(test_label)- totDOI
    NuiR = (fpr*totNUI)/(capR*totDOI + fpr*totNUI)
    if (type==1):
        return NuiR,capR,thresholds
    else:
        DoiY = capR*totDOI
        NuiX = fpr*totNUI
        return NuiX,DoiY,thresholds


def printCM(pred_label,real_label):
    f1Score = metrics.f1_score(real_label, pred_label, average='binary')
    cm_pd = pd.crosstab(real_label, pred_label,rownames=['Actual'], colnames=['Predicted'],margins=True)
    output = StringIO()
    cm_pd.to_csv(output)
    output.seek(0)
    tab = prettytable.from_csv(output)
    return tab,f1Score


def saveROC(fpath,pred_label,pred_prob,real_label):
    valuesROC = sorted(zip(pred_label, pred_prob, real_label), key=lambda x: x[1] * -1)
    headerROC = ['pred', 'conf', 'truth']
    oobROC = pd.DataFrame(valuesROC, columns=headerROC)
    oobROC.to_csv(fpath+'RocCountPlot.csv', encoding='utf-8', index=False)

    tempROC = pd.DataFrame(valuesROC)
    xNui, yDoi,thresholds = GetBPPROC(np.ravel(tempROC.loc[:, 2]), np.ravel(tempROC.loc[:, 1]), pos_label=1, type=1)
    xNuiCount, yDoiCount,thresholds = GetBPPROC(np.ravel(tempROC.loc[:, 2]), np.ravel(tempROC.loc[:, 1]), pos_label=1,type=0)
    Roc = pd.DataFrame(list(zip(xNui,yDoi,thresholds)))
    Roc.to_csv(fpath+"Roc.csv",encoding='utf-8',index=False,header=False)
    Count = pd.DataFrame(list(zip(xNuiCount,yDoiCount,thresholds)))
    Count.to_csv(fpath+"Count.csv",encoding='utf-8',index=False,header=False)



def visualImg(fpath,convs,imgIdx,sampleId):

    for j in range(len(imgIdx)):
        for k in range(len(convs)):
            filters = convs[k].shape[3]
            plt.figure(1)
            if filters < 8:
                n_columns = filters
                n_rows = 1
            else:
                n_columns = 8
                n_rows = math.ceil(filters / n_columns) + 1
            for i in range(filters):
                plt.subplot(n_rows, n_columns, i + 1)
                #plt.axis('off')
                plt.imshow(convs[k][imgIdx[j], :, :, i], interpolation="nearest", cmap="gray")

            plt.savefig(fpath+str(sampleId[j])+"_layer"+str(k+1)+".png")
    plt.close()