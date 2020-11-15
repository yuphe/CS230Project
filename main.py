import os
import warnings
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from image_loader.DataProcess import sampleCodes,mapImage,convertImage
from image_loader.DataProcess import stratifiedSplit,XYsplit
from trainer.cnn_module import cnnMod
from metrics.MeasureMetrics import printCM,saveROC,visualImg
from parameter_search.HyperParamSearch import bayesianSearch
import InitInput

warnings.filterwarnings('ignore')

def main(choice,*argv):

    LOT_FOLDER = InitInput.LOT_FOLDER
    LOT_FOLDER_RESULTS = LOT_FOLDER + "Results/"
    os.makedirs(os.path.dirname(LOT_FOLDER_RESULTS), exist_ok=True)

    LOT_CH = InitInput.LOT_CH
    IMG_CH = InitInput.IMG_CH
    BinCode = InitInput.BinCode
    CNN_PARAMS = InitInput.CNN_PARAMS
    visualDefId = np.array(InitInput.Defect_ID, dtype=np.int64)


    if choice == 1:
        FracVal = float(input("Fraction of validation size?"))
        print("Processing images...")
        DAdata = sampleCodes(LOT_FOLDER, BinCode)
        ImgPath = LOT_FOLDER + 'Archive/'
        imgData = mapImage(ImgPath, LOT_CH, DAdata)
        convtImg = convertImage(imgData, IMG_CH)

        IMG_FOLDER_train = LOT_FOLDER_RESULTS + "TrainData/"
        IMG_FOLDER_val = LOT_FOLDER_RESULTS + "ValData/"
        os.makedirs(os.path.dirname(IMG_FOLDER_train), exist_ok=True)
        os.makedirs(os.path.dirname(IMG_FOLDER_val), exist_ok=True)

        if FracVal == 0.0:
            Data_train, Data_val = convtImg, []
            Data_train.to_csv(IMG_FOLDER_train + 'Defects_train.csv', encoding='utf-8', index=False)
            Xtrain, Ytrain  = XYsplit(Data_train, Data_val)

            ftrain = h5py.File(IMG_FOLDER_train + 'train.hdf5', "w")
            ftrain["x"] = Xtrain
            ftrain["y"] = Ytrain
        else:
            Data_train, Data_val = stratifiedSplit(convtImg, FracVal)
            Data_train.to_csv(IMG_FOLDER_train + 'Defects_train.csv', encoding='utf-8', index=False)
            Data_val.to_csv(IMG_FOLDER_val + 'Defects_val.csv', encoding='utf-8', index=False)

            Xtrain, Ytrain, Xval, Yval = XYsplit(Data_train, Data_val)

            ftrain = h5py.File(IMG_FOLDER_train + 'train.hdf5', "w")
            ftrain["x"] = Xtrain
            ftrain["y"] = Ytrain

            fval = h5py.File(IMG_FOLDER_val + 'val.hdf5', "w")
            fval["x"] = Xval
            fval["y"] = Yval

        print("Training and validation data are ready.")

    elif choice == 2:

        ftrain = h5py.File(LOT_FOLDER_RESULTS + 'TrainData/train.hdf5', "r")
        Xtrain = np.array(ftrain["x"])
        Ytrain = np.array(ftrain["y"])

        fval = h5py.File(LOT_FOLDER_RESULTS + 'ValData/val.hdf5', "r")
        Xval = np.array(fval["x"])
        Yval = np.array(fval["y"])

        print("Number of training data = ", Ytrain.shape[0])

        Nepochs = int(input("Number Epochs ?"))
        batch_size = int(input("Batch Size ?"))
        numIter = int(input("How many iteration?"))

        _, bestParams = bayesianSearch(LOT_FOLDER_RESULTS, Xtrain, Ytrain,
                                            Xval, Yval, Nepochs, batch_size, CNN_PARAMS, numIter)

        bestParams.to_csv(LOT_FOLDER_RESULTS + "paramsOpt.csv", encoding='utf-8', index=False)

        print("Hyperparameter search is done.")


    elif choice == 3:

        JobQ = input("A: Continue or B: Restart ?")
        JobQ = JobQ.upper()
        if JobQ == "A":
            chkPath = os.path.exists(LOT_FOLDER_RESULTS+"models/")
            if chkPath == True:
                CONTINUE = True
            else:
                print("\n Model does not exist!")
                exit(0)
        elif JobQ == "B":
            CONTINUE = False
            try:
                HyperPara = pd.read_csv(LOT_FOLDER_RESULTS + "paramsOpt.csv")
                paraName = HyperPara.columns.values
                paraValue = HyperPara.values[0]
                HyperPara = {paraName[i]:paraValue[i] for i in range(len(paraName))}
                print(HyperPara)

            except FileNotFoundError:
                print("paramsOpt.csv does not exist")
                return
        else:
            print("\n Input [A] or [B] !")
            return 0



        ftrain = h5py.File(LOT_FOLDER_RESULTS+'TrainData/train.hdf5', "r")
        Xtrain = np.array(ftrain["x"])
        Ytrain = np.array(ftrain["y"])

        fval = h5py.File(LOT_FOLDER_RESULTS+'ValData/val.hdf5', "r")
        Xval = np.array(fval["x"])
        Yval = np.array(fval["y"])

        print("Number of training data = ",Ytrain.shape[0])

        Nepochs = int(input("Number Epochs ?"))
        batch_size = int(input("Batch Size ?"))


        saveImg = input("Save hidden layer images (Yes or No)?")
        saveImg = saveImg.upper()
        HID_IMG_FOLDER = LOT_FOLDER_RESULTS + "LayerImages/"
        os.makedirs(os.path.dirname(HID_IMG_FOLDER), exist_ok=True)

        if saveImg == "YES":
            DAtrain = pd.read_csv(LOT_FOLDER_RESULTS+'TrainData/Defects_train.csv')
            imgIdx = DAtrain[DAtrain['Id'].isin(visualDefId)].index.tolist()

            for j in range(len(imgIdx)):
                for i in range(Xtrain.shape[3]):
                    plt.subplot(1, Xtrain.shape[3], i + 1)
                    plt.title(list(DAtrain)[2+i])
                    plt.imshow(Xtrain[imgIdx[j], :, :, i], interpolation="nearest", cmap="gray")
                plt.savefig(HID_IMG_FOLDER+str(visualDefId[j])+"_Original.png")
            plt.close()

        elif saveImg == "NO":
            visualDefId= []
        else:
            print("\n Input [YES] or [NO] !")
            return 0

        CNN = cnnMod(LOT_FOLDER_RESULTS,SPP=CNN_PARAMS["SPP"], weight_decay=HyperPara["weight_decay"],
                     learning_rate=HyperPara["learning_rate"],keepProb=HyperPara["keepProb"])

        Ytrain_,YtrainPred,Yval_,YvalPred,convs = CNN.train(Xtrain, Ytrain,Xval,Yval,
                                                               Nepochs=Nepochs, batch_size=batch_size,CONTINUE=CONTINUE,
                                                               early_stop_round=CNN_PARAMS["early_stop_round"])


        if len(visualDefId) != 0:
            visualImg(HID_IMG_FOLDER,convs,imgIdx,visualDefId)

        ytrain_pred_prob = YtrainPred[:,0]

        ytrain_pred_label =  np.zeros(len(ytrain_pred_prob))
        for i in range(len(ytrain_pred_prob)):
            if ytrain_pred_prob[i] >= CNN_PARAMS['confidence']:
                ytrain_pred_label[i] = 1
            else:
                ytrain_pred_label[i] = 0

        ytrain_true_label = Ytrain_[:,0]

        ytrain_true_label_pd = pd.Series(ytrain_true_label, name='Actual')
        ytrain_pred_label_pd= pd.Series(ytrain_pred_label, name='Predicted')

        fpathTrain =LOT_FOLDER_RESULTS +"Train_ROC/"
        os.makedirs(os.path.dirname(fpathTrain), exist_ok=True)
        saveROC(fpathTrain, ytrain_pred_label, ytrain_pred_prob,ytrain_true_label )

        print("Training Confusion Matrix:")
        trainCM,trainF1Score = printCM(ytrain_true_label_pd, ytrain_pred_label_pd)
        print(trainCM)
        print("Training F1 score = ",trainF1Score)

        with open(LOT_FOLDER_RESULTS+"trainCM.txt",'w') as f1:
            print(trainCM, file = f1)

        yval_pred_prob = YvalPred[:,0]
        yval_pred_label = np.zeros(len(yval_pred_prob))
        for i in range(len(yval_pred_prob)):
            if yval_pred_prob[i] >= CNN_PARAMS['confidence']:
                yval_pred_label[i] = 1
            else:
                yval_pred_label[i] = 0

        yval_true_label = Yval_[:,0]

        yval_true_label_pd = pd.Series(yval_true_label, name='Actual')
        yval_pred_label_pd = pd.Series(yval_pred_label, name='Predicted')


        fpathVal =LOT_FOLDER_RESULTS +"Val_ROC/"
        os.makedirs(os.path.dirname(fpathVal), exist_ok=True)
        saveROC(fpathVal, yval_pred_label, yval_pred_prob,yval_true_label )

        print("Validation Confusion Matrix:")
        valCM,valF1Score = printCM(yval_true_label_pd, yval_pred_label_pd)
        print(valCM)
        print("Validation F1 score = ", valF1Score)

        with open(LOT_FOLDER_RESULTS+"ValCM.txt",'w') as f2:
            print(valCM, file=f2)

        print("Model training is done.")


if __name__ == "__main__":
    ans = True
    while ans:
        print("""
                1.Split data for training and validation
                2.Search Hyperparameter
                3.Train model
                4.Exit/Quit
                """)
        ans = input("What would you like to do?")
        if ans == "1":
            main(1)
        elif ans == "2":
            main(2)
        elif ans == "3":
            main(3)
        elif ans == "4":
            print("GOOD BYE!")
            ans = False
        elif ans != "":
            print("\n Not Valid Choice Try again.")
