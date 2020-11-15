# working directory
LOT_FOLDER = 'D:/ML_DATA/DeepLearning/Intel_MIMOP_W093_W096_train/'
# input image channels
LOT_CH = ['Defective','Difference','Reference']
# training image channels
IMG_CH = ['Ref','Def','Diff']
# grouping/binning for training
BinCode = {
    1 : [366],
    2 :  [99]
}

# parameters for cnn training
CNN_PARAMS = {
    "SPP" : True,
    "early_stop_round": 0,
    "confidence" : 0.1
}

# printing hidden layer images
Defect_ID = [19020000,20380000]

