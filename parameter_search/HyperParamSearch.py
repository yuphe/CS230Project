import numpy as np
import pandas as pd
from GPyOpt.methods import BayesianOptimization
from metrics.MeasureMetrics import printCM
from trainer.cnn_module import cnnMod

def bayesianSearch(wpath,Xtrain, Ytrain, Xval, Yval,Nepochs,batch_size,CNN_PARAMS,MaxIter):

    bds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.00001, 0.001)},
        {'name': 'weight_decay', 'type': 'continuous', 'domain': (0.00001, 0.01)},
        {'name': 'keepProb', 'type': 'continuous', 'domain': (0.0, 1)}
    ]

    def f1_score(xs):
        parameters = xs[0]
        CNN = cnnMod(wpath, SPP=CNN_PARAMS["SPP"],learning_rate=parameters[0],
                     weight_decay=parameters[1],keepProb=parameters[2])

        Ytrain_, YtrainPred, Yval_, YvalPred, convs = CNN.train(Xtrain, Ytrain, Xval, Yval,
                                                                Nepochs=Nepochs, batch_size=batch_size,
                                                                CONTINUE=False,
                                                                early_stop_round=CNN_PARAMS["early_stop_round"])

        ytrain_pred_prob = YtrainPred[:, 0]

        ytrain_pred_label = np.zeros(len(ytrain_pred_prob))
        for i in range(len(ytrain_pred_prob)):
            if ytrain_pred_prob[i] >= CNN_PARAMS['confidence']:
                ytrain_pred_label[i] = 1
            else:
                ytrain_pred_label[i] = 0

        ytrain_true_label = Ytrain_[:, 0]

        ytrain_true_label_pd = pd.Series(ytrain_true_label, name='Actual')
        ytrain_pred_label_pd = pd.Series(ytrain_pred_label, name='Predicted')

        _, trainF1Score = printCM(ytrain_true_label_pd, ytrain_pred_label_pd)

        yval_pred_prob = YvalPred[:, 0]
        yval_pred_label = np.zeros(len(yval_pred_prob))
        for i in range(len(yval_pred_prob)):
            if yval_pred_prob[i] >= CNN_PARAMS['confidence']:
                yval_pred_label[i] = 1
            else:
                yval_pred_label[i] = 0

        yval_true_label = Yval_[:, 0]

        yval_true_label_pd = pd.Series(yval_true_label, name='Actual')
        yval_pred_label_pd = pd.Series(yval_pred_label, name='Predicted')
        _, valF1Score = printCM(yval_true_label_pd, yval_pred_label_pd)

        score = valF1Score-np.abs(valF1Score-trainF1Score)
        #score = valF1Score
        return score

    optimizer = BayesianOptimization(f=f1_score,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.01,
                                     exact_feval=False,
                                     normalize_Y=False,
                                     maximize=True,
                                     num_cores=1,
                                     verbosity=3,
                                     initial_design_numdata=3,
                                     initial_design_type='random',
                                     acquisition_optimizer_type='lbfgs')

    optimizer.run_optimization(max_iter=MaxIter, verbosity=3)

    nameOptPara = [d['name'] for d in bds]
    valOptPara = np.array([i for i in optimizer.x_opt]).reshape(1, len(bds))
    OptPara = pd.DataFrame(valOptPara, columns=nameOptPara)

    return optimizer, OptPara