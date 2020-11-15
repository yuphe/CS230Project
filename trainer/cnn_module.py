import time
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from arch.cnn_arch import network_2 as network
import pandas as pd

def performance(y_pred, y):
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def cal_cost(logits, y_pred, y,use_logits=True):
    if use_logits:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    else:
        cross_entropy = -tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='cost')


class cnnMod:

    def __init__(self,fpath,SPP=False,weight_decay=0,learning_rate=0.01,keepProb=0.7):
        self.wd = weight_decay
        self.SPP =SPP
        self.learning_rate = learning_rate
        self.fpath = fpath
        self.keepProb = keepProb

    def saveVars(self):
        vars_to_train = tf.trainable_variables()
        vars_for_bn1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_1/bn')
        vars_for_bn2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_2/bn')
        vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
        vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))
        saver = tf.train.Saver(vars_to_train)
        return saver,vars_to_train


    def train(self,Xtrain,Ytrain,Xtest,Ytest,Nepochs=50,batch_size=50,
              CONTINUE=False,early_stop_round=0):
        print("Training Start...")
        #tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
        tf.reset_default_graph()
        with tf.variable_scope("Training"):
            with tf.Graph().as_default():
                Nbatches = int(Ytrain.shape[0] / batch_size)
                print("Number of batches = ",Nbatches)
                w, h, c = Xtrain.shape[1:]
                nclasses = Ytrain.shape[1]
                x = tf.placeholder(tf.float32, [None, w, h, c])
                y_ = tf.placeholder(tf.float32, [None, nclasses])
                keep_prob = tf.placeholder(tf.float32)
                phase_train = tf.placeholder(tf.bool, name='phase_train')

                ### choose network
                conv1, y_pred, logits= network(x, w, h, c, nclasses, keep_prob, phase_train,
                                                              self.wd, self.SPP)
                accuracy = performance(y_pred, y_)
                cost = cal_cost(logits, y_pred, y_,use_logits=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                   beta1=0.9, beta2=0.999, epsilon=1e-08)
                global_step = tf.Variable(0, name='global_step', trainable=True)
                train_op = optimizer.minimize(cost, global_step=global_step)

                ### save and print trainable variables
                saver,vars = self.saveVars()
                vars_name = [v.name for v in vars if 'conv_1/weight' in v.name]

                init = tf.global_variables_initializer()

                LossTrain = []
                LossTest =[]
                Iepoch = []

                with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                    start_time = time.time()
                    writer = tf.summary.FileWriter(self.fpath+'graphs', sess.graph)
                    sess.run(init)
                    if CONTINUE == True:
                        saver.restore(sess, self.fpath+'models/simple_cnn.ckpt')
                    for epoch in range(Nepochs):
                        epoch_cost = 0
                        if epoch % 10 == 0:
                             save_path = saver.save(sess, self.fpath+'models/simple_cnn.ckpt')
                             print("Model saved in file: %s" % save_path)
                        idx = np.random.permutation(Ytrain.shape[0])
                        for batch in range(Nbatches):
                            batch_idx = idx[batch * batch_size:(batch + 1) * batch_size]
                            batch_xs = Xtrain[batch_idx, :, :, :]
                            batch_ys = Ytrain[batch_idx]
                            batch_fd = {x: batch_xs, y_: batch_ys, keep_prob:self.keepProb, phase_train: True}
                            _, train_cost, train_accuracy = sess.run([train_op, cost, accuracy], feed_dict=batch_fd)

                            current_time = time.time()
                            duration = float((current_time - start_time))
                            start_time = current_time
                            print('epoch-batch, loss, accuracy, duration = '
                                  '%3d-%6d: %8.4f, %8.4f, %.3f s' %
                                  (epoch, batch, train_cost, train_accuracy, duration))

                        test_fd = {x: Xtest, y_: Ytest, keep_prob:1.0, phase_train: False}

                        YtestPred, test_cost, test_accuracy = sess.run([y_pred, cost, accuracy],feed_dict=test_fd)
                        LossTrain.append(train_cost)
                        LossTest.append(test_cost)
                        Iepoch.append(epoch)
                        print('Test accuracy :',test_accuracy)


                        if epoch > early_stop_round and early_stop_round > 0 :
                            signTest,_ = np.polyfit(Iepoch[-early_stop_round:],LossTest[-early_stop_round:],1)
                            if signTest > 0 :
                                break

                    #### print trainable variable values
                    values = sess.run(vars_name)
                    for k, v in zip(vars_name,values):
                        print("Variable: ",k)
                        print("Shape: ",v.shape)
                        print(type(v))

                    train_fd = {x: Xtrain, y_: Ytrain, keep_prob:1.0,phase_train: False}
                    units,YtrainPred,_, train_accuracy = sess.run([conv1,y_pred,cost, accuracy],feed_dict=train_fd)


                    test_fd = {x: Xtest, y_: Ytest, keep_prob:1.0, phase_train: False}
                    YtestPred, test_cost, test_accuracy = sess.run([y_pred, cost, accuracy], feed_dict=test_fd)


                    # Save the variables to disk.
                    save_path = saver.save(sess,  self.fpath+'models/simple_cnn.ckpt')

                    print("Model saved in file: %s" % save_path)

                    print("Train Accuracy:",train_accuracy)
                    print("Test Accuracy:",test_accuracy)

        header = ["epoch","loss_train","loss_test"]
        pdLoss = pd.DataFrame(list(zip(Iepoch,np.log(LossTrain),np.log(LossTest))), columns=header)
        pdLoss.to_csv(self.fpath+"LossEpoch.csv",encoding='utf-8',index=False)
        return Ytrain,YtrainPred,Ytest,YtestPred,units


    def pred(self,Xdata,nclasses,batch_size):
        print("Prediction Start...")
        with tf.variable_scope("Prediction"):
            with tf.Graph().as_default():
                Nbatches = int(Xdata.shape[0] / batch_size)
                print("Number of batches = ", Nbatches)
                w, h, c = Xdata.shape[1:]
                x = tf.placeholder(tf.float32, [None, w, h, c])
                y_ = tf.placeholder(tf.float32, [None, nclasses])
                keep_prob = tf.placeholder(tf.float32)
                phase_train = tf.placeholder(tf.bool, name='phase_train')

                _, y_pred,_ = network(x, w, h, c, nclasses, keep_prob, phase_train,self.wd, self.SPP)
                saver, vars = self.saveVars()
                init = tf.global_variables_initializer()
                Ypred = []
                with tf.Session() as sess:
                    sess.run(init)
                    saver.restore(sess, self.fpath + 'models/simple_cnn.ckpt')
                    for batch in range(Nbatches):
                        batch_xs = Xdata[batch * batch_size:(batch + 1) * batch_size,:,:,:]
                        batch_fd = {x:batch_xs,keep_prob:1.0,phase_train:False}
                        temp = sess.run(y_pred,feed_dict=batch_fd)
                        Ypred.extend(temp)
        print("Prediction End")
        return np.array(Ypred)
