import build
import time
import argparse

seed_value= 0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

from evaluate import test_perf

# np.set_printoptions(threshold=np.nan)

#Main Run Thread
if __name__=='__main__':
    parser = argparse.ArgumentParser()
	# n-step prediction
    parser.add_argument('-s','--step', type=int, default=3)
	# data path
    # parser.add_argument('-d','--data_file', type=str, default='../dataset/stocknet.npy')
    # parser.add_argument('-t', '--text_file', type=str,
     #                    default='../dataset/tw')

    parser.add_argument('-d', '--data_file', type=str,
                        default='../dataset/news.npy')
    parser.add_argument('-t', '--text_file', type=str,
                        default='../dataset/news')

	# dimension
    parser.add_argument('-hd','--hidden_dim', type=int, default=50)
    parser.add_argument('-td', '--text_dim', type=int, default=50)
    parser.add_argument('-ad', '--att_dim', type=int, default=50)
    parser.add_argument('-tn', '--text_num', type=int, default=3)
    parser.add_argument('-f','--freq_dim', type=int, default=10)
	# training parameter
    parser.add_argument('-n','--niter', type=int, default=2000)
    parser.add_argument('-ns', '--nsnapshot', type=int, default=40)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    
    args = parser.parse_args()
    step = args.step
    print args

    global_start_time = time.time()

    print '> Loading data... '
    
    data_file = args.data_file
    X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = build.load_data(data_file, step)
    train_len = X_train.shape[1]
    val_len = X_val.shape[1]-X_train.shape[1]
    test_len = X_test.shape[1]-X_val.shape[1]
    print train_len, val_len, test_len

    M_train, E_train, M_val, E_val, M_test, E_test = build.load_tw_data(
        args.text_file, tw_thres=args.text_num, step=step, emb_size=args.text_dim)

    print '> Data Loaded. Compiling...'
    model = build.build_isfm([1, args.hidden_dim, 1, args.text_num, args.text_dim, args.att_dim], args.freq_dim, args.learning_rate)
    best_error = np.inf
    best_iter = 0
    best_tes_error = np.inf
    best_tes_mse = np.inf


    dataset_short = args.text_file.split('/')[-1]
    for ii in range(int(args.niter/args.nsnapshot)):
        model.fit(
            # X_train,
            [X_train, M_train, E_train],
            y_train,
            batch_size=X_train.shape[0],
            nb_epoch=args.nsnapshot,
            validation_split=0)
        
        num_iter = str(args.nsnapshot * (ii+1))
        model.save_weights('../model/{}/step{}/weights{}.hdf5'.format(dataset_short, step, num_iter), overwrite = True)
        
        # predicted = model.predict(X_train)
        predicted = model.predict([X_train, M_train, E_train])
        train_error = np.sum((predicted[:,:,0] - y_train[:,:,0])**2) / \
                      (predicted.shape[0] * predicted.shape[1])
        
        print num_iter, ' training error ', train_error

        # predicted = model.predict(X_val)
        predicted = model.predict([X_val, M_val, E_val])
        val_error = np.sum((predicted[:,-val_len:,0] - y_val[:,-val_len:,0])**2) / \
                    (val_len * predicted.shape[0])
        
        print ' val error ', val_error

        # predicted = model.predict(X_test)
        predicted = model.predict([X_test, M_test, E_test])
        tes_error = np.sum((predicted[:, -test_len:, 0] - y_test[:, -test_len:, 0]) ** 2) / \
                    (test_len * predicted.shape[0])

        print ' test error ', tes_error

        # denormalization
        prediction = (predicted[:, :, 0] * (max_data - min_data) + (
        max_data + min_data)) / 2

        # tes_mse = np.sum((prediction[:, -test_len:] - gt_test[:, -test_len:]) ** 2) / \
        #         (test_len * prediction.shape[0])
        # print 'MSE: %f' % tes_mse

        tes_p = test_perf(prediction[:, -test_len:], gt_test[:, -test_len:],
                          M_test[:, -test_len:, 0])
        print 'MSE:', tes_p

        if (val_error < best_error):
            best_error = val_error
            best_iter = args.nsnapshot * (ii + 1)
            best_tes_error = tes_error
            best_tes_mse = tes_p
    
    print 'Training duration (s) : ', time.time() - global_start_time
    print 'best iteration ', best_iter
    print 'smallest val error ', best_error
    print 'associated tes error ', best_tes_error
    print 'associated tes mse ', best_tes_mse
