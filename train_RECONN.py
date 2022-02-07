import tensorflow as tf
import pandas as pd
import numpy as np
import os, json, argparse, tqdm, sklearn, plotting
import matplotlib.pyplot as plt
from plotting.plotter import plotter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
from random import seed
from random import random
from random import randint
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers.core import Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
# from keras.layers.core import Activation
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
tf.compat.v1.disable_eager_execution()
print(tf.compat.v1.get_default_graph())

def compile_model(num_variables,learn_rate=0.001):
    model = Sequential()
    model.add(Dense(num_variables, input_dim=num_variables))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(Dense(1, activation="sigmoid"))
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model

def main():
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-i', '--input_dataframe', dest='input_dataframe', help='Path to input dataframe', default='', type=str)
    parser.add_argument('-o', '--outputs', dest='outputdir', help='Name of output directory', default='test', type=str)
    parser.add_argument('-f', '--fit_model', dest='fit_model', help='Flag: 0 to evaluate a pre-existing model,  1 to fit a new model', default=1, type=int)
    args = parser.parse_args()
    input_dataframe = args.input_dataframe
    outputdir = args.outputdir
    fit_model = args.fit_model
    # Make instance of plotter tool
    Plotter = plotter()

    if os.path.isdir(outputdir) != 1:
        os.mkdir(outputdir)
    plots_dir = os.path.join(outputdir,'plots/')
    if os.path.isdir(plots_dir) != 1:
        os.mkdir(plots_dir)

    # Create dataset from .csv
    event_data = pd.read_csv(input_dataframe)
    print("Input Dataframe: ", event_data.shape)
    input_columns_training = event_data.columns[:-3]
    #column_headers = ['HW1_jet1_pt','HW1_jet1_eta','HW1_jet1_phi','HW1_jet1_E','HW1_jet2_pt','HW1_jet2_eta','HW1_jet2_phi','HW1_jet2_E','HW1_jet3_pt','HW1_jet3_eta','HW1_jet3_phi','HW1_jet3_E','HW1_jet4_pt','HW1_jet4_eta','HW1_jet4_phi','HW1_jet4_E','target']
    #column_headers = ['HW1_jet1_pt','HW1_jet1_eta','dRj1_photon1','dRj1_photon2','HW1_jet2_pt','HW1_jet2_eta','dRj2_photon1','dRj2_photon2','HW1_jet3_pt','HW1_jet3_eta','dRj3_photon1','dRj3_photon2','HW1_jet4_pt','HW1_jet4_eta','dRj4_photon1','dRj4_photon2','target','event_ID']
    model_output = os.path.join(outputdir,'model/')
    if fit_model == 1:
        traindataset, valdataset = train_test_split(event_data, test_size=0.1)
        print('Using columns: ', input_columns_training.values)
        print('Using labels: ', event_data['target'].values)
        print('Using event_ID: ', event_data['event_ID'].values)
        print('Using njets: ', event_data['njets'].values)
        train_input_ = traindataset[input_columns_training].values
        train_target_ = traindataset['target'].values
        test_input_ = valdataset[input_columns_training].values
        test_target_ = valdataset['target'].values

        # Event weights if wanted
        train_weights = np.ones(len(traindataset['target']))
        print('train_weights: ' , train_weights)
        test_weights = np.ones(len(valdataset['target']))
        labels = np.array(event_data['target'])
        print(labels)

        class_weights = np.array(class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels ))
        class_weights = dict(enumerate(class_weights))
        print('class_weights: ', class_weights)

        # Fit label encoder to Y_train
        newencoder = LabelEncoder()
        newencoder.fit(train_target_)
        # Transform to encoded array
        encoded_Y = newencoder.transform(train_target_)
        encoded_Y_test = newencoder.transform(test_target_)

        histories = []
        labels = []
        print(train_input_)

        # Fitting the model
        early_stopping_monitor = EarlyStopping(patience=100, monitor='val_loss', min_delta=0.01, verbose=1)
        model = compile_model(len(input_columns_training), learn_rate=0.001)
        history_ = model.fit(train_input_,train_target_,validation_split=0.1,class_weight=class_weights,epochs=200,batch_size=512,verbose=1,shuffle=True,callbacks=[early_stopping_monitor])

        # Store model in file
        model_output_name = os.path.join(model_output)
        model.save(model_output_name)
        weights_output_name = os.path.join(model_output,'model_weights.h5')
        model.save_weights(weights_output_name)
        model_json = model.to_json()
        model_json_name = os.path.join(model_output,'model_serialised.json')
        with open(model_json_name,'w') as json_file:
            json_file.write(model_json)
        train_pred_ = model.predict(np.array(train_input_))
        test_pred_ = model.predict(np.array(test_input_))

        prediction_data = []
        for entry in range(0,len(test_pred_)):
            prediction_data.append( [test_pred_[entry],test_target_] )
        df = pd.DataFrame(prediction_data)
        df.columns = ['prediction','label']
        df.to_csv(os.path.join(outputdir,"predictions_dataframe.csv"), index=False)

        # Make instance of plotter tool (need for DNN response plots)
        Plotter = plotter()
        # Initialise output directory.
        Plotter.plots_directory = os.path.join(plots_dir,'perf')
        Plotter.output_directory = outputdir
        # History plots
        Plotter.history_plot(history_, label='loss')
        Plotter.save_plots(dir=Plotter.plots_directory, filename='history_loss.png')
        # ROC curves
        Plotter.ROC(train_target_,train_pred_,test_target_,test_pred_)
        Plotter.save_plots(dir=Plotter.plots_directory, filename='ROC.png')
        # Make overfitting plots of output nodes
        Plotter.binary_overfitting(model, train_target_, test_target_, train_pred_, test_pred_, train_weights, test_weights)
        Plotter.save_plots(dir=Plotter.plots_directory, filename='response.png')
    elif fit_model == 0:
        model_name = os.path.join(model_output)
        model = keras.models.load_model(model_name)
        evaluation_inputs_ = event_data[input_columns_training].values
        evaluation_targets_ = event_data['target'].values
        evaluation_IDs_ = event_data['event_ID'].values
        event_njets = event_data['njets'].values
        predictions_ = model.predict(np.array(evaluation_inputs_))

        previous_evID=0
        max_DNN_score = -9
        truth_label_of_max_score = -9
        correct_predictions = 0
        incorrect_predictions = 0

        correct_4jet_predictions = 0
        incorrect_4jet_predictions = 0
        correct_5jet_predictions = 0
        incorrect_5jet_predictions = 0
        correct_6jet_predictions = 0
        incorrect_6jet_predictions = 0
        N_4jet_examples = 0
        N_5jet_examples = 0
        N_6jet_examples = 0
        for evID_index in range(0,len(evaluation_IDs_)):
            # If first event
            if evID_index == 0:
                previous_evID = evaluation_IDs_[evID_index]

            if event_njets[evID_index] == 4 and evaluation_targets_[evID_index] == 0:
                print('WARNING: 4 jet event with target label == 0 ')
                print('Event: ', evaluation_IDs_[evID_index-1])
                print('NJets: ', event_njets[evID_index-1])
                print('Target label: ', evaluation_targets_[evID_index-1])

            # Elif same event ID as previous event
            elif evaluation_IDs_[evID_index] == previous_evID:
                # If prediction has largest response so far
                if predictions_[evID_index] > max_DNN_score:
                    max_DNN_score = predictions_[evID_index]
                    truth_label_of_max_score = evaluation_targets_[evID_index]

            # Else we check the result for the previous event and start a new event
            elif evaluation_IDs_[evID_index] != previous_evID:
                if event_njets[evID_index-1] == 4:
                    N_4jet_examples+=1
                    if truth_label_of_max_score == 1 :
                        correct_4jet_predictions += 1
                    else:
                        print("evaluation ID:", evaluation_IDs_[evID_index])
                        print("truth label:",truth_label_of_max_score)
                        incorrect_4jet_predictions += 1
                if event_njets[evID_index-1] == 5:
                    N_5jet_examples+=1
                    if truth_label_of_max_score == 1 :
                        correct_5jet_predictions += 1
                    else:
                        incorrect_5jet_predictions += 1
                if event_njets[evID_index-1] == 6:
                    N_6jet_examples+=1
                    if truth_label_of_max_score == 1 :
                        correct_6jet_predictions += 1
                    else:
                        incorrect_6jet_predictions += 1
                # If the DNN score a signal permutation (target == 1) as the highest -> correct
                if truth_label_of_max_score == 1 :
                    correct_predictions += 1
                # Otherwise incorrect
                else:
                    incorrect_predictions += 1
                previous_evID = evaluation_IDs_[evID_index]
                max_DNN_score = predictions_[evID_index]

        print('Dataset contained %s 4-jet, %s 5-jet, %s 6-jet events' %(N_4jet_examples,N_5jet_examples,N_6jet_examples))
        print('# 4-jet correct: %s, # 4-jet incorrect: %s (%s percent)' % (correct_4jet_predictions,incorrect_4jet_predictions,(correct_4jet_predictions/(correct_4jet_predictions+incorrect_4jet_predictions))))
        print('# 5-jet correct: %s, # 5-jet incorrect: %s (%s percent)' % (correct_5jet_predictions,incorrect_5jet_predictions,(correct_5jet_predictions/(correct_5jet_predictions+incorrect_5jet_predictions))))
        print('# 6-jet correct: %s, # 6-jet incorrect: %s (%s percent)' % (correct_6jet_predictions,incorrect_6jet_predictions,(correct_6jet_predictions/(correct_6jet_predictions+incorrect_6jet_predictions))))
        print('# total correct: %s, # total incorrect %s (%s percent)' % (correct_predictions,incorrect_predictions,(correct_predictions/(correct_predictions+incorrect_predictions))))

    exit(0)

main()
