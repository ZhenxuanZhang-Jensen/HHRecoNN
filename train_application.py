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
from itertools import permutations
os.environ['KERAS_BACKEND'] = 'tensorflow'
# import keras
from tensorflow import keras
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
from tensorflow.keras.models import load_model
tf.compat.v1.disable_eager_execution()
print(tf.compat.v1.get_default_graph())
# the Event.py python file save the jet and event class and some useful functions
import Event 
import ROOT
from ROOT import TTree, TFile
from Event import momentum_tensor,sphericity

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
    parser.add_argument('-o', '--outputs', dest='outputdir', help='Name of output directory', default='test', type=str)
    parser.add_argument('-l', '--load_model_option', dest='load_model_option', help='Option to load full model (\'m\') or from weights file (\'w\')', default='m', type=str)
    args = parser.parse_args()
    outputdir = args.outputdir
    load_model_option = args.load_model_option
    # Make instance of plotter tool
    Plotter = plotter()

    if os.path.isdir(outputdir) != 1:
        os.mkdir(outputdir)
    plots_dir = os.path.join(outputdir,'plots/')
    if os.path.isdir(plots_dir) != 1:
        os.mkdir(plots_dir)

    # Create dataset from .csv
    #column_headers = ['HW1_jet1_pt','HW1_jet1_eta','HW1_jet1_phi','HW1_jet1_E','HW1_jet2_pt','HW1_jet2_eta','HW1_jet2_phi','HW1_jet2_E','HW1_jet3_pt','HW1_jet3_eta','HW1_jet3_phi','HW1_jet3_E','HW1_jet4_pt','HW1_jet4_eta','HW1_jet4_phi','HW1_jet4_E','target']
    #column_headers = ['HW1_jet1_pt','HW1_jet1_eta','dRj1_photon1','dRj1_photon2','HW1_jet2_pt','HW1_jet2_eta','dRj2_photon1','dRj2_photon2','HW1_jet3_pt','HW1_jet3_eta','dRj3_photon1','dRj3_photon2','HW1_jet4_pt','HW1_jet4_eta','dRj4_photon1','dRj4_photon2','target','event_ID']
    model_output = os.path.join(outputdir,'model')
    if(load_model_option == 'm'):
        model_name = os.path.join(model_output,'model.h5')
        model = keras.models.load_model(model_name)
    ordered_jets = []
    tfile = TFile.Open("GluGluToRadionToHHTo2G4Q_M300.root","READ")
    input_ttree = tfile.Get("Events")
    nentries = input_ttree.GetEntries()
    Index_file = ROOT.TFile("GluGluToRadionToHHTo2G4Q_M300_DNNJetsIndex.root", "RECREATE")
    # Index_tree = ROOT.TTree("Index", "An Example Tree")
    Index_tree = input_ttree.CloneTree(0)
    jet_index_vec = ROOT.vector('double')()
    Index_tree.Branch("jet_index_vec",jet_index_vec)
    for entry in range(nentries):
        max_score = -99
        input_ttree.GetEntry(entry)
        this_event = Event.Event(input_ttree)
    #     ordered_jets.append(this_event.good_jets_vector)
    #     print(input_ttree.pho1_pt)
    #     print(this_event.good_jets_vector[1].index)
        permutations_list = list(permutations(this_event.good_jets_vector,len(this_event.good_jets_vector)))
        fourjet_perm_list = []
        fourjet_pt_list = []
        for perm_ in permutations_list:
            tmp_list = []
            data_list = []
            perm_list = []
            is_good_perm = 1
            pt_list = []
            # Check if this permutation makes sense
            for jet_index in range(0,4):
                perm_list.append(perm_[jet_index])
                pt_list.append(perm_[jet_index].LorentzVector.Pt())
                # print("pt:",perm_[jet_index].LorentzVector.Pt())
                if (perm_[jet_index].LorentzVector.Pt() >= 900):
                    # Reject permutation: non-existent jet was permuted into first 4 jet positions
                    is_good_perm = 0
            if is_good_perm == 0:
                # Reject bad/pointless permutation
                continue
            # Reject permutation if the permutation is just of the additional (>4) jets
            # pt_array = np.array(pt_list)
            if perm_list in fourjet_perm_list:
                is_good_perm = 0
            else:
                fourjet_perm_list.append(perm_list)
                fourjet_pt_list.append(pt_list)

            if is_good_perm == 0:
                # Reject bad/pointless permutation
                continue
            for jet_index in range(0,4):
                #Important
                tmp_list.append(np.log(perm_[jet_index].LorentzVector.Pt()))
                tmp_list.append(perm_[jet_index].LorentzVector.Eta())
                tmp_list.append(perm_[jet_index].LorentzVector.Phi())
                tmp_list.append(np.log(perm_[jet_index].LorentzVector.E()))
                tmp_list.append(perm_[jet_index].LorentzVector.DeltaR(this_event.photon1))
                tmp_list.append(perm_[jet_index].LorentzVector.DeltaR(this_event.photon2))
                eigvals_, eigvecs_ = momentum_tensor([
                perm_[0].LorentzVector,
                perm_[1].LorentzVector,
                perm_[2].LorentzVector,
                perm_[3].LorentzVector])
            spher_ = Event.sphericity(eigvals_)
            tmp_list.append(spher_)
            data_list.append(tmp_list)
            predictions_ = model.predict(np.array(data_list))
            if (predictions_[0][0] > max_score):
                max_score = predictions_[0][0]
                list_index = []
                list_index.append(perm_[0].index)
                list_index.append(perm_[1].index)
                list_index.append(perm_[2].index)
                list_index.append(perm_[3].index)
                list_index.sort()
                index1 = list_index[0]
                index2 = list_index[1]
                index3 = list_index[2]
                index4 = list_index[3]
            # print("predic max:", max_score)
            # print("index1:",index1)
            # print("index2:",index2)
            # print("index3:",index3)
            # print("index4:",index4)
        jet_index_vec.push_back(index1)
        jet_index_vec.push_back(index2)
        jet_index_vec.push_back(index3)
        jet_index_vec.push_back(index4)
        Index_tree.Fill()
        jet_index_vec.clear()
    Index_tree.Write("")
    Index_file.Close()
    print("finish")
    exit(0)

main()
