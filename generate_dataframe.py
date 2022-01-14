#############################################
# Author: Joshuha Thomas-Wilsker
# Institute: IHEP Beijing
# Description:
# Assigns reco jets to GenQuarks using dR matching
# Selects events with 4 'well-matched' jets (dR<0.3)
# Constructs vectors of 4-vectors of jets
# Generates all possible permutations of jets
# Correct permutation = first four jets are dR matched to GenQuarks in ascending order
# Incorrect permutation = any other permutation
# Creates dataframe:
#   Signal: Target = 1 = correct permutation
#   Background: Target = 0 = incorrect permutations
#############################################
# To do:
#   Should make signal any event with all four dR matched jets in the first 4 positions?

import optparse, json, argparse, math, os, pickle, uuid
import numpy as np
from numpy import linalg
import pandas as pd
import ROOT
from ROOT import TTree, TFile
from array import array
import root_numpy
from root_numpy import tree2array
from collections import OrderedDict
from itertools import permutations

class Jet:
    """Event class for jets to include btag info in object"""
    def __init__(self):
        self.BDisc = -99.
        self.LorentzVector = ROOT.TLorentzVector()

    def SetLorentzVector(self, Pt, Eta, Phi, Mass):
        self.LorentzVector.SetPtEtaPhiM(Pt, Eta, Phi, Mass)

    def SetBTag(self, BDisc):
        self.BDisc = BDisc

class Event:
    """Event class has TLorentzVectors for jets and W bosons from Higgs as attributes"""
    def __init__(self,input_ttree):
        self.unique_id = uuid.uuid1()
        self.njets = input_ttree.nGoodAK4jets
        self.gen_W_1 = ROOT.TLorentzVector()
        self.gen_W_1.SetPtEtaPhiE(input_ttree.GEN_W1_pT,input_ttree.GEN_W1_eta,input_ttree.GEN_W1_phi,input_ttree.GEN_W1_energy)
        self.gen_W_2 = ROOT.TLorentzVector()
        self.gen_W_2.SetPtEtaPhiE(input_ttree.GEN_W2_pT,input_ttree.GEN_W2_eta,input_ttree.GEN_W2_phi,input_ttree.GEN_W2_energy)

        self.quark0 = ROOT.TLorentzVector()
        self.quark0.SetPtEtaPhiE(input_ttree.GEN_Q1_pT,input_ttree.GEN_Q1_eta,input_ttree.GEN_Q1_phi,input_ttree.GEN_Q1_energy)
        self.quark1 = ROOT.TLorentzVector()
        self.quark1.SetPtEtaPhiE(input_ttree.GEN_Q2_pT,input_ttree.GEN_Q2_eta,input_ttree.GEN_Q2_phi,input_ttree.GEN_Q2_energy)
        self.quark2 = ROOT.TLorentzVector()
        self.quark2.SetPtEtaPhiE(input_ttree.GEN_Q3_pT,input_ttree.GEN_Q3_eta,input_ttree.GEN_Q3_phi,input_ttree.GEN_Q3_energy)
        self.quark3 = ROOT.TLorentzVector()
        self.quark3.SetPtEtaPhiE(input_ttree.GEN_Q4_pT,input_ttree.GEN_Q4_eta,input_ttree.GEN_Q4_phi,input_ttree.GEN_Q4_energy)

        self.jet0 = Jet()
        self.jet0.SetLorentzVector(input_ttree.FullyResolved_Jet1_pt,input_ttree.FullyResolved_Jet1_eta,input_ttree.FullyResolved_Jet1_phi,input_ttree.FullyResolved_Jet1_M)
        # self.jet0.SetBTag(input_ttree.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet1 = Jet()
        self.jet1.SetLorentzVector(input_ttree.FullyResolved_Jet2_pt,input_ttree.FullyResolved_Jet2_eta,input_ttree.FullyResolved_Jet2_phi,input_ttree.FullyResolved_Jet2_M)
        # self.jet1.SetBTag(input_ttree.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet2 = Jet()
        self.jet2.SetLorentzVector(input_ttree.FullyResolved_Jet3_pt,input_ttree.FullyResolved_Jet3_eta,input_ttree.FullyResolved_Jet3_phi,input_ttree.FullyResolved_Jet3_M)
        # self.jet2.SetBTag(input_ttree.goodJets_2_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_2_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_2_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet3 = Jet()
        self.jet3.SetLorentzVector(input_ttree.FullyResolved_Jet4_pt,input_ttree.FullyResolved_Jet4_eta,input_ttree.FullyResolved_Jet4_phi,input_ttree.FullyResolved_Jet4_M)
        # self.jet3.SetBTag(input_ttree.goodJets_3_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_3_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_3_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet4 = Jet()
        self.jet4.SetLorentzVector(input_ttree.FullyResolved_Jet5_pt,input_ttree.FullyResolved_Jet5_eta,input_ttree.FullyResolved_Jet5_phi,input_ttree.FullyResolved_Jet5_M)
        # self.jet4.SetBTag(input_ttree.goodJets_4_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_4_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_4_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet5 = Jet()
        self.jet5.SetLorentzVector(input_ttree.FullyResolved_Jet6_pt,input_ttree.FullyResolved_Jet6_eta,input_ttree.FullyResolved_Jet6_phi,input_ttree.FullyResolved_Jet6_M)
        # self.jet5.SetBTag(input_ttree.goodJets_5_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_5_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_5_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet6 = Jet()
        self.jet6.SetLorentzVector(input_ttree.FullyResolved_Jet7_pt,input_ttree.FullyResolved_Jet7_eta,input_ttree.FullyResolved_Jet7_phi,input_ttree.FullyResolved_Jet7_M)
        # self.jet6.SetBTag(input_ttree.goodJets_6_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_6_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_6_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.good_jet_lorentz_vectors = [self.jet0.LorentzVector,self.jet1.LorentzVector,self.jet2.LorentzVector,self.jet3.LorentzVector,self.jet4.LorentzVector,self.jet5.LorentzVector,self.jet6.LorentzVector]
        self.good_jets_vector = [self.jet0,self.jet1,self.jet2,self.jet3,self.jet4,self.jet5,self.jet6]

        self.photon1 = ROOT.TLorentzVector()
        self.photon1.SetPtEtaPhiE(input_ttree.pho1_pt,input_ttree.pho1_eta,input_ttree.pho1_phi,input_ttree.pho1_E)
        self.photon2 = ROOT.TLorentzVector()
        self.photon2.SetPtEtaPhiE(input_ttree.pho2_pt,input_ttree.pho2_eta,input_ttree.pho2_phi,input_ttree.pho2_E)
    # Matrix of dR between generator quarks and 6 leading jets
    def create_dR_matrix(self):
        self.dR_matrix = np.array([
        [self.quark0.DeltaR(self.jet0.LorentzVector),self.quark1.DeltaR(self.jet0.LorentzVector),self.quark2.DeltaR(self.jet0.LorentzVector),self.quark3.DeltaR(self.jet0.LorentzVector)],
        [self.quark0.DeltaR(self.jet1.LorentzVector),self.quark1.DeltaR(self.jet1.LorentzVector),self.quark2.DeltaR(self.jet1.LorentzVector),self.quark3.DeltaR(self.jet1.LorentzVector)],
        [self.quark0.DeltaR(self.jet2.LorentzVector),self.quark1.DeltaR(self.jet2.LorentzVector),self.quark2.DeltaR(self.jet2.LorentzVector),self.quark3.DeltaR(self.jet2.LorentzVector)],
        [self.quark0.DeltaR(self.jet3.LorentzVector),self.quark1.DeltaR(self.jet3.LorentzVector),self.quark2.DeltaR(self.jet3.LorentzVector),self.quark3.DeltaR(self.jet3.LorentzVector)],
        [self.quark0.DeltaR(self.jet4.LorentzVector),self.quark1.DeltaR(self.jet4.LorentzVector),self.quark2.DeltaR(self.jet4.LorentzVector),self.quark3.DeltaR(self.jet4.LorentzVector)],
        [self.quark0.DeltaR(self.jet5.LorentzVector),self.quark1.DeltaR(self.jet5.LorentzVector),self.quark2.DeltaR(self.jet5.LorentzVector),self.quark3.DeltaR(self.jet5.LorentzVector)],
        [self.quark0.DeltaR(self.jet6.LorentzVector),self.quark1.DeltaR(self.jet6.LorentzVector),self.quark2.DeltaR(self.jet6.LorentzVector),self.quark3.DeltaR(self.jet6.LorentzVector)]])
        '''[self.quark0.DeltaR(self.jet7),self.quark1.DeltaR(self.jet7),self.quark2.DeltaR(self.jet7),self.quark3.DeltaR(self.jet7)],
        [self.quark0.DeltaR(self.jet8),self.quark1.DeltaR(self.jet8),self.quark2.DeltaR(self.jet8),self.quark3.DeltaR(self.jet8)],
        [self.quark0.DeltaR(self.jet9),self.quark1.DeltaR(self.jet9),self.quark2.DeltaR(self.jet9),self.quark3.DeltaR(self.jet9)]])
        '''

# Assign jets to generator quarks
def dRmatch_qj(dR_matrix):
    mindR_coordinates_ = []
    dR_ordered_indices = []
    # Order (smallest->largest) dR matrix
    for index in range(0,len(dR_matrix.flatten())):
        tmp_val_ = np.partition(dR_matrix.flatten(),index).item(index)
        tmp_index_ = np.where(dR_matrix == tmp_val_)
        # Fill array containing ordered indices of dR(q,j)
        dR_ordered_indices.append(tmp_index_)
    for entry in dR_ordered_indices:
        tmp_coords_ = [entry[0][0],entry[1][0]]
        # Check if jet has already been assigned to a quark
        jet_already_assigned = bool([existing_entries_ for existing_entries_ in mindR_coordinates_ if(tmp_coords_[0] == existing_entries_[0])])
        # Check if jets closest matched quark is already assigned
        NJets_assigned_to_quark = [test_coords_[1] for test_coords_ in mindR_coordinates_].count(tmp_coords_[1])
        if jet_already_assigned:
            continue
        elif NJets_assigned_to_quark>0:
            continue
        else:
            # Add jets coordinates in a matrix to list of dR with quarks
            mindR_coordinates_.append(tmp_coords_)
    return mindR_coordinates_

def momentum_tensor(list_of_jets_lorentzvectors_):
    M_xy = np.array([[0.,0.],[0.,0.]])
    for v_ in list_of_jets_lorentzvectors_:
        #Transverse momentum tensor (symmetric matrix)
        M_xy += np.array([
        [v_.Px()*v_.Px(),v_.Px()*v_.Py()],
        [v_.Px()*v_.Py(),v_.Py()*v_.Py()]]
        )
        eigvals, eigvecs = linalg.eig(M_xy)
    eigvals.sort()
    return eigvals, eigvecs

def sphericity(eigvals):
    # Definition: http://sro.sussex.ac.uk/id/eprint/44644/1/art%253A10.1007%252FJHEP06%25282010%2529038.pdf
    spher_ = 2*eigvals[0] / (eigvals[1]+eigvals[0])
    return spher_

def make_plot_(histograms,axis_titles,dimensions,save_title,draw_style):
    c1 = ROOT.TCanvas('c1',',1000,1000')
    p1 = ROOT.TPad('p1','p1',0.0,0.0,1.0,1.0)
    p1.Draw()
    p1.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    legend = ROOT.TLegend(0.7,0.8,0.9,0.9)
    colour_index = 1
    maxY = 0.
    for hist_ in histograms:
        if hist_.GetMaximum() > maxY:
            maxY = hist_.GetMaximum()
        hist_.SetMaximum(maxY*1.5)
        hist_.SetLineColor(colour_index)
        hist_.GetXaxis().SetTitle(axis_titles[0])
        legend.AddEntry(hist_,hist_.GetTitle(),"l")
        if dimensions==2:
            hist_.SetMarkerStyle(1)
            hist_.SetMarkerSize(1)
            hist_.SetMarkerColor(colour_index)
            hist_.GetYaxis().SetTitle(axis_titles[1])
        if colour_index==1:
            hist_.Draw(draw_style)
        else:
            hist_.Draw(draw_style+" SAME")
        colour_index+=1
    legend.Draw('SAME')
    c1.SaveAs(save_title,'png')

def load_data(inputPath,column_headers,output_dir):
    # Dictionary tranlsating file names to labels
    process_filename = OrderedDict([
        ('HH' , ['GluGluToRadionToHHTo2G4Q_M300','Events','HH',1]),
    ])

    for files in process_filename:
        # Now open the TFile and load TTree of the file you want to append the MVA branch to.
        filename = process_filename.get(files)[0]
        treename = process_filename.get(files)[1]
        process_ID = process_filename.get(files)[2]
        target = process_filename.get(files)[3]
        inpath = os.path.join(inputPath, filename) + '.root'

        try:
            tfile = TFile.Open(inpath)
        except FileNotFoundError:
            raise FileNotFoundError('Input file %s not found!' % (inpath))

        try:
            input_ttree = tfile.Get(treename)
        except ReferenceError:
            raise ReferenceError('TTree %s not found' % treename)
            exit(0)

        # Create dataframe for ttree
        # ==2 leptons already applied when making ntuples
        # Need to apply remaining pre-selection

        nentries = input_ttree.GetEntries()
        W1_eta_phi_h = ROOT.TH2F('w1etaphi', 'W1 Gen', 20,-3,3,20,-5,5)
        W2_eta_phi_h = ROOT.TH2F('w2etaphi', 'W2 Gen', 20,-3,3,20,-5,5)
        j1_eta_phi_h = ROOT.TH2F('w1etaphi', '\'good jet\' 1', 20,-3,3,20,-5,5)
        sphericity_correct_h = ROOT.TH1F('sphericity_correct_h', 'Sphericity (correct)', 20,0,1)
        sphericity_incorrect_h = ROOT.TH1F('sphericity_incorrect_h', 'Sphericity (incorrect)', 20,0,1)

        min_dR_jet_quark = ROOT.TH1F('min_dR_jet_quark', 'min. dR(jet,quark)', 10,0,0.5)
        submin_dR_jet_quark = ROOT.TH1F('submin_dR_jet_quark', '2nd min. dR(jet,quark)', 10,0,0.5)
        thirdmin_dR_jet_quark = ROOT.TH1F('thirdmin_dR_jet_quark', '3rd min. dR(jet,quark)', 10,0,0.5)
        fourthmin_dR_jet_quark = ROOT.TH1F('submin_dR_jet_quark', '4th min. dR(jet,quark)', 10,0,0.5)

        ptjet_of_min_dR_pair = ROOT.TH1F('ptjet_of_min_dR_pair', 'pT of jet in min. dR(jet,quark)', 50,0,200)
        ptjet_of_submin_dR_pair = ROOT.TH1F('ptjet_of_submin_dR_pair', 'pT of jet in 2nd min. dR(jet,quark)', 50,0,200)
        ptjet_of_thirdmin_dR_pair = ROOT.TH1F('ptjet_of_thirdmin_dR_pair', 'pT of jet in 3rd min. dR(jet,quark)', 50,0,150)
        ptjet_of_fourthmin_dR_pair = ROOT.TH1F('ptjet_of_fourthmin_dR_pair', 'pT of jet in 4th min. dR(jet,quark)', 50,0,150)

        nmatched_jets_h = ROOT.TH1F('nmatched_jets_h', '# Gen-matched jets', 7,0,7)
        jet1_pt_correct = ROOT.TH1F('jet1_pt_correct', 'Jet pT (correct perm.)', 50,0,200)
        jet1_pt_incorrect = ROOT.TH1F('jet1_pt_incorrect', 'Jet pT (incorrect perm.)', 50,0,200)
        jet1_eta_correct = ROOT.TH1F('jet1_eta_correct', 'Jet eta (correct perm.)', 10,0,3)
        jet1_eta_incorrect = ROOT.TH1F('jet1_eta_incorrect', 'Jet eta (incorrect perm.)', 10,0,3)
        dR_j1_photon1_correct = ROOT.TH1F('dR_j1_photon1_correct', 'dR(j1,y1) (correct perm.)', 10,0,5)
        dR_j1_photon1_incorrect = ROOT.TH1F('dR_j1_photon1_incorrect', 'dR(j1,y1) (incorrect perm.)', 10,0,5)
        dR_j1_photon2_correct = ROOT.TH1F('dR_j1_photon2_correct', 'dR(j1,y2) (correct perm.)', 10,0,5)
        dR_j1_photon2_incorrect = ROOT.TH1F('dR_j1_photon2_incorrect', 'dR(j1,y2) (incorrect perm.)', 10,0,5)
        jet1_btag_correct = ROOT.TH1F('jet1_btag_correct', 'Jet 1 BTag (correct perm.)', 40,0,0.1)
        jet1_btag_incorrect = ROOT.TH1F('jet1_btag_incorrect', 'Jet 1 BTag (incorrect perm.)', 40,0,0.1)

        jet2_pt_correct = ROOT.TH1F('jet2_pt_correct', 'Jet pT (correct perm.)', 50,0,200)
        jet2_pt_incorrect = ROOT.TH1F('jet2_pt_incorrect', 'Jet pT (incorrect perm.)', 50,0,200)
        jet2_eta_correct = ROOT.TH1F('jet2_eta_correct', 'Jet eta (correct perm.)', 10,0,3)
        jet2_eta_incorrect = ROOT.TH1F('jet2_eta_incorrect', 'Jet eta (incorrect perm.)', 10,0,3)
        dR_j2_photon1_correct = ROOT.TH1F('dR_j2_photon1_correct', 'dR(j2,y1) (correct perm.)', 10,0,5)
        dR_j2_photon1_incorrect = ROOT.TH1F('dR_j2_photon1_incorrect', 'dR(j2,y1) (incorrect perm.)', 10,0,5)
        dR_j2_photon2_correct = ROOT.TH1F('dR_j2_photon2_correct', 'dR(j2,y2) (correct perm.)', 10,0,5)
        dR_j2_photon2_incorrect = ROOT.TH1F('dR_j2_photon2_incorrect', 'dR(j2,y2) (incorrect perm.)', 10,0,5)
        jet2_btag_correct = ROOT.TH1F('jet2_btag_correct', 'Jet 2 BTag (correct perm.)', 40,0,0.1)
        jet2_btag_incorrect = ROOT.TH1F('jet2_btag_incorrect', 'Jet 2 BTag (incorrect perm.)', 40,0,0.1)

        jet3_pt_correct = ROOT.TH1F('jet3_pt_correct', 'Jet pT (correct perm.)', 50,0,200)
        jet3_pt_incorrect = ROOT.TH1F('jet3_pt_incorrect', 'Jet pT (incorrect perm.)', 50,0,200)
        jet3_eta_correct = ROOT.TH1F('jet3_eta_correct', 'Jet eta (correct perm.)', 10,0,3)
        jet3_eta_incorrect = ROOT.TH1F('jet3_eta_incorrect', 'Jet eta (incorrect perm.)', 10,0,3)
        dR_j3_photon1_correct = ROOT.TH1F('dR_j3_photon1_correct', 'dR(j3,y1) (correct perm.)', 10,0,5)
        dR_j3_photon1_incorrect = ROOT.TH1F('dR_j3_photon1_incorrect', 'dR(j3,y1) (incorrect perm.)', 10,0,5)
        dR_j3_photon2_correct = ROOT.TH1F('dR_j3_photon2_correct', 'dR(j3,y2) (correct perm.)', 10,0,5)
        dR_j3_photon2_incorrect = ROOT.TH1F('dR_j3_photon2_incorrect', 'dR(j3,y2) (incorrect perm.)', 10,0,5)
        jet3_btag_correct = ROOT.TH1F('jet3_btag_correct', 'Jet 3 BTag (correct perm.)', 40,0,0.1)
        jet3_btag_incorrect = ROOT.TH1F('jet3_btag_incorrect', 'Jet 3 BTag (incorrect perm.)', 40,0,0.1)

        jet4_pt_correct = ROOT.TH1F('jet4_pt_correct', 'Jet pT (correct perm.)', 50,0,200)
        jet4_pt_incorrect = ROOT.TH1F('jet4_pt_incorrect', 'Jet pT (incorrect perm.)', 50,0,200)
        jet4_eta_correct = ROOT.TH1F('jet4_eta_correct', 'Jet eta (correct perm.)', 10,0,3)
        jet4_eta_incorrect = ROOT.TH1F('jet4_eta_incorrect', 'Jet eta (incorrect perm.)', 10,0,3)
        dR_j4_photon1_correct = ROOT.TH1F('dR_j4_photon1_correct', 'dR(j4,y1) (correct perm.)', 10,0,5)
        dR_j4_photon1_incorrect = ROOT.TH1F('dR_j4_photon1_incorrect', 'dR(j4,y1) (incorrect perm.)', 10,0,5)
        dR_j4_photon2_correct = ROOT.TH1F('dR_j4_photon2_correct', 'dR(j4,y2) (correct perm.)', 10,0,5)
        dR_j4_photon2_incorrect = ROOT.TH1F('dR_j4_photon2_incorrect', 'dR(j4,y2) (incorrect perm.)', 10,0,5)
        jet4_btag_correct = ROOT.TH1F('jet4_btag_correct', 'Jet 4 BTag (correct perm.)', 40,0,0.1)
        jet4_btag_incorrect = ROOT.TH1F('jet4_btag_incorrect', 'Jet 4 BTag (incorrect perm.)', 40,0,0.1)

        n_jets_h = ROOT.TH1F('n_jets_h', '# reco jets', 10,0,10)
        data = []
        n_saved_entries = 0
        for entry in range(0,nentries):
            input_ttree.GetEntry(entry)

            if entry % 100 == 0:
                print(entry)
                #already pass the ele+muon>1 events in the selection file
            if input_ttree.nGoodAK4jets < 4 :
                continue
            this_event = Event(input_ttree)
            event_ID = this_event.unique_id

            #if input_ttree.nGoodAK4jets != 4:
            #    continue

            # Matrix of all possible dR(recojet,genQ)
            this_event.create_dR_matrix()

            # List of indices of dR *assigned* jet-quark pairs in dR matrix
            mindR_coordinates_ = dRmatch_qj(this_event.dR_matrix) # only 4 quarks so max entries == 4

            NMatched_jets = 0
            # List of matched jet lorentz vectors
            match_ordered_jets = []
            tmp_list = []
            # Loop over assigned pairings and check if dR fulfills matching requirements
            for coords_ in mindR_coordinates_:
                if this_event.dR_matrix.item(coords_[0],coords_[1]) <= 0.4:
                    NMatched_jets+=1
                # Get jet object
                match_ordered_jets.append(this_event.good_jets_vector[coords_[0]])

            # Require four jets are matched to the W bosons decay products for 'good' events
            nmatched_jets_h.Fill(NMatched_jets)
            n_jets_h.Fill(input_ttree.nGoodAK4jets)
            if NMatched_jets < 4:
                continue

            # Test a limited number of entries for debugging
            n_saved_entries += 1
            # print(n_saved_entries)
            if n_saved_entries > 5000:
                break

            # Create list of indices of assigned recojets in jets list
            matched_jet_indices = []
            for coords_ in mindR_coordinates_:
                matched_jet_indices.append(coords_[0])
            # Append non-assigned jets to list
            for jet_index_ in range(0,len(this_event.good_jets_vector)):
                if jet_index_ not in matched_jet_indices:
                    match_ordered_jets.append(this_event.good_jets_vector[jet_index_])

            min_dR_jet_quark.Fill(this_event.dR_matrix[mindR_coordinates_[0][0],mindR_coordinates_[0][1]])
            submin_dR_jet_quark.Fill(this_event.dR_matrix[mindR_coordinates_[1][0],mindR_coordinates_[1][1]])
            thirdmin_dR_jet_quark.Fill(this_event.dR_matrix[mindR_coordinates_[2][0],mindR_coordinates_[2][1]])
            fourthmin_dR_jet_quark.Fill(this_event.dR_matrix[mindR_coordinates_[3][0],mindR_coordinates_[3][1]])

            ptjet_of_min_dR_pair.Fill(this_event.good_jets_vector[matched_jet_indices[0]].LorentzVector.Pt() )
            ptjet_of_submin_dR_pair.Fill(this_event.good_jets_vector[matched_jet_indices[1]].LorentzVector.Pt() )
            ptjet_of_thirdmin_dR_pair.Fill(this_event.good_jets_vector[matched_jet_indices[2]].LorentzVector.Pt() )
            ptjet_of_fourthmin_dR_pair.Fill(this_event.good_jets_vector[matched_jet_indices[3]].LorentzVector.Pt() )

            # Get list of all permutations for list of the jet's Lorentz vectors
            permutations_list = list(permutations(match_ordered_jets,len(match_ordered_jets)))
            # print("list pt:", permutations_list[0].LorentzVector.Pt())
            perm_index = 0
            n_signal_perms = 0
            fourjet_pt_list = []
            good_perm_count_ = 0
            # Look at all possible permutations
            for perm_ in permutations_list:
                # variables used to reject 'bad' permutations
                is_good_perm = 1
                pt_list = []
                # List that will be added to the dataframe if permuation deemed reasonable
                tmp_list = []
                # Check if this permutation makes sense
                for jet_index in range(0,4):
                    pt_list.append(perm_[jet_index].LorentzVector.Pt())
                    # print("pt:",perm_[jet_index].LorentzVector.Pt())
                    if (perm_[jet_index].LorentzVector.Pt() >= 800 or perm_[jet_index].LorentzVector.Pt() == 1.000000e+11 or perm_[jet_index].LorentzVector.Pt() == 99):
                        print("pt:",perm_[jet_index].LorentzVector.Pt())
                        # Reject permutation: non-existent jet was permuted into first 4 jet positions
                        is_good_perm = 0
                # Reject permutation if the permutation is just of the additional (>4) jets
                if pt_list in fourjet_pt_list:
                    is_good_perm = 0
                else:
                    fourjet_pt_list.append(pt_list)

                if is_good_perm == 0:
                    # Reject bad/pointless permutation
                    continue
                # debug
                # for jet_index in range(0,4):
                #     print("jet pt :",perm_[jet_index].LorentzVector.Pt())
                #     print("all jet pt",match_ordered_jets[jet_index].LorentzVector.Pt())
                # ---------------#

                # Append the Variables for the matching 4 jets of this permutation
                n_jets_in_signal_pos = 0
                pt_perm_jets = []
                for jet_index in range(0,4):
                    pt_perm_jets.append(perm_[jet_index].LorentzVector.Pt())
                    tmp_list.append(np.log(perm_[jet_index].LorentzVector.Pt()))
                    tmp_list.append(perm_[jet_index].LorentzVector.Eta())
                    tmp_list.append(perm_[jet_index].LorentzVector.Phi())
                    tmp_list.append(np.log(perm_[jet_index].LorentzVector.E()))
                    # print("E:\t ",perm_[jet_index].LorentzVector.E())
                    tmp_list.append(perm_[jet_index].LorentzVector.DeltaR(this_event.photon1))
                    tmp_list.append(perm_[jet_index].LorentzVector.DeltaR(this_event.photon2))

                    # For the first four jets in this permutation
                    # Check if the pT matches any of the four jets matched jets
                    # We don't care which jet is matched -> invariant under WW(qqqq)
                    if perm_[jet_index].LorentzVector.Pt() == match_ordered_jets[0].LorentzVector.Pt():
                        n_jets_in_signal_pos+=1
                    if perm_[jet_index].LorentzVector.Pt() == match_ordered_jets[1].LorentzVector.Pt():
                        n_jets_in_signal_pos+=1
                    if perm_[jet_index].LorentzVector.Pt() == match_ordered_jets[2].LorentzVector.Pt():
                        n_jets_in_signal_pos+=1
                    if perm_[jet_index].LorentzVector.Pt() == match_ordered_jets[3].LorentzVector.Pt():
                        n_jets_in_signal_pos+=1

                eigvals_, eigvecs_ = momentum_tensor([
                perm_[0].LorentzVector,
                perm_[1].LorentzVector,
                perm_[2].LorentzVector,
                perm_[3].LorentzVector])
                spher_ = sphericity(eigvals_)
                tmp_list.append(spher_)

                # Append 1 if signal, 0 if background
                # print("n:",n_jets_in_signal_pos)
                if n_jets_in_signal_pos==4:
                    label=1
                    sphericity_correct_h.Fill(spher_)
                    jet1_pt_correct.Fill(perm_[0].LorentzVector.Pt())
                    jet1_eta_correct.Fill(perm_[0].LorentzVector.Eta())
                    dR_j1_photon1_correct.Fill(perm_[0].LorentzVector.DeltaR(this_event.photon1))
                    dR_j1_photon2_correct.Fill(perm_[0].LorentzVector.DeltaR(this_event.photon2))

                    jet2_pt_correct.Fill(perm_[1].LorentzVector.Pt())
                    jet2_eta_correct.Fill(perm_[1].LorentzVector.Eta())
                    dR_j2_photon1_correct.Fill(perm_[1].LorentzVector.DeltaR(this_event.photon1))
                    dR_j2_photon2_correct.Fill(perm_[1].LorentzVector.DeltaR(this_event.photon2))

                    jet3_pt_correct.Fill(perm_[2].LorentzVector.Pt())
                    jet3_eta_correct.Fill(perm_[2].LorentzVector.Eta())
                    dR_j3_photon1_correct.Fill(perm_[2].LorentzVector.DeltaR(this_event.photon1))
                    dR_j3_photon2_correct.Fill(perm_[2].LorentzVector.DeltaR(this_event.photon2))

                    jet4_pt_correct.Fill(perm_[3].LorentzVector.Pt())
                    jet4_eta_correct.Fill(perm_[3].LorentzVector.Eta())
                    dR_j4_photon1_correct.Fill(perm_[3].LorentzVector.DeltaR(this_event.photon1))
                    dR_j4_photon2_correct.Fill(perm_[3].LorentzVector.DeltaR(this_event.photon2))

                    jet1_btag_correct.Fill(perm_[0].BDisc)
                    jet2_btag_correct.Fill(perm_[1].BDisc)
                    jet3_btag_correct.Fill(perm_[2].BDisc)
                    jet4_btag_correct.Fill(perm_[3].BDisc)
                else:
                    label=0
                    # print("label=0")
                    sphericity_incorrect_h.Fill(spher_)
                    jet1_pt_incorrect.Fill(perm_[0].LorentzVector.Pt())
                    jet1_eta_incorrect.Fill(perm_[0].LorentzVector.Eta())
                    dR_j1_photon1_incorrect.Fill(perm_[0].LorentzVector.DeltaR(this_event.photon1))
                    dR_j1_photon2_incorrect.Fill(perm_[0].LorentzVector.DeltaR(this_event.photon2))

                    jet2_pt_incorrect.Fill(perm_[1].LorentzVector.Pt())
                    jet2_eta_incorrect.Fill(perm_[1].LorentzVector.Eta())
                    dR_j2_photon1_incorrect.Fill(perm_[1].LorentzVector.DeltaR(this_event.photon1))
                    dR_j2_photon2_incorrect.Fill(perm_[1].LorentzVector.DeltaR(this_event.photon2))

                    jet3_pt_incorrect.Fill(perm_[2].LorentzVector.Pt())
                    jet3_eta_incorrect.Fill(perm_[2].LorentzVector.Eta())
                    dR_j3_photon1_incorrect.Fill(perm_[2].LorentzVector.DeltaR(this_event.photon1))
                    dR_j3_photon2_incorrect.Fill(perm_[2].LorentzVector.DeltaR(this_event.photon2))

                    jet4_pt_incorrect.Fill(perm_[3].LorentzVector.Pt())
                    jet4_eta_incorrect.Fill(perm_[3].LorentzVector.Eta())
                    dR_j4_photon1_incorrect.Fill(perm_[3].LorentzVector.DeltaR(this_event.photon1))
                    dR_j4_photon2_incorrect.Fill(perm_[3].LorentzVector.DeltaR(this_event.photon2))

                    jet1_btag_incorrect.Fill(perm_[0].BDisc)
                    jet2_btag_incorrect.Fill(perm_[1].BDisc)
                    jet3_btag_incorrect.Fill(perm_[2].BDisc)
                    jet4_btag_incorrect.Fill(perm_[3].BDisc)

                if label==1:
                    n_signal_perms+=1

                #if label == 0:
                #    print('Matched jets pt: %s , %s , %s , %s' % (match_ordered_jets[0].LorentzVector.Pt(), match_ordered_jets[1].LorentzVector.Pt(), match_ordered_jets[2].LorentzVector.Pt(), match_ordered_jets[3].LorentzVector.Pt()))
                #    print("Perm jet 0 pt = %s, Perm jet 1 pt = %s, Perm jet 2 pt = %s, Perm jet 3 pt = %s" % (perm_[0].LorentzVector.Pt(), perm_[1].LorentzVector.Pt(), perm_[2].LorentzVector.Pt(), perm_[3].LorentzVector.Pt()))
                #    print("Signal/background: ", label)

                tmp_list.append(label)
                tmp_list.append(event_ID)
                tmp_list.append(input_ttree.nGoodAK4jets)
                perm_index += 1
                data.append(tmp_list)
                good_perm_count_ += 1

            # Control histograms
            # print('N ordered jets: ', len(match_ordered_jets))
            # print('N jets: ' , input_ttree.nGoodAK4jets)
            # print('N perms: ' , len(permutations_list))
            # print('N good perms: ',good_perm_count_)
            # print('N siganl perms: ', n_signal_perms)
            W1_eta_phi_h.Fill(this_event.gen_W_1.Eta(), this_event.gen_W_1.Phi())
            W2_eta_phi_h.Fill(this_event.gen_W_2.Eta(), this_event.gen_W_2.Phi())
            j1_eta_phi_h.Fill(this_event.jet0.LorentzVector.Eta(), this_event.jet0.LorentzVector.Phi())

        kin_file_name = os.path.join(output_dir,"kinematics_file.root")
        kinematics_file = ROOT.TFile(kin_file_name,"recreate")

        # Write histograms to file
        sphericity_correct_h.Write()
        sphericity_incorrect_h.Write()

        min_dR_jet_quark.Write()
        submin_dR_jet_quark.Write()
        thirdmin_dR_jet_quark.Write()
        fourthmin_dR_jet_quark.Write()
        ptjet_of_min_dR_pair.Write()
        ptjet_of_submin_dR_pair.Write()
        ptjet_of_thirdmin_dR_pair.Write()
        ptjet_of_fourthmin_dR_pair.Write()

        nmatched_jets_h.Write()
        n_jets_h.Write()
        jet1_pt_correct.Write()
        jet1_eta_correct.Write()
        dR_j1_photon1_correct.Write()
        dR_j1_photon2_correct.Write()
        jet1_pt_incorrect.Write()
        jet1_eta_incorrect.Write()
        dR_j1_photon1_incorrect.Write()
        dR_j1_photon2_incorrect.Write()

        jet2_pt_correct.Write()
        jet2_eta_correct.Write()
        dR_j2_photon1_correct.Write()
        dR_j2_photon2_correct.Write()
        jet2_pt_incorrect.Write()
        jet2_eta_incorrect.Write()
        dR_j2_photon1_incorrect.Write()
        dR_j2_photon2_incorrect.Write()

        jet3_pt_correct.Write()
        jet3_eta_correct.Write()
        dR_j3_photon1_correct.Write()
        dR_j3_photon2_correct.Write()
        jet3_pt_incorrect.Write()
        jet3_eta_incorrect.Write()
        dR_j3_photon1_incorrect.Write()
        dR_j3_photon2_incorrect.Write()

        jet4_pt_correct.Write()
        jet4_eta_correct.Write()
        dR_j4_photon1_correct.Write()
        dR_j4_photon2_correct.Write()
        jet4_pt_incorrect.Write()
        jet4_eta_incorrect.Write()
        dR_j4_photon1_incorrect.Write()
        dR_j4_photon2_incorrect.Write()
        W1_eta_phi_h.Write()
        W2_eta_phi_h.Write()

        jet1_btag_correct.Write()
        jet2_btag_correct.Write()
        jet3_btag_correct.Write()
        jet4_btag_correct.Write()
        jet1_btag_incorrect.Write()
        jet2_btag_incorrect.Write()
        jet3_btag_incorrect.Write()
        jet4_btag_incorrect.Write()

        kinematics_file.Close()

        jet_histo_pairs = [
        [jet1_pt_correct,jet1_pt_incorrect],
        [jet2_pt_correct,jet2_pt_incorrect],
        [jet3_pt_correct,jet3_pt_incorrect],
        [jet4_pt_correct,jet4_pt_incorrect],
        ]
        axis_titles = [["reco 1 jet pT"],["reco 2 jet pT"],["reco 3 jet pT"],["reco 4 jet pT"]]
        dimensions = 1
        save_titles_ = [
        os.path.join(output_dir,"reco_jet1_pt.png"),
        os.path.join(output_dir,"reco_jet2_pt.png"),
        os.path.join(output_dir,"reco_jet3_pt.png"),
        os.path.join(output_dir,"reco_jet4_pt.png")
        ]
        draw_style = "HIST"
        # Normalise histograms
        for jet_hist_pair in jet_histo_pairs:
            for hist_ in jet_hist_pair:
                hist_.Scale(1/hist_.Integral(0,hist_.GetNbinsX()+1))
        for index_ in range(0,len(jet_histo_pairs)):
            make_plot_(jet_histo_pairs[index_],axis_titles[index_],dimensions,save_titles_[index_],draw_style)


        jet_histo_pairs = [
        [jet1_eta_correct,jet1_eta_incorrect],
        [jet2_eta_correct,jet2_eta_incorrect],
        [jet3_eta_correct,jet3_eta_incorrect],
        [jet4_eta_correct,jet4_eta_incorrect],
        ]
        axis_titles = [["reco 1 jet eta"],["reco 2 jet eta"],["reco 3 jet eta"],["reco 4 jet eta"]]
        dimensions = 1
        save_titles_ = [
        os.path.join(output_dir,"reco_jet1_eta.png"),
        os.path.join(output_dir,"reco_jet2_eta.png"),
        os.path.join(output_dir,"reco_jet3_eta.png"),
        os.path.join(output_dir,"reco_jet4_eta.png")
        ]
        draw_style = "HIST"
        # Normalise histograms
        for jet_hist_pair in jet_histo_pairs:
            for hist_ in jet_hist_pair:
                hist_.Scale(1/hist_.Integral(0,hist_.GetNbinsX()+1))
        for index_ in range(0,len(jet_histo_pairs)):
            make_plot_(jet_histo_pairs[index_],axis_titles[index_],dimensions,save_titles_[index_],draw_style)



        jet_histo_pairs = [
        [jet1_btag_correct,jet1_btag_incorrect],
        [jet2_btag_correct,jet2_btag_incorrect],
        [jet3_btag_correct,jet3_btag_incorrect],
        [jet4_btag_correct,jet4_btag_incorrect],
        ]
        axis_titles = [["reco 1 jet BDisc"],["reco 2 jet BDisc"],["reco 3 jet BDisc"],["reco 4 jet BDisc"]]
        dimensions = 1
        save_titles_ = [
        os.path.join(output_dir,"reco_jet1_BDisc.png"),
        os.path.join(output_dir,"reco_jet2_BDisc.png"),
        os.path.join(output_dir,"reco_jet3_BDisc.png"),
        os.path.join(output_dir,"reco_jet4_BDisc.png")
        ]
        draw_style = "HIST"
        # Normalise histograms
        for jet_hist_pair in jet_histo_pairs:
            for hist_ in jet_hist_pair:
                hist_.Scale(1/hist_.Integral(0,hist_.GetNbinsX()+1))
        for index_ in range(0,len(jet_histo_pairs)):
            make_plot_(jet_histo_pairs[index_],axis_titles[index_],dimensions,save_titles_[index_],draw_style)



        jet_histo_pairs = [
        [dR_j1_photon1_correct,dR_j1_photon1_incorrect],
        [dR_j2_photon1_correct,dR_j2_photon1_incorrect],
        [dR_j3_photon1_correct,dR_j3_photon1_incorrect],
        [dR_j4_photon1_correct,dR_j4_photon1_incorrect],
        ]
        axis_titles = [
        ["dR(reco jet 1, leading photon)"],
        ["dR(reco jet 2, leading photon)"],
        ["dR(reco jet 3, leading photon)"],
        ["dR(reco jet 4, leading photon)"]]
        dimensions = 1
        save_titles_ = [
        os.path.join(output_dir,"dR_j1_photon1.png"),
        os.path.join(output_dir,"dR_j2_photon1.png"),
        os.path.join(output_dir,"dR_j3_photon1.png"),
        os.path.join(output_dir,"dR_j4_photon1.png")
        ]
        draw_style = "HIST"
        # Normalise histograms
        for jet_hist_pair in jet_histo_pairs:
            for hist_ in jet_hist_pair:
                hist_.Scale(1/hist_.Integral(0,hist_.GetNbinsX()+1))
        for index_ in range(0,len(jet_histo_pairs)):
            make_plot_(jet_histo_pairs[index_],axis_titles[index_],dimensions,save_titles_[index_],draw_style)

        jet_histo_pairs = [
        [dR_j1_photon2_correct,dR_j1_photon2_incorrect],
        [dR_j2_photon2_correct,dR_j2_photon2_incorrect],
        [dR_j3_photon2_correct,dR_j3_photon2_incorrect],
        [dR_j4_photon2_correct,dR_j4_photon2_incorrect],
        ]
        axis_titles = [
        ["dR(reco jet 1, subleading photon)"],
        ["dR(reco jet 2, subleading photon)"],
        ["dR(reco jet 3, subleading photon)"],
        ["dR(reco jet 4, subleading photon)"]]
        dimensions = 1
        save_titles_ = [
        os.path.join(output_dir,"dR_j1_photon2.png"),
        os.path.join(output_dir,"dR_j2_photon2.png"),
        os.path.join(output_dir,"dR_j3_photon2.png"),
        os.path.join(output_dir,"dR_j4_photon2.png")
        ]
        draw_style = "HIST"
        # Normalise histograms
        for jet_hist_pair in jet_histo_pairs:
            for hist_ in jet_hist_pair:
                hist_.Scale(1/hist_.Integral(0,hist_.GetNbinsX()+1))
        for index_ in range(0,len(jet_histo_pairs)):
            make_plot_(jet_histo_pairs[index_],axis_titles[index_],dimensions,save_titles_[index_],draw_style)

        histograms = [sphericity_correct_h,sphericity_incorrect_h]
        for hist_ in histograms:
            hist_.Scale(1/hist_.Integral(0,hist_.GetNbinsX()+1))
        axis_titles = ["Sphericity"]
        dimensions = 1
        save_title = os.path.join(output_dir,"sphericity.png")
        draw_style = "HIST"
        make_plot_(histograms,axis_titles,dimensions,save_title,draw_style)

        # Draw histograms
        histograms = [W1_eta_phi_h,W2_eta_phi_h,j1_eta_phi_h]
        axis_titles = ["Eta","Phi"]
        dimensions = 2
        save_title = os.path.join(output_dir,"W1W2_etaphi_Surface.png")
        draw_style = "SURF"
        make_plot_(histograms,axis_titles,dimensions,save_title,draw_style)

        histograms = [nmatched_jets_h]
        axis_titles = ["# matched jets [dR(genQ,recojet)<0.4]"]
        dimensions = 1
        save_title = os.path.join(output_dir,"N_matched_jets.png")
        draw_style = "HIST"
        make_plot_(histograms,axis_titles,dimensions,save_title,draw_style)

        histograms = [min_dR_jet_quark,submin_dR_jet_quark,thirdmin_dR_jet_quark,fourthmin_dR_jet_quark]
        axis_titles = ["min[dR(recojet,genQ)]"]
        dimensions = 1
        save_title = os.path.join(output_dir,"min_dR_jet_quark.png")
        draw_style = "HIST"
        make_plot_(histograms,axis_titles,dimensions,save_title,draw_style)

        histograms = [ptjet_of_min_dR_pair,ptjet_of_submin_dR_pair,ptjet_of_thirdmin_dR_pair,ptjet_of_fourthmin_dR_pair]
        axis_titles = ["pT of jets with min[dR(recojet,genQ)]"]
        dimensions = 1
        save_title = os.path.join(output_dir,"jet_pt_for_min_dR.png")
        draw_style = "HIST"
        make_plot_(histograms,axis_titles,dimensions,save_title,draw_style)

        histograms = [n_jets_h]
        axis_titles = ["# reco jets (>=4 jet presel)"]
        dimensions = 1
        save_title = os.path.join(output_dir,"N_reco_jets.png")
        draw_style = "HIST"
        make_plot_(histograms,axis_titles,dimensions,save_title,draw_style)
 
        input_ttree.Delete()
        tfile.Close()

    df = pd.DataFrame(data)
    df.columns = column_headers
    df = df.replace(to_replace=-999.000,value=-9.0)
    # df = df.replace(to_replace=99.000,value=-9.0)
    df = df.replace(to_replace=-1.000000e+11,value=-9.0)
    df = df.replace(to_replace=1.000000e+11,value=-9.0)
    df = df.fillna(-9.0)
    df.to_csv(os.path.join(output_dir,"HH_dataframe.csv"), index=False)
    return

def main():

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing input files', default='./', type=str)
    parser.add_argument('-o', '--outputs', dest='output_dir', help='Name of output directory', default='test', type=str)
    args = parser.parse_args()
    inputs_file_path = args.inputs_file_path
    output_dir = args.output_dir
    if os.path.isdir(output_dir) != 1:
        os.mkdir(output_dir)

    # Create list of headers for dataset .csv
    column_headers = ['HW1_jet1_pt','HW1_jet1_eta','HW1_jet1_phi','HW1_jet1_E','dRj1_photon1','dRj1_photon2','HW1_jet2_pt','HW1_jet2_eta','HW1_jet2_phi','HW1_jet2_E','dRj2_photon1','dRj2_photon2','HW1_jet3_pt','HW1_jet3_eta','HW1_jet3_phi','HW1_jet3_E','dRj3_photon1','dRj3_photon2','HW1_jet4_pt','HW1_jet4_eta','HW1_jet4_phi','HW1_jet4_E','dRj4_photon1','dRj4_photon2','sphericity','target','event_ID','njets']

    # Load ttree into .csv including all variables listed in column_headers
    newdataframename = os.path.join(output_dir,'output_dataframe.csv')
    print('Creating new data .csv @: %s . . . . ' % (inputs_file_path))

    load_data(inputs_file_path,column_headers,output_dir)

    exit(0)

main()