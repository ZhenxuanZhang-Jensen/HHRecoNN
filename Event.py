import ROOT
import numpy as np
from numpy import linalg
import pandas as pd
from ROOT import TTree, TFile
from array import array
import root_numpy
from root_numpy import tree2array
from collections import OrderedDict
from itertools import permutations
import uuid
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
        self.jet0.index = 0

        # self.jet0.SetBTag(input_ttree.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet1 = Jet()
        self.jet1.SetLorentzVector(input_ttree.FullyResolved_Jet2_pt,input_ttree.FullyResolved_Jet2_eta,input_ttree.FullyResolved_Jet2_phi,input_ttree.FullyResolved_Jet2_M)
        self.jet1.index = 1
        # self.jet1.SetBTag(input_ttree.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet2 = Jet()
        self.jet2.SetLorentzVector(input_ttree.FullyResolved_Jet3_pt,input_ttree.FullyResolved_Jet3_eta,input_ttree.FullyResolved_Jet3_phi,input_ttree.FullyResolved_Jet3_M)
        self.jet2.index = 2
        # self.jet2.SetBTag(input_ttree.goodJets_2_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_2_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_2_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet3 = Jet()
        self.jet3.SetLorentzVector(input_ttree.FullyResolved_Jet4_pt,input_ttree.FullyResolved_Jet4_eta,input_ttree.FullyResolved_Jet4_phi,input_ttree.FullyResolved_Jet4_M)
        self.jet3.index = 3
        
        # self.jet3.SetBTag(input_ttree.goodJets_3_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_3_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_3_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet4 = Jet()
        self.jet4.SetLorentzVector(input_ttree.FullyResolved_Jet5_pt,input_ttree.FullyResolved_Jet5_eta,input_ttree.FullyResolved_Jet5_phi,input_ttree.FullyResolved_Jet5_M)
        self.jet4.index = 4
        # self.jet4.SetBTag(input_ttree.goodJets_4_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_4_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_4_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet5 = Jet()
        self.jet5.SetLorentzVector(input_ttree.FullyResolved_Jet6_pt,input_ttree.FullyResolved_Jet6_eta,input_ttree.FullyResolved_Jet6_phi,input_ttree.FullyResolved_Jet6_M)
        self.jet5.index = 5
        # self.jet5.SetBTag(input_ttree.goodJets_5_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_5_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_5_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.jet6 = Jet()
        self.jet6.SetLorentzVector(input_ttree.FullyResolved_Jet7_pt,input_ttree.FullyResolved_Jet7_eta,input_ttree.FullyResolved_Jet7_phi,input_ttree.FullyResolved_Jet7_M)
        self.jet6.index = 6
        # self.jet6.SetBTag(input_ttree.goodJets_6_bDiscriminator_mini_pfDeepFlavourJetTags_probb + input_ttree.goodJets_6_bDiscriminator_mini_pfDeepFlavourJetTags_probbb + input_ttree.goodJets_6_bDiscriminator_mini_pfDeepFlavourJetTags_problepb)

        self.good_jet_lorentz_vectors = [self.jet0.LorentzVector,self.jet1.LorentzVector,self.jet2.LorentzVector,self.jet3.LorentzVector,self.jet4.LorentzVector,self.jet5.LorentzVector,self.jet6.LorentzVector]
        self.good_jets_vector = [self.jet0,self.jet1,self.jet2,self.jet3,self.jet4,self.jet5,self.jet6]

        self.photon1 = ROOT.TLorentzVector()
        self.photon1.SetPtEtaPhiE(input_ttree.pho1_pt,input_ttree.pho1_eta,input_ttree.pho1_phi,input_ttree.pho1_E)
        self.photon2 = ROOT.TLorentzVector()
        self.photon2.SetPtEtaPhiE(input_ttree.pho2_pt,input_ttree.pho2_eta,input_ttree.pho2_phi,input_ttree.pho2_E)
def sphericity(eigvals):
    # Definition: http://sro.sussex.ac.uk/id/eprint/44644/1/art%253A10.1007%252FJHEP06%25282010%2529038.pdf
    spher_ = 2*eigvals[0] / (eigvals[1]+eigvals[0])
    return spher_
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