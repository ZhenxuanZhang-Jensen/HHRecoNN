import ROOT
import numpy

f = ROOT.TFile("/eos/user/f/fmonti/HHWWgg/ntuples/FlashggNtuples_WithMoreVars/2018/GluGluToHHTo2G4Q_node_cHHH0.root")
inputTree = f.Get("tagsDumper/trees/GluGluToHHTo2G4Q_node_cHHH0_13TeV_HHWWggTag_1")
inputTree.SetBranchStatus("*",1)
nentries = inputTree.GetEntries()

subfiles = 5
events_per_file = nentries/subfiles

subfile_num = 0
subfile = ROOT.TFile("GluGluToHHTo2G4Q_node_cHHH0_%s.root"%subfile_num,"recreate")
subfile_tree = inputTree.CloneTree(0)

print("Events per file: ", events_per_file)

n_entries_in_batch = 0
for entry in range(0,nentries):
    inputTree.GetEntry(entry)
    n_entries_in_batch+=1
    if n_entries_in_batch<events_per_file:
        subfile_tree.Fill()
    else:
        subfile.Write()
        subfile.Close()
        n_entries_in_batch=0
        subfile_num+=1
        if subfile_num > subfiles:
            print("Filled %s subfiles. Now exiting." % subfiles)
            exit(0)
        else:
            print('Moving to next file: ', subfile_num)
            new_subfile_name = "GluGluToHHTo2G4Q_node_cHHH0_%s.root"%subfile_num
            subfile = ROOT.TFile(new_subfile_name,"recreate")
            subfile_tree = inputTree.CloneTree(0)
