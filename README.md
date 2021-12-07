# HHRecoNN
### Authors: Joshuha Thomas-Wilsker
### Institutes: IHEP Beijing, CERN
Package used to train a reconstruction neural network for HH->WWyy analysis.

## Environment settings
I run this on my local machine (mac OSX) inside a conda environment. *If you have root access*, first setup a conda environment for python 3.7:
```
conda create -n <env_title> python=3.7 (anaconda)
```
Check the python version you are now using:
```
python --version
```
I use several non-standard libraries which must be present in your working environment. If they aren't, you may encounter problems. To see which libraries are currently available inside your environment use the following command inside once the env is setup:
```
conda list
```
Check for the following libraries:
- python 3.7
- shap
- keras
- tensorflow
- root
- root_numpy
- numpy (numpy 2.X is not compatible with TF so use numpy==1.19.5 if designing models with TF)
- sklearn
- matplotlib
- seaborn
If any packages are missing the code, you can add the package to the environment easily assuming it doesn't clash or require something you haven't got in the environment setup:
```
conda install <new_library>
```
## Datasets
Signal samples taken from here:
```
/eos/user/f/fmonti/HHWWgg/ntuples/FlashggNtuples_WithMoreVars/
```
Use the script 'split_root_file.py' to copy large files across. Splits into several files and only the nominal ttree (ignoring systematic variations).
## The code
The code consists of three distinct steps.
Step 1: We generate a dataset from the root file(s)
Step 2: Training the algorithm
Step 3: Validating the algorithm -> use training script with option '-f 0'

### Step 1
This is done using the generate_dataframe.py script. The script can be run using a command similar to the one beneath:
```
python generate_dataframe.py -i <input directory> -o <output directory>
```
This will generate a .csv file in the output directory using the .root file in the input directory. Here, a basic selection can be implemented.

The dataset is generated using the permutations of jets in each event. First, jets are matched to their closest (minimum dR) generator level quark (from the Higgs->WW->qqqq decay chain). Each quark will have only one match (its closest jet) and each jet can only be matched to one quark. Given there are only ever 4 generator level quarks in an event, if there are more than 4 jets in an event, the additional jets will not be matched.

We then form a list of jets, starting from the best matched jet and ending with the unmatched jets. Only events with 4 'good' matches are kept. The dataset is then generated using all possible permutations of these jets, leaving N! examples to train on for each event, where N! is the number of jets one requires in the event selection.

The 'correct' permutation is considered signal, where the 4 matched jets are in the first 4 positions of the list ordered by dR. Other permutations are considered background so we have an imbalance in the dataset, which is remedied using class weights during the training step.

NOTE: Could the definition of 'signal' be changed so they don't have to be ordered and we take any permutation with the matched jets in the first 4 positions?

### Step 2
Training the algorithm is done using the following command:
```
python train_RECONN.py -i <input .csv file> -o <output dir> -j <input variable .json file> -f <train or use pre-trained model>
```
The training is done using the tensorflow back-end in Keras. It uses the labelled dataset from step one as input and outputs predictions for jet permutations.

The networks architecture can be altered in the 'compile_model' function and hyper-parameters can be changed inside the 'model.fit()' function. The trained model is stored in the output directory along with plots to designed to check the performance of the algorithm. The plotting requires the classes and functions inside the 'plotting' package.

### Step 3
The application stage is performed by the train_RECONN.py script with the -f option set to 0.

The dataset is taken from a predefined input dataset (constructed in the same way as for the training). A high score (response->1) means the DNN believes this permutation is likely to be correct. One then takes the permutation of jets with the highest score to be the correct. If the highest score is == 1, we selected a correct permutation. One then uses this to assign the 4 jets in this permutation to the Higgs.
