import numpy as np
#import shap
import pandas
import os
import sklearn
import subprocess
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class plotter(object):

    def __init__(self):
        self.separations_categories = []
        self.output_directory = ''
        self.nbins = np.linspace(0.0,1.0,num=50)
        w, h = 4, 4
        self.yscores_train_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_test_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_train_non_categorised = [[0 for x in range(w)] for y in range(h)]
        self.yscores_test_non_categorised = [[0 for x in range(w)] for y in range(h)]
        self.plots_directory = ''
        pass

    def save_plots(self, dir='', filename=''):
        self.check_dir(dir)
        filepath = os.path.join(dir,filename)
        self.fig.savefig(filepath)
        return self.fig

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def set_plot_labels(self,plot_title):
        plt.clf() # clear figure
        plt.cla() # clear axes
        plt.title(plot_title[0])
        plt.ylabel(plot_title[1])
        plt.xlabel(plot_title[2])
        plt.legend(loc='upper right')

    def correlation_matrix(self, data, **kwds):
        self.data = data.iloc[:, :-4]
        self.labels = self.data.corr(**kwds).columns.values
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(20,20))
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : 1*100, "annot_kws" : {"size":8}, "cmap" : plt.get_cmap("Blues",20), 'fmt' : '0.2f',}
        self.ax1.set_title("Correlations")
        sns.heatmap(self.data.corr(method='spearman')*100, **opts)
        for ax in (self.ax1,):
            # Shift tick location to bin centre
            ax.set_xticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_xticklabels(self.labels, minor=False, ha='right', rotation=45)
            ax.set_yticklabels(self.labels, minor=False, rotation=45)
        plt.tight_layout()
        return

    def ROC(self,train_label,train_pred,test_label,test_pred):
        # ROC curve and AUC calculated using absolute number of examples, not weighted events.
        # ROC curve creation: scans the output distribution and calculates the true and false
        # positive rates (tpr, fpr) for given thresholds.
        fpr_keras_test, tpr_keras_test, thresholds_keras_test = roc_curve(test_label, test_pred.ravel())
        print("thresholds_keras_test: " , thresholds_keras_test)
        print("tpr_keras_test: " , tpr_keras_test)
        print("fpr_keras_test: " , fpr_keras_test)
        auc_keras_test = auc(fpr_keras_test, tpr_keras_test)
        fpr_keras_train, tpr_keras_train, thresholds_keras_train = roc_curve(train_label, train_pred.ravel())
        auc_keras_train = auc(fpr_keras_train, tpr_keras_train)
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras_test, tpr_keras_test, label='Test (area = {:.3f})'.format(auc_keras_test))
        plt.plot(fpr_keras_train, tpr_keras_train, label='Train (area = {:.3f})'.format(auc_keras_train))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.tight_layout()
        return

    def history_plot(self, history, label='accuracy'):
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        plt.plot(history.history[label])
        plt.plot(history.history['val_'+label])
        plt.title('model '+label)
        plt.ylabel(label)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.tight_layout()
        return

    def GetSeparation(self, hist_sig, hist_bckg):
        # compute "separation" defined as
        # <s2> = (1/2) Int_-oo..+oo { (S(x) - B(x))^2/(S(x) + B(x)) dx }
        minima = np.minimum(hist_sig, hist_bckg)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_bckg))
        return intersection

    def draw_binary_overfitting_plot(self, y_scores_train, y_scores_test, plot_info, test_weights):
        colours = plot_info[0]
        data_type = plot_info[1]
        plots_dir = plot_info[2]
        plot_title = plot_info[3]
        name = filter(str.isalnum, str(data_type).split(".")[-1])
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.675,0.7375,0.8,0.8625,0.9375,1.0])

        index=0
        for index in range(0,len(y_scores_train)):
            train_bin_errors = np.zeros(len(bin_edges_low_high)-1)
            test_bin_errors = np.zeros(len(bin_edges_low_high)-1)

            y_train = y_scores_train[index]
            y_test = y_scores_test[index]
            colour = colours[index]
            width = np.diff(bin_edges_low_high)
            if index==0:
                print('<plotter> Overfitting plot: Signal')
                label='signal'
            if index==1:
                label='bckg'
                print('<plotter> Overfitting plot: Background')

            # Setup training histograms
            histo_train_, bin_edges = np.histogram(y_train, bins=bin_edges_low_high)
            dx_scale_train =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])

            bin_errors_sumw2 = 0

            # Scale training histograms to: (hist / sum of histogram entries) / (range / number of bins)
            # so like scaling hist relative to its integral.
            # then scaling the result relative to its avergae bin width.
            histo_train_ = (histo_train_ / np.sum(histo_train_, dtype=np.float32)) / dx_scale_train
            plt.bar(bincenters, histo_train_, width=width, color=colour, edgecolor=colour, alpha=0.5, label=label+' training')

            if index == 0:
                histo_train_sig = histo_train_
            if index == 1:
                histo_train_bckg = histo_train_

            histo_test_, bin_edges = np.histogram(y_test, bins=bin_edges_low_high)
            dx_scale_test =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])

            # Protection against low stats in validation dataset.
            if np.sum(histo_test_, dtype=np.float32) <= 0 :
                histo_test_ = histo_test_
                err = 0
                plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=label+' testing')
                if index == 0:
                    histo_test_sig = histo_test_
                if index == 1:
                    histo_test_bckg = histo_test_
            else:
                # Currently just taking the sqrt of the bin entry
                # Correct way:
                # Errors calculated using error propogation and the histograms intrinsic poissonian statistics
                # err(sum weights)^2 == variance on the sum of weights = sum of the variance of each weight =  sum{var(w_i)} [i=1,2,N]
                # Varianceof weight i is determined by the statistical fluctuation of the number of events considered.
                # var(w_i) = var(w_i * 1 event) = w_i^2 * var(1 event) = w_i^2
                # err(sum weights) = sqrt( sum{var(w_i)}[i=1,2,N] )
                #                  = sqrt( sum{w_i^2}[i=1,2,N] )
                bin_errors_sumw2 = 0
                for yval_index in range(0,len(y_scores_test[0])):
                    for bin_index in range(0,len(bin_edges_low_high)-1):
                        bin_low_edge = bin_edges_low_high[bin_index]
                        bin_high_edge = bin_edges_low_high[bin_index+1]
                        if y_scores_test[0][yval_index] > bin_low_edge and y_scores_test[0][yval_index] < bin_high_edge:
                            test_bin_errors[[bin_index]] += 1#test_weights[yval_index]**2
                # Take square root of sum of bins.
                test_bin_errors = (np.sqrt(test_bin_errors)/np.sum(histo_test_, dtype=np.float32)) / dx_scale_test
                histo_test_ = ( histo_test_ / np.sum(histo_test_, dtype=np.float32) ) / dx_scale_test
                plt.errorbar(bincenters, histo_test_, yerr=test_bin_errors, fmt='o', c=colour, label=label+' testing')
                if index == 0:
                    histo_test_sig = histo_test_
                if index == 1:
                    histo_test_bckg = histo_test_

        train_SvsBSep = "{0:.5g}".format(self.GetSeparation(histo_train_sig,histo_train_bckg))
        test_SvsBSep = "{0:.5g}".format(self.GetSeparation(histo_test_sig,histo_test_bckg))

        S_v_B_train_sep = 'SvsB train Sep.: %s' % ( train_SvsBSep )
        self.ax.annotate(S_v_B_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
        S_v_B_test_sep = 'SvsB test Sep.: %s' % ( test_SvsBSep )
        self.ax.annotate(S_v_B_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.5), fontsize=9)

        separations_forTable = r'''%s & \textbackslash ''' % (S_v_B_test_sep)

        title_ = '%s output node' % (plot_title)
        plt.title(title_)
        label_name = 'Output Score'
        plt.xlabel(label_name)
        plt.ylabel('(1/N)dN/dX')

        leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=9)
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')

        #overfitting_plot_file_name = 'overfitting_plot_BinaryClassifier_%s.png' % (plot_title)
        #plots_dir = self.plots_directory
        #print('Saving : %s/%s' % (plots_dir, overfitting_plot_file_name))
        #self.save_plots(dir=plots_dir, filename=overfitting_plot_file_name)
        return separations_forTable

    def separation_table(self , outputdir):
        content = r'''\documentclass{article}
\begin{document}
\begin{center}
\begin{table}
\begin{tabular}{| c | c | c | c | c |} \hline
Node \textbackslash Background & HH & yyjets & GJets & DY \\ \hline
HH & %s \\
yyjets & %s \\
GJets & %s \\
DY & %s \\ \hline
\end{tabular}
\caption{Separation power on each output node. The separation is given with respect to the `signal' process the node is trained to separate (one node per row) and the background processes for that node (one background per column).}
\end{table}
\end{center}
\end{document}
'''
        table_path = os.path.join(outputdir,'separation_table')
        table_tex = table_path+'.tex'
        print('table_tex: ', table_tex)
        with open(table_tex,'w') as f:
            f.write( content % (self.separations_categories[0], self.separations_categories[1], self.separations_categories[2], self.separations_categories[3] ) )
        return

    def binary_overfitting(self, estimator, Y_train, Y_test, result_probs, result_probs_test, train_weights, test_weights, nbins=50):

        model = estimator
        data_type = type(model)

        #Arrays to store all results
        y_scores_train_signal_sample = []
        y_scores_train_bckg_sample = []
        y_scores_test_signal_sample = []
        y_scores_test_bckg_sample = []
        for i in range(0,len(result_probs)-1):
            train_event_weight = train_weights[i]
            if Y_train[i] == 1:
                y_scores_train_signal_sample.append(result_probs[i])
            if Y_train[i] == 0:
                y_scores_train_bckg_sample.append(result_probs[i])
        for i in range(0,len(result_probs_test)-1):
            test_event_weight = test_weights[i]
            if Y_test[i] == 1:
                y_scores_test_signal_sample.append(result_probs_test[i])
            if Y_test[i] == 0:
                y_scores_test_bckg_sample.append(result_probs_test[i])

        # Create 2D lists (dimension 2x2) to hold max DNN discriminator values for each sample. One for train data, one for test data.
        yscores_train_binary=[]
        yscores_test_binary=[]
        yscores_train_binary.append([y_scores_train_signal_sample, y_scores_train_bckg_sample])
        yscores_test_binary.append([y_scores_test_signal_sample, y_scores_test_bckg_sample])

        counter =0
        separations_all = []
        for y_scores_train_nonCat,y_scores_test_nonCat in zip(yscores_train_binary,yscores_test_binary):
            colours = ['r','steelblue']
            plot_title = 'Binary'
            plot_info = [colours,data_type,self.plots_directory,plot_title]
            separations_all.append(self.draw_binary_overfitting_plot(y_scores_train_nonCat,y_scores_test_nonCat,plot_info,test_weights))
            counter = counter+1
        return

    def plot_dot(self, title, x, shap_values, column_headers):
        plt.figure()
        if x is None:
          print('<plotter> No x defined. Leaving class function')
          return
        shap.summary_plot(shap_values[0], features=x, feature_names=column_headers, show=False, max_display=10)
        plt.gca().set_title(title)
        plt.tight_layout()
        plt.savefig("{}/plots/{}.png".format(self.output_directory, title), bbox_inches='tight')

    def plot_dot_bar(self, title, x, shap_values, column_headers):
        plt.figure()
        if x is None:
            print('<plotter> No x defined. Leaving class function')
            return
        shap.summary_plot(shap_values[0], features=x, feature_names=column_headers, show=False,plot_type='bar',max_display=10)
        plt.gca().set_title(title)
        plt.tight_layout()
        plt.savefig("{}/plots/{}.png".format(self.output_directory,title), bbox_inches='tight')

    def plot_dot_bar_all(self , title, x, shap_values, column_headers):
        plt.figure()
        if x is None:
            print('<plotter> No x defined. Leaving class function')
            return
        shap.summary_plot(shap_values[0], features=x, feature_names=column_headers, show=False,plot_type='bar',max_display=len(column_headers))
        plt.gca().set_title(title)
        plt.tight_layout()
        plt.savefig("{}/plots/{}.png".format(self.output_directory,title), bbox_inches='tight')
