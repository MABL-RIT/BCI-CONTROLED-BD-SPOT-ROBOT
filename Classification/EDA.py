#%%
import numpy as np
import os.path as op
import pyxdf
import pandas as pd
import mne
from scipy import signal
import matplotlib.pyplot as plt


#%%
files = ['../dataset/trial0/1_eeg.xdf', 
        '../dataset/trial0/2_eeg.xdf', 
        '../dataset/trial0/3_eeg.xdf']

df = pd.DataFrame()
for f in files:
    streams, header = pyxdf.load_xdf(f)
    eeg = streams[1]["time_series"][:, 0:8]
    eeg_ts = streams[1]["time_stamps"]

    eeg_df = pd.DataFrame(eeg)
    eeg_df.columns = ['Fp1', 'Fp2', 'F4', 'F3', 'T7', 'T8', 'P4', 'P3']
    eeg_df['ts'] = eeg_ts
 
    stim = streams[0]["time_series"]
    stim_ts = streams[0]["time_stamps"]

    stim_df = pd.DataFrame(stim)
    stim_df.columns = ['stim']
    stim_df['ts'] = stim_ts

    temp = pd.merge_asof(right=stim_df, left=eeg_df, on='ts', direction='nearest')

    df = pd.concat([df, temp], ignore_index=True)

df = df.drop(columns=['ts'])
data = df.to_numpy().T

#%%
sfreq = float(streams[1]["info"]["nominal_srate"][0])

info = mne.create_info(ch_names=['Fp1', 'Fp2', 'F4', 'F3', 'T7', 'T8', 'P4', 'P3', "stim"],
                        ch_types= ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "stim"],
                        sfreq=sfreq)

raw = mne.io.RawArray(data, info)

#%%
raw.ch_names

#%%
raw.load_data().filter(0.5, 10)
events = mne.find_events(raw, 'stim')

event_ids = {
        'forward'    : 1,   
        'backward'   : 2, 
        # 'left'       : 3, 
        # 'right'      : 4, 
        'turn right' : 5, 
        'turn left'  : 6, 
        # 'stand'      : 7, 
        # 'sit'        : 8,
        'nothing'    : 9, 
        'transition' : 10, 
    }

# %%
tmin, tmax = -2, 10  # define epochs around events (in s)
epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5, baseline=None, preload=True)

# %% #Access to the data
data = epochs._data

n_events = len(data) # or len(epochs.events)
print("Number of events: " + str(n_events)) 

n_channels = len(data[0,:]) # or len(epochs.ch_names)
print("Number of channels: " + str(n_channels))

n_times = len(data[0,0,:]) # or len(epochs.times)
print("Number of time instances: " + str(n_times))

# %%
plt.plot(data[14:20,0,:].T)
plt.title("Exemplar single-trial epoched data, for electrode 0")
plt.show()

# %%
evoked = epochs.average()

# %%
evoked_data = evoked.data
n_channels = len(evoked_data) # or len(evoked.ch_names)
print("Number of channels: " + str(n_channels))

n_times = len(evoked_data[0,:]) # or len(evoked.times)
print("Number of time instances: " + str(n_times))

# %%
plt.plot(evoked._data[0,:].T)
plt.title("Average epoched data, for electrode 0")
plt.show()

# %%
montage = mne.channels.make_standard_montage('biosemi16') 
print('Number of channels: ' + str(len(montage.ch_names)))

to_be_deleted = ['Fp1', 'Fp2', 'F4', 'F3', 'T7', 'T8', 'P4', 'P3']
to_be_deleted.reverse()

for name in to_be_deleted:
    print(name)
    id = montage.ch_names.index(name) 

    del montage.dig[id+3]
    del montage.ch_names[id]

plot = montage.plot(show_names=True)

# %%
epochs = mne.EpochsArray(data=data, info=info, events=events, event_id=event_ids)
epochs.set_montage(montage, on_missing="ignore")
epochs.drop_bad()

# %%
# Load necessary libraries
import mne
from mne.decoding import Vectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Models
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

#%%
epochs_UN = epochs['backward', 'forward', 'turn left', 'turn right'] # Unpleasant vs. Neutral

#%%
# Dataset with unpleasant and neutral events
data_UN = epochs_UN.get_data()
labels_UN = epochs_UN.events[:,-1]

# %%
train_data_UN, test_data_UN, labels_train_UN, labels_test_UN = train_test_split(data_UN, labels_UN, test_size=0.3, random_state=42)

# %%
clf_svm_0 = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(kernel='rbf', C=1))
scores = cross_val_score(clf_svm_0, data_UN, labels_UN, cv=3)

for i in range(len(scores)):   
    print('Accuracy of ' + str(i+1) + 'th fold is ' + str(scores[i]) + '\n')

# %% #svm
clf_svm_pip = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(random_state=42))
parameters = {'svc__kernel':['rbf', 'sigmoid'], 'svc__C':[0.1, 1, 10]}
gs_cv_svm = GridSearchCV(clf_svm_pip, parameters, scoring='accuracy', 
            cv=StratifiedKFold(n_splits=3), return_train_score=True)


gs_cv_svm.fit(train_data_UN, labels_train_UN)
print('Best Parameters: {}'.format(gs_cv_svm.best_params_))
print('Best Score: {}'.format(gs_cv_svm.best_score_))

#Prediction
predictions_svm = gs_cv_svm.predict(test_data_UN)

# #Evaluate
acc_svm = accuracy_score(labels_test_UN, predictions_svm)
print("Accuracy of SVM model: {}".format(acc_svm))

# precision_svm,recall_svm,fscore_svm,support_svm=precision_recall_fscore_support(labels_test_UN,predictions_svm,average='macro')
# print('Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_svm,recall_svm,fscore_svm))

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(gs_cv_svm, test_data_UN, labels_test_UN)  

# %%
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP

# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs_UN.get_data()
epochs_data_train = epochs_UN.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

labels = epochs_UN.events[:,-1]

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)


# %%
import neurokit2 as nk

# eeg = nk.mne_data("filt-0-40_raw")

eeg = nk.eeg_rereference(eeg, 'average')
eeg = eeg.get_data()[:, 0:500]  # Get the 500 first data points

# Compare L1 and L2 norms
l1 = nk.eeg_gfp(eeg, method="l1", normalize=True)
l2 = nk.eeg_gfp(eeg, method="l2", normalize=True)
nk.signal_plot([l1, l2])

# Mean-based vs. Median-based
gfp = nk.eeg_gfp(eeg, normalize=True)
gfp_r = nk.eeg_gfp(eeg, normalize=True, robust=True)
nk.signal_plot([gfp, gfp_r])

# Standardize the data
gfp = nk.eeg_gfp(eeg, normalize=True)
gfp_z = nk.eeg_gfp(eeg, normalize=True, standardize_eeg=True)
nk.signal_plot([gfp, gfp_z])