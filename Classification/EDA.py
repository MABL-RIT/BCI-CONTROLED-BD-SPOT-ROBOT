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
eeg_channels = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'P08', 'Oz']
for f in files:
    streams, header = pyxdf.load_xdf(f)
    eeg = streams[1]["time_series"][:, 0:8]
    eeg_ts = streams[1]["time_stamps"]

    eeg_df = pd.DataFrame(eeg)
    eeg_df.columns = eeg_channels
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

info = mne.create_info(ch_names=['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'P08', 'Oz', 'stim'],
                        ch_types= ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "stim"],
                        sfreq=sfreq)

raw = mne.io.RawArray(data, info)

#%%
raw.ch_names

#%%
raw.load_data().filter(1., 50.)
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
        'nothing'    : 10, 
        'transition' : 9, 
    }

# %%
tmin, tmax = -0, 10  # define epochs around events (in s)
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

#%%
from mne.decoding import UnsupervisedSpatialFilter

from sklearn.decomposition import PCA, FastICA

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

epochs = mne.Epochs(raw, events, event_ids, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True,
                    verbose=False)

epochs = epochs['backward', 'forward', 'turn left', 'turn right', 'nothing'] 

X = epochs.get_data()

#%%
pca = UnsupervisedSpatialFilter(PCA(8), average=False)
pca_data = pca.fit_transform(X)
ev = mne.EvokedArray(np.mean(pca_data, axis=0),
                     mne.create_info(8, 250, ch_types='eeg'), tmin=tmin)

ev.plot(show=False, window_title="PCA", time_unit='s')

#%%
ica = UnsupervisedSpatialFilter(FastICA(8), average=False)
ica_data = ica.fit_transform(X)
ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
                      mne.create_info(8, epochs.info['sfreq'],
                                      ch_types='eeg'), tmin=tmin)
ev1.plot(show=False, window_title='ICA', time_unit='s')

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
montage = mne.channels.make_standard_montage('standard_1020') 
print('Number of channels: ' + str(len(montage.ch_names)))

# to_be_deleted = ['Fp1', 'Fp2', 'F4', 'F3', 'T7', 'T8', 'P4', 'P3']
# to_be_deleted.reverse()

# for name in to_be_deleted:
#     print(name)
#     id = montage.ch_names.index(name) 

#     del montage.dig[id+3]
#     del montage.ch_names[id]

plot = montage.plot(show_names=True)

# %%
epochs = mne.EpochsArray(data=data, info=info, events=events, event_id=event_ids)
epochs.set_montage(montage, on_missing="ignore")
epochs.drop_bad()

# %%
# Load necessary libraries
import mne
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Models
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

#%%
epochs
#%%
epochs_UN = epochs['backward', 'forward', 'turn left', 'turn right', 'nothing'] 

#%%
# Dataset with unpleasant and neutral events
data_UN = epochs_UN.get_data()
labels_UN = epochs_UN.events[:,-1]

# %%
train_data_UN, test_data_UN, labels_train_UN, labels_test_UN = train_test_split(data_UN, labels_UN, test_size=0.3, random_state=42)


# %% #svm
clf_svm_pip = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(random_state=42))
parameters = {'svc__kernel':['rbf', 'sigmoid'], 'svc__C':[0.1, 1, 10]}
gs_cv_svm = GridSearchCV(clf_svm_pip, parameters, scoring='accuracy', 
            cv=5, return_train_score=True)


gs_cv_svm.fit(train_data_UN, labels_train_UN)
print('Best Parameters: {}'.format(gs_cv_svm.best_params_))
print('Best Score: {}'.format(gs_cv_svm.best_score_))

#Prediction
predictions_svm = gs_cv_svm.predict(test_data_UN)

# #Evaluate
acc_svm = accuracy_score(labels_test_UN, predictions_svm)
print("Accuracy of SVM model: {}".format(acc_svm))

plot_confusion_matrix(gs_cv_svm, test_data_UN, labels_test_UN)  


#%%
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from sklearn.pipeline import Pipeline

sampling_rate = 250


nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

features = []
for epoch in epochs_UN:
    sample_feature = []

    d = epoch[:8, :].reshape(-1)
    # plt.plot(d)

    # psd = DataFilter.get_psd_welch(d, nfft, nfft // 2, sampling_rate,
    #                             WindowFunctions.BLACKMAN_HARRIS.value)

    # band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
    # band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)

    # features.append([band_power_alpha, band_power_beta])

    for i in range(8):
        ch_data = epoch[i, :]
        DataFilter.detrend(ch_data, DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(ch_data, nfft, nfft // 2, sampling_rate,
                                    WindowFunctions.BLACKMAN_HARRIS.value)

        band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
        band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)

        sample_feature.append(band_power_alpha)
        sample_feature.append(band_power_beta)

    features.append(sample_feature)

X = np.array(features)
y = labels_UN

#%%
train_data_UN, test_data_UN, labels_train_UN, labels_test_UN = train_test_split(X, y, test_size=0.3, random_state=42)


clf_svm_pip = Pipeline(steps=[('scaler', StandardScaler()), 
                             ('rf', RandomForestClassifier(random_state=42))])

parameters = {'rf__max_depth':[ 5, 6, 7 ], 'rf__n_estimators':[50, 60, 100], 'rf__max_features':[3, 4, 5]}

gs_cv_svm = GridSearchCV(clf_svm_pip, parameters, scoring='accuracy', 
            cv=5, return_train_score=True)


gs_cv_svm.fit(train_data_UN, labels_train_UN)
print('Best Parameters: {}'.format(gs_cv_svm.best_params_))
print('Best Score: {}'.format(gs_cv_svm.best_score_))

#Prediction
predictions_svm = gs_cv_svm.predict(test_data_UN)

# #Evaluate
acc_svm = accuracy_score(labels_test_UN, predictions_svm)
print("Accuracy of SVM model: {}".format(acc_svm))

plot_confusion_matrix(gs_cv_svm, test_data_UN, labels_test_UN)  

# %%
import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, create_info, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import AverageTFR

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
# %%

# Assemble the classifier using scikit-learn pipeline
clf = make_pipeline(CSP(n_components=4, reg=None, log=True, norm_trace=False),
                    LinearDiscriminantAnalysis())
                    
n_splits = 3  # for cross-validation, 5 is better, here we use 3 for speed
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Classification & time-frequency parameters
tmin, tmax = -.200, 2.000
n_cycles = 10.  # how many complete cycles: used to define window size
min_freq = 8.
max_freq = 20.
n_freqs = 6  # how many frequency bins to use

# Assemble list of frequency range tuples
freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples

# Infer window spacing from the max freq and number of cycles to avoid gaps
window_spacing = (n_cycles / np.max(freqs) / 2.)
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)

# Instantiate label encoder
le = LabelEncoder()
# %%
# init scores
freq_scores = np.zeros((n_freqs - 1,))

# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):

    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

    # Apply band-pass filter to isolate the specified frequencies
    raw_filter = raw.copy().filter(fmin, fmax, fir_design='firwin', skip_by_annotation='edge')

    # Extract epochs from filtered data, padded by window size
    epochs = Epochs(raw_filter, events, event_ids, tmin - w_size, tmax + w_size,
                    proj=False, baseline=None, preload=True)
                    
    epochs.drop_bad()
    y = le.fit_transform(epochs.events[:, 2])

    X = epochs.get_data()

    # Save mean scores over folds for each frequency and time window
    freq_scores[freq] = np.mean(cross_val_score(
        estimator=clf, X=X, y=y, scoring='roc_auc', cv=cv), axis=0)
# %%
plt.bar(freqs[:-1], freq_scores, width=np.diff(freqs)[0],
        align='edge', edgecolor='black')
plt.xticks(freqs)
plt.ylim([0, 1])
plt.axhline(len(epochs['forward']) / len(epochs), color='k', linestyle='--',
            label='chance level')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decoding Scores')
plt.title('Frequency Decoding Scores')
# %%
