import pandas as pd
import numpy as np

import librosa
import librosa.display
from scipy import stats

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


import sys

sys.path.append('../')

import utils

import pickle

def train_mlp():
    tracks = utils.load('../data/fma_metadata/tracks.csv')
    feats = utils.load('../data/fma_metadata/features.csv')

    small = tracks['set', 'subset'] <= 'small'

    X = feats.loc[small]
    y = tracks['track', 'genre_top'].loc[small]

    #uncomment for 5-class or 2-class
    y = y.loc[(y == 'Rock') | (y == 'Electronic') | (y == 'Instrumental') | (y == 'Hip-Hop') | (y == 'Pop')]
    #y = y.loc[(y == 'Rock') | (y == 'Electronic')]

    X = X.loc[X.index.isin(y.index)]

    #uncomment for mfcc
    #col = X.columns.get_loc('mfcc')
    #X = X.iloc[:, col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    sc = StandardScaler()
    X_train_copy = X_train.copy()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    mlp = MLPClassifier(solver = 'sgd', random_state = 42, activation = 'logistic', learning_rate_init = 0.2, batch_size = 100, hidden_layer_sizes = (200,100), max_iter = 100)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return mlp, X_train_copy

def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    return columns.sort_values()


def feature_extract(filename):
    x, sr = librosa.load(filename, sr=None, mono=True)
    
    columns()
    features = pd.Series(index=columns(), dtype=np.float32)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                    n_bins=7*12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
    del x

    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    feature_stats('chroma_stft', f)

    f = librosa.feature.rms(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    features = features.to_frame().T

    return features

def main():
    mlp, X_train_copy = train_mlp()

    # uncomment to create model pkl file
    #mlp_file = 'mlp_model.pkl'
    #pickle.dump(mlp, open(mlp_file, 'wb'))

    filename = "../feature extraction/files/1.mp3"
    #filename = "../feature extraction/files/2.mp3"
    #filename = "../feature extraction/files/3.mp3"
    #filename = "../feature extraction/files/4.mp3"
    #filename = "../feature extraction/files/5.mp3"
    #filename = "../feature extraction/files/6.mp3"

    features = feature_extract(filename)

    sc = StandardScaler()
    sc.fit(X_train_copy)
    
    # uncomment to create sc pkl file
    #sc_file = 'mlp_standard_scaler.pkl'
    #pickle.dump(sc, open(sc_file, 'wb'))

    features = sc.transform(features)

    prediction = mlp.predict(features)

    print(prediction)


main()