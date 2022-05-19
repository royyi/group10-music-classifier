import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew, kurtosis


from eda import util

import librosa

AUDIO_DIR_PATH = '../data/fma_small/'

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

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()



def load_genre(genre, subset, tracks):
    filtered = tracks.loc[(tracks.track.genre_top == genre) & (tracks.set.subset == subset)]

    return filtered.index.to_numpy()


def compute_features(track_id):
    try:
        warnings.filterwarnings("ignore", '.*PySoundFile.*',)

        path = util.get_audio_path(audio_dir=AUDIO_DIR_PATH, track_id=track_id)
        x, sr = librosa.load(path, sr=None, mono=True)



        features = pd.Series(index=columns(), dtype=np.float32, name=track_id)

        def feature_stats(name, values):
            features[name, 'mean'] = np.mean(values, axis=1)
            features[name, 'std'] = np.std(values, axis=1)
            features[name, 'skew'] = skew(values, axis=1)
            features[name, 'kurtosis'] = kurtosis(values, axis=1)
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

        return features

    except:
        print('Error encountered with track {}'.format(track_id))

def main() -> int:
    tracks = util.load('data/fma_metadata/tracks.csv')

    small_rock = load_genre('Rock', 'small', tracks=tracks)
    small_electronic = load_genre('Electronic', 'small', tracks=tracks)


    features_rock = pd.DataFrame(columns=columns())

    for i, track_id in enumerate(small_rock):
        if i % 10 == 0:
            print(i)
        features_rock.loc[track_id] = compute_features(track_id)
    features_rock.to_csv('small_rock_features.csv')

    features_elec = pd.DataFrame(columns=columns())

    for i, track_id in enumerate(small_electronic):
        if i % 10 == 0:
            print(i)
        features_elec.loc[track_id] = compute_features(track_id)
    features_elec.to_csv('small_elec_features.csv')


    return 0


if __name__ == '__main__':
    sys.exit(main())