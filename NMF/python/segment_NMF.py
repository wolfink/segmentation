import numpy as np
import soundfile as sf
import librosa as lr
import cmath
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

BLOCKSIZE = int(4096)
HOPLEN = int(1/4 * BLOCKSIZE)
#OVERLAP = BLOCKSIZE - HOPLEN
N = 50
FILENAME = 'boat_docking.wav'

# Load file, compute spectrogram
print (BLOCKSIZE, 'samples per block,', HOPLEN, 'hop length,', N, 'features')
print('Loading', FILENAME + '...', end = '')
y, sr = lr.load("../../soundfiles/" + FILENAME)
print(len(y), 'samples')
#y, sr = sf.read('coque.wav')
#y = np.asfortranarray(y.T[0])
print('Calculating STFT...')
D = lr.stft(y, n_fft=BLOCKSIZE, hop_length=HOPLEN)
fftf = lr.core.fft_frequencies(sr=sr, n_fft=BLOCKSIZE)
print(np.shape(D)[0], 'buckets,', np.shape(D)[1], 'blocks')
S, _ = lr.magphase(D)

# Calculate NMF decomposition
print('Calculating S = WH...', end = '')
model = NMF(n_components=N, init='random', random_state=0)
W = model.fit_transform(S)
H = model.components_
print('Done!')

# Output individual features & full NMF appx.
for n in range(N + 2):
    if n == N:
        S = W @ H
        filename = 'output/out.wav'
        print('outputting full NMF approximation as out.wav')
    elif n == N + 1:
        S, _ = lr.magphase(D)
        filename = 'output/orig.wav'
    else:
        S = np.array(np.matrix(W.T[n]).T * np.matrix(H[n]))
        plt.subplot(N + 1, 2, 2 * n + 1)
        plt.xscale('log')
        plt.plot(fftf, W.T[n])
        plt.subplot(N + 1, 2, 2 * n + 2)
        plt.plot(H[n])
        filename = 'output/f' + str(n + 1) + '.wav'
        print('outputting feature ' + str(n + 1) + ' as', filename[7:])
    # Recover phase data, algorithm idea from https://dsp.stackexchange.com/questions/9877/reconstruction-of-audio-signal-from-spectrogram
    y_out = np.random.rand(len(y))
    for _ in range(3):
        D_out = lr.stft(y_out, n_fft=BLOCKSIZE, hop_length=HOPLEN)
        D_out = S * np.exp(complex(0.0, 1.0) * np.angle(D_out))
        y_out = lr.istft(D_out, hop_length=HOPLEN)
    sf.write(filename, y_out, sr)
plt.show()
print ('Output located in \'output\' folder')
