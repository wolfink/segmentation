import numpy as np
import soundfile as sf
import librosa as lr
import cmath
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from skimage import data, segmentation, io, color
from skimage.future import graph

BLOCKSIZE = int(4096)
HOPLEN = int(1/4 * BLOCKSIZE)
#OVERLAP = BLOCKSIZE - HOPLEN
N = 50
FILENAME = 'sf1.wav'

# Load file, compute spectrogram
print (BLOCKSIZE, 'samples per block,', HOPLEN, 'hop length,', N, 'features')
print('Loading', FILENAME + '...', end = '')
y, sr = lr.load("../soundfiles/" + FILENAME)
print(len(y), 'samples')
print('Calculating STFT...')
D = lr.stft(y, n_fft=BLOCKSIZE, hop_length=HOPLEN)
fftf = lr.core.fft_frequencies(sr=sr, n_fft=BLOCKSIZE)
print(np.shape(D)[0], 'buckets,', np.shape(D)[1], 'blocks')
S, _ = lr.magphase(D)
S = ((S ** 2 / np.linalg.norm(S)) * 255).astype('int8')

img = S #data.coffee()
#io.imshow(img)
labels1 = segmentation.slic(img, compactness=5, n_segments=4000)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color (img, labels1, mode='similarity')
labels2 = graph.cut_normalized(labels1, g)
print(labels2)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
        a.axis('off')

plt.tight_layout()
plt.show()
