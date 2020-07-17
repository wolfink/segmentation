import numpy as np
import soundfile as sf
import librosa as lr
import cmath
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy import linalg
from scipy import sparse

BLOCKSIZE = int(256)
HOPLEN = int(1/4 * BLOCKSIZE)
#OVERLAP = BLOCKSIZE - HOPLEN
FILENAME = 'boat_docking.wav'
a = 6 # X radius
b = 3 # Y radius
sigma_f = 1
sigma_b = 0.05

# Load file, compute spectrogram
print (BLOCKSIZE, 'samples per block,', HOPLEN, 'hop length,')
print('Loading', FILENAME + '...', end = '')
y, sr = lr.load("../soundfiles/" + FILENAME)
y = y[:HOPLEN * 29 + BLOCKSIZE]
print(len(y), 'samples')
print('Calculating STFT...')
D = lr.stft(y, n_fft=BLOCKSIZE, hop_length=HOPLEN)
fftf = lr.core.fft_frequencies(sr=sr, n_fft=BLOCKSIZE)
print(np.shape(D)[0], 'buckets,', np.shape(D)[1], 'blocks')
S, _ = lr.magphase(D)

#Construct sparse matrix
def ncuts(S):
	L = np.arange(len(S.flatten()))
	def createWeightDiag(offset):
		dlen = len(L) - offset
		X = L[:dlen] % np.shape(S)[1] - L[len(L) - dlen:] % np.shape(S)[1]
		Y = L[:dlen] // np.shape(S)[1] - L[len(L) - dlen:] // np.shape(S)[1]
		F = np.sqrt(X ** 2 + (a / b * Y) ** 2)
		B = np.abs(S.flatten()[:dlen] - S.flatten()[len(L) - dlen:])
		return np.exp(-B / sigma_b) * np.exp(-F / sigma_f)
	diags = []
	offsets = []
	for i in range(b):
		for j in range(a):
			index = np.shape(S)[1] * i + j
			if (a/b*i) ** 2 + j ** 2 < a ** 2:
				w = createWeightDiag(index)
				diags += [w]
				offsets += [index]
				if i > 0 or j > 0:
					diags += [w]
					offsets += [-index]
	W = sparse.diags (diags, offsets)
	D = sparse.diags ([(W @ np.ones(len(L)))], [0])
	vals, vecs = sparse.linalg.eigsh(D - W, k = 2, M = D, which='SM', maxiter=1000)
	print(vals)
	return vecs
vecs = ncuts(S)
mask1 = np.reshape((np.sign(vecs.T[1]) + 1) / 2, np.shape(S))
print(mask1)
mask2 = np.reshape((np.sign(vecs.T[1]) - 1) / -2, np.shape(S))
print(mask2)
vecs = ncuts(mask1 * S)
mask3 = np.reshape((np.sign(vecs.T[1]) + 1) / 2, np.shape(S))
print(mask3)
mask4 = np.reshape((np.sign(vecs.T[1]) - 1) / -2, np.shape(S))
print(mask4)

filename = 'output/f1.wav'
y_out = np.random.rand(len(y))
for _ in range(3):
    D_out = lr.stft(y_out, n_fft=BLOCKSIZE, hop_length=HOPLEN)
    D_out = S * mask3 * np.exp(complex(0.0, 1.0) * np.angle(D_out))
    y_out = lr.istft(D_out, hop_length=HOPLEN)
sf.write(filename, y_out, sr)

filename = 'output/f2.wav'
y_out = np.random.rand(len(y))
for _ in range(3):
    D_out = lr.stft(y_out, n_fft=BLOCKSIZE, hop_length=HOPLEN)
    D_out = S * mask4 * np.exp(complex(0.0, 1.0) * np.angle(D_out))
    y_out = lr.istft(D_out, hop_length=HOPLEN)
sf.write(filename, y_out, sr)

filename = 'output/f3.wav'
y_out = np.random.rand(len(y))
for _ in range(3):
    D_out = lr.stft(y_out, n_fft=BLOCKSIZE, hop_length=HOPLEN)
    D_out = S * mask2 * np.exp(complex(0.0, 1.0) * np.angle(D_out))
    y_out = lr.istft(D_out, hop_length=HOPLEN)
print(y_out)
sf.write(filename, y_out, sr)


"""
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
"""