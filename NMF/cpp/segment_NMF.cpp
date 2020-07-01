#include <iostream>
#include <fstream>
#include <random>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#define BLOCKSIZE_DEFAULT 8192
#define HOPLEN_DEFAULT (BLOCKSIZE_DEFAULT / 4)
#define N_DEFAULT 4

struct WAVHeader {
    char ChunkID [4];
    uint32_t ChunkSize;
    char Format [4];
    char Subchunk1ID [4];
    uint32_t Subchunk1Size;
    uint16_t AudioFormat;
    uint16_t NumChannels;
    uint32_t SampleRate;
    uint32_t ByteRate;
    uint16_t BlockAlign;
    uint16_t BitsPerSample;
    char Subchunk2ID [4];
    uint32_t Subchunk2Size;
};

class WAVFile {
    WAVHeader _header;
    void *_data;
    bool _loaded;
    std::string _filename;
public:
    WAVFile (std::string filename) {
        _loaded = false;
        _filename = filename;
        std::ifstream file (filename, std::ifstream::binary);
        if (file.fail()) return;
        file.read ((char *) &_header, sizeof(WAVHeader));
        if (file.fail() || _header.BitsPerSample > 32) return;
        _data = (void *) new char[_header.Subchunk2Size];
        file.read((char *) _data, _header.Subchunk2Size);
        if (file.fail()) return;
        _loaded = true;
    }
    WAVFile (std::string filename, WAVHeader header, int size) {
        _filename = filename;
        _header = header;
        _data = (void *) new char[size + 1]{0};
        _loaded = true;
    }

    WAVHeader getHeader (void) { return _header; }
    double getSample(int channel, int index);
    void modSample(int channel, int index, double sample);
    void write (void);

    //friend int istft_WAV (WAVFile *, gsl_matrix *, int, int);
};

double WAVFile::getSample(int channel, int n) {
    if (!_loaded || n > _header.Subchunk2Size / _header.BlockAlign || channel > _header.NumChannels)
        return 0;
    char *datap = (char *) _data;
    datap += n * _header.BlockAlign + channel * _header.BitsPerSample / 8;
    int sample = 0;
    char *samplep = (char *) &sample;
    for (int i = _header.BitsPerSample / 8; i > 0; i--, datap++, samplep++)
        *samplep = *datap;
    if (sample & (1 << _header.BitsPerSample - 1))
        sample = (~sample & (1 << _header.BitsPerSample) - 1) * -1 - 1;
    return (double) sample / (double) (1 << _header.BitsPerSample - 1);
}

void WAVFile::modSample(int channel, int n, double sample) {
    if (!_loaded || n > _header.Subchunk2Size / _header.BlockAlign || channel > _header.NumChannels)
        return;
    char *datap = (char *) _data;
    datap += n * _header.BlockAlign + channel * _header.BitsPerSample / 8;
    int sampleint = sample * (1 << _header.BitsPerSample - 1);

    /* Clip input */
    if (sampleint & (1 << 31) && sampleint >= 1 << _header.BitsPerSample)
        sampleint = 1 << _header.BitsPerSample - 1;
    else if (sampleint >= 1 << _header.BitsPerSample - 1)
        sampleint = (1 << _header.BitsPerSample - 1) - 1;

    char *samplep = (char *) &sampleint;
    for (int i = _header.BitsPerSample / 8; i > 0; i--, datap++, samplep++)
        *datap = *samplep;
}

void WAVFile::write (void) {
    if (!_loaded) return;
    std::ofstream file (_filename, std::ofstream::out);
    file.write ((char *) &_header, sizeof(WAVHeader));
    file.write ((char *) _data, _header.Subchunk2Size);
}

gsl_matrix *stft_WAV (WAVFile *file, int blocksize, int hoplen) {
    WAVHeader header = file->getHeader();
    unsigned int numSamples = header.Subchunk2Size / header.BlockAlign;
    unsigned int numBlocks = (numSamples - blocksize) / hoplen + ((numSamples - blocksize) % hoplen ? 1 : 0);
    gsl_matrix *matrix = gsl_matrix_alloc(numBlocks, blocksize);
    double *p = matrix->data;
    for (unsigned int jj = 0; jj < numBlocks; jj++, p += matrix->tda) {
        for (unsigned int j = 0; j < GSL_MIN(numSamples - jj * hoplen, blocksize); j++) {
            double sample = 0.0;
            for (int i = 0; i < header.NumChannels; i++)
                sample += file->getSample(i, jj * hoplen + j);
            p[j] = sample / header.NumChannels; // Convert to mono
            if ((jj != 0 || j >= blocksize / 2) && (jj != numBlocks - 1 || j < blocksize / 2))
                p[j] *= 0.54f - 0.46f * cos(2.f * M_PI * j / blocksize); // Hamming window function
        }
        if (gsl_fft_real_radix2_transform(p, 1, matrix->size2) != GSL_SUCCESS)
            return nullptr;
    }
    return matrix;
}

int istft_WAV (WAVFile *file, gsl_matrix *matrix, int blocksize, int hoplen) {
    WAVHeader header = file->getHeader();
    unsigned int numSamples = header.Subchunk2Size / header.BlockAlign;
    double *output = new double[numSamples]{0.0f};
    /* Apply IFFT */
    do {
        double *p = matrix->data;
        for (unsigned int i = 0; i < matrix->size1; i++, p += matrix->tda)
            if (gsl_fft_halfcomplex_radix2_inverse(p, 1, matrix->size2) != GSL_SUCCESS)
                return 1;
    } while (0);
    /* Add windows */
    do {
        unsigned int i;
        double *p = matrix->data;
        for (i = 0; i + hoplen < numSamples; i += hoplen, p += matrix->tda) {
            for (unsigned int j = 0; j < matrix->size2; j++) {
                    /* normalize first block, TODO: figure out how to normalize last block too.
                    if (i > 0 || j >= matrix->size2 / 2)
                    */
                    p[j] *= 0.462962963f;
                    //p[j] *= 0.8f;
                    output[i + j] += p[j];
            }
        }
        if (i < numSamples) for (unsigned int j = 0; i < numSamples; i++, j += 2)
            output[i] += p[j];
    } while (0);

    for (int i = 0; i < numSamples; i++)
        file->modSample(0, i, output[i]);
    return 0;
}

int main (int argc, char *argv[]) {
    std::string infilename = "boat_docking.wav";
    int BLOCKSIZE = BLOCKSIZE_DEFAULT, HOPLEN = HOPLEN_DEFAULT, N = N_DEFAULT;
    switch (argc) {
        case 5:
            HOPLEN = atoi(argv[4]);
        case 4:
            BLOCKSIZE = atoi(argv[3]);
        case 3:
            N = atoi(argv[2]);
        case 2:
            infilename = std::string(argv[1]);
        default:
            break;
    }
    std::cout << "Reading " << infilename << "\n";
    WAVFile *in = new WAVFile("../../soundfiles/" + infilename);
    std::cout << "finding " << N << " features, blocksize: " << BLOCKSIZE << ", hoplen: " << HOPLEN << "\n";
    WAVHeader inHeader = in->getHeader();
    int numSamples = inHeader.Subchunk2Size / inHeader.BlockAlign;
    /* Sanity Check
    int numSamples_pwr2 = 1;
    for (; numSamples_pwr2 < numSamples; numSamples_pwr2 *= 2)
        ;
    double *input = new double[numSamples_pwr2];
    for (int i = 0; i < numSamples_pwr2; i++)
        input[i] = in->getSample(0, i);
    if (gsl_fft_real_radix2_transform(input, 1, numSamples_pwr2) != GSL_SUCCESS) return 1;
    if (gsl_fft_halfcomplex_radix2_inverse(input, 1, numSamples_pwr2) != GSL_SUCCESS) return 1;
    for (int i = 0; i < numSamples; i++)
        out->modSample(0, i, input[i]);
    */
    gsl_matrix *D = stft_WAV(in, BLOCKSIZE, HOPLEN);
    if (D == nullptr)
        std::cout << "Error: Failed to analyze soundfile." << std::endl;
    std::cout << "STFT of size: (" << D->size1 << ", " << D->size2 << ")\n";

    /* Compute S (spectrogram) from D = STFT(y) */
    gsl_matrix *S = gsl_matrix_alloc(D->size1, D->size2 / 2 + 1);
    double *Sp = S->data;
    double *Dp = D->data;
    do {
        gsl_complex_packed_array gp = new double[D->size2 * 2];
        for (int i = 0; i < S->size1; i++, Sp += S->tda, Dp += D->tda) {
            gsl_fft_halfcomplex_radix2_unpack(Dp, gp, 1, D->size2);
            for (int j = 0; j < S->size2; j++)
                gsl_matrix_set(S, i, j, gsl_complex_abs(*(gsl_complex *) (gp + 2 * j)));
        }
    } while (0);


    std::cout << "Computing NMF of spectrogram...\n";
    /* Compute NMF of S */
    gsl_matrix *W = gsl_matrix_alloc(S->size1, N);
    gsl_matrix *H = gsl_matrix_alloc(N, S->size2);
    do {
        std::default_random_engine gen (10);
        std::uniform_real_distribution<double> dist(0.0,1.0);
        for (int m = 0; m < S->size1; m++)
            for (int k = 0; k < N; k++)
                gsl_matrix_set(W, m, k, dist(gen));
        for (int n = 0; n < S->size2; n++)
            for (int k = 0; k < N; k++)
                gsl_matrix_set(H, k, n, dist(gen));
                    
    } while (0);
    gsl_matrix *W_1 = gsl_matrix_alloc(S->size1, N);
    gsl_matrix *H_1 = gsl_matrix_alloc(N, S->size2);
    gsl_matrix *W_2 = gsl_matrix_alloc(S->size1, N);
    gsl_matrix *H_2 = gsl_matrix_alloc(N, S->size2);
    gsl_matrix *sq = gsl_matrix_alloc(N, N);
    for (int i = 0; i < 20; i++) {
        /* Update H */
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W, S, 0.0, H_1); // H_1 = W^T x S
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, W, W, 0.0, sq);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sq, H, 0.0, H_2); // H_2 = W^T x W x H
        gsl_matrix_div_elements(H_1, H_2);
        gsl_matrix_mul_elements(H, H_1); // H_new = H * H_1 / H_2

        /* Update W */
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, S, H, 0.0, W_1); // W_1 = S x H^T
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, H, H, 0.0, sq);
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, W, sq, 0.0, W_2); // W_2 = W x H x H^T
        gsl_matrix_div_elements(W_1, W_2);
        gsl_matrix_mul_elements(W, W_1); // W_new = W * W_1 / W_2
    }
    gsl_matrix_free(W_1);
    gsl_matrix_free(H_1);
    gsl_matrix_free(W_2);
    gsl_matrix_free(H_2);
    gsl_matrix_free(sq);

    WAVHeader outHeader = inHeader;
    outHeader.NumChannels = 1;
    outHeader.BlockAlign = inHeader.BlockAlign / inHeader.NumChannels;
    outHeader.Subchunk2Size = inHeader.Subchunk2Size / inHeader.NumChannels;

    std::cout << "Outputting features of NMF as soundfiles...\n";
    for (int i = 0; i <= N; i++) {
        std::string outfilename;
        /* Compute features */
        if (i == N) {
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, W, H, 0.0, S); // S = W * H
            outfilename = "out.wav";
            std::cout << "NMF reconstruction of " << infilename << " output as " << outfilename << "\n";
            outfilename = "output/" + outfilename;
        } else {
            gsl_matrix W_i = gsl_matrix_submatrix(W, 0, i, W->size1, 1).matrix;
            gsl_matrix H_i = gsl_matrix_submatrix(H, i, 0, 1, H->size2).matrix;
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &W_i, &H_i, 0.0, S); // S = W * H
            outfilename = "f" + std::to_string(i + 1) + ".wav";
            std::cout << "feature " << i + 1 << " output as " << outfilename << "\n";
            outfilename = "output/" + outfilename;
        }

        WAVFile *out = new WAVFile (outfilename, outHeader, outHeader.Subchunk2Size);

        /* Resynthesize output from spectrogram */
        std::default_random_engine gen (10);
        std::uniform_real_distribution<double> dist(-1.0,1.0);
        for (int i = 0; i < numSamples; i++)
            out->modSample(0, i, dist(gen));
        gsl_complex_packed_array gp = new double[D->size2 * 2];
        for (int n = 0; n < 2; n++) {
            D = stft_WAV(out, BLOCKSIZE, HOPLEN);
            Dp = D->data;
            Sp = S->data;
            for (int i = 0; i < D->size1; i++, Sp += S->tda, Dp += D->tda) {
                gsl_fft_halfcomplex_radix2_unpack(Dp, gp, 1, D->size2);
                for (int j = 0; j < S->size2; j++) {
                    gsl_complex eiaD;
                    if (j == S->size1 - 1)
                        eiaD = gsl_complex_rect(1.f, 0.f);
                    else {
                        double angle = gsl_complex_arg(*(gsl_complex *) (gp + 2 * j));
                        eiaD = gsl_complex_exp(gsl_complex_rect(0.f, 1.f * angle));
                    }
                    Dp[j] = Sp[j] * GSL_REAL(eiaD);
                    if (j > 0 && j < S->size2 - 1) Dp[D->size2 - j] = Sp[j] * GSL_IMAG(eiaD);
                }
            }
            istft_WAV(out, D, BLOCKSIZE, HOPLEN);
        }
        out->write();
    }
    gsl_matrix_free(W);
    gsl_matrix_free(H);
    std::cout << "Done!" << std::endl;

    //gsl_matrix_free(D);
    //gsl_matrix_free(S);
    
    return 0;
}
