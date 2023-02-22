import unittest, h5py, extra, os
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

def compute_features(waveforms, nceps=25):
    '''Compute two types of feature matrices, for every input waveform.

    Inputs:
    waveforms (dict of lists of (nsamps) arrays):
        waveforms[y][n] is the n'th waveform of class y
    nceps (scalar):
        Number of cepstra to retain, after windowing.

    Returns:
    cepstra (dict of lists of (nframes,nceps) arrays):
        cepstra[y][n][t,:] = windowed cepstrum of the t'th frame of the n'th waveform of the y'th class.
    spectra (dict of lists of (nframes,nceps) arrays):
        spectra[y][n][t,:] = liftered spectrum of the t'th frame of the n'th waveform of the y'th class. 

    Implementation Cautions:
        Computed with 200-sample frames with an 80-sample step.  This is reasonable if sample_rate=8000.
    '''
    cepstra = { y:[] for y in waveforms.keys() }
    spectra = { y:[] for y in waveforms.keys() }
    for y in waveforms.keys():
        for x in waveforms[y]:
            nframes = 1+int((len(x)-200)/80)
            frames = np.stack([ x[t*80:t*80+200] for t in range(nframes) ])
            spectrogram = np.log(np.maximum(0.1,np.absolute(np.fft.fft(frames)[:,1:100])))
            cepstra[y].append(np.fft.fft(spectrogram)[:,0:nceps])
            spectra[y].append(np.real(np.fft.ifft(cepstra[y][-1])))
            cepstra[y][-1] = np.real(cepstra[y][-1])
    return cepstra, spectra

def get_data():
    train_waveforms = {}
    dev_waveforms = {}
    test_waveforms = {}
    with h5py.File('data.hdf5','r') as f:
        for y in f['train'].keys():
            train_waveforms[y] = [ f['train'][y][i][:] for i in sorted(f['train'][y].keys()) ]
            dev_waveforms[y] = [ f['dev'][y][i][:] for i in sorted(f['dev'][y].keys()) ]
            test_waveforms[y] = [ f['test'][y][i][:] for i in sorted(f['test'][y].keys()) ]
    train_cepstra, train_spectra = compute_features(train_waveforms)
    dev_cepstra, dev_spectra = compute_features(dev_waveforms)
    test_cepstra, test_spectra = compute_features(test_waveforms)
    X_tmp = []
    Y_tmp = []
    for y in test_cepstra.keys():
        for n in range(len(test_cepstra[y])):
            X_tmp.append(test_cepstra[y][n])
            Y_tmp.append(y)
    seq = np.random.permutation(np.arange(len(X_tmp)))
    X_test = []
    Y_test = []
    for s in seq:
        X_test.append(X_tmp[s])
        Y_test.append(Y_tmp[s])
    return train_cepstra, dev_cepstra, X_test, Y_test
    
# TestSequence
class TestStep(unittest.TestCase):
    @weight(1)
    def test_extra_accuracy_above_40(self):
        train_cepstra, dev_cepstra, X_test, Y_test = get_data()
        hyps = extra.recognize(train_cepstra, dev_cepstra, X_test)
        accuracy = np.count_nonzero([y==yhat for (y,yhat) in zip(Y_test,hyps)])/len(Y_test)
        self.assertGreater(accuracy, 0.4, msg='Accuracy not greater than 0.4')

    @weight(1)
    def test_extra_accuracy_above_50(self):
        train_cepstra, dev_cepstra, X_test, Y_test = get_data()
        hyps = extra.recognize(train_cepstra, dev_cepstra, X_test)
        accuracy = np.count_nonzero([y==yhat for (y,yhat) in zip(Y_test,hyps)])/len(Y_test)
        self.assertGreater(accuracy, 0.5, msg='Accuracy not greater than 0.5')

    @weight(1)
    def test_extra_accuracy_above_60(self):
        train_cepstra, dev_cepstra, X_test, Y_test = get_data()
        hyps = extra.recognize(train_cepstra, dev_cepstra, X_test)
        accuracy = np.count_nonzero([y==yhat for (y,yhat) in zip(Y_test,hyps)])/len(Y_test)
        self.assertGreater(accuracy, 0.6, msg='Accuracy not greater than 0.6')

    @weight(1)
    def test_extra_accuracy_above_70(self):
        train_cepstra, dev_cepstra, X_test, Y_test = get_data()
        hyps = extra.recognize(train_cepstra, dev_cepstra, X_test)
        accuracy = np.count_nonzero([y==yhat for (y,yhat) in zip(Y_test,hyps)])/len(Y_test)
        self.assertGreater(accuracy, 0.7, msg='Accuracy not greater than 0.7')

    @weight(1)
    def test_extra_accuracy_above_80(self):
        train_cepstra, dev_cepstra, X_test, Y_test = get_data()
        hyps = extra.recognize(train_cepstra, dev_cepstra, X_test)
        accuracy = np.count_nonzero([y==yhat for (y,yhat) in zip(Y_test,hyps)])/len(Y_test)
        self.assertGreater(accuracy, 0.8, msg='Accuracy not greater than 0.8')

