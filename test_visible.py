import unittest, h5py, submitted, os
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
    with h5py.File('data.hdf5','r') as f:
        for y in f['train'].keys():
            train_waveforms[y] = [ f['train'][y][i][:] for i in sorted(f['train'][y].keys()) ]
            dev_waveforms[y] = [ f['dev'][y][i][:] for i in sorted(f['dev'][y].keys()) ]
    train_cepstra, train_spectra = compute_features(train_waveforms)
    dev_cepstra, dev_spectra = compute_features(dev_waveforms)
    return train_cepstra, dev_cepstra
    
# TestSequence
class TestStep(unittest.TestCase):
    @weight(6.25)
    def test_initialize_hmm(self):
        train_cepstra, dev_cepstra = get_data()
        with h5py.File('solutions.hdf5','r') as h5:
            nstates = 5
            Lambda = { y:submitted.initialize_hmm(train_cepstra[y], 5) for y in train_cepstra.keys() }
            for k,y in enumerate(train_cepstra.keys()):
                A = Lambda[y][0]
                ref = h5['Lambda'][y]['0']
                for i in range(nstates):
                    err = np.average(np.abs(A[i,:]-ref[i,:]))/np.average(np.abs(ref[i,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** A[%s][%d,:] is off by more than 4%%'%(y,i)
                    )
                Mu = Lambda[y][1]
                ref = h5['Lambda'][y]['1']
                for i in range(nstates):
                    err = np.average(np.abs(Mu[i,:]-ref[i,:]))/np.average(np.abs(ref[i,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Mu[%s][%d,:] is off by more than 4%%'%(y,i)
                    )
                Sigma = Lambda[y][2]
                ref = h5['Lambda'][y]['2']
                for i in range(nstates):
                    err = np.average(np.abs(np.diag(Sigma[i,:,:]-ref[i,:,:])))/np.average(np.abs(ref[i,:,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Diagonals of Sigma[%s][%d,:,:] off by more than 4%%'%(y,i)
                    )


    @weight(6.25)
    def test_observation_pdfs(self):
        train_cepstra, dev_cepstra = get_data()
        with h5py.File('solutions.hdf5','r') as h5:
            for k,y in enumerate(train_cepstra.keys()):
                B = submitted.observation_pdf(train_cepstra[y][0], h5['Lambda'][y]['1'][:],
                                              h5['Lambda'][y]['2'][:])
                ref = h5['B_dict'][y]['0']
                (nframes,nstates) = B.shape
                for i in range(nstates):
                    err = np.average(np.abs(B[:,i]-ref[:,i]))/np.average(np.abs(ref[:,i]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** B[%s][0][:,%d] is off by more than 4%%'%(y,i)
                    )

    @weight(6.25)
    def test_scaled_forward(self):
        with h5py.File('solutions.hdf5','r') as h5:
            for k,y in enumerate(h5['B_dict'].keys()):
                Alpha_Hat, G = submitted.scaled_forward(h5['Lambda'][y]['0'][:], h5['B_dict'][y]['0'][:])
                ref = h5['Alpha_dict'][y]['0']
                (nframes,nstates) = Alpha_Hat.shape
                for i in range(nstates):
                    err = np.average(np.abs(Alpha_Hat[:,i]-ref[:,i]))/np.average(np.abs(ref[:,i]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Alpha_Hat[%s][0][:,%d] is off by more than 4%%'%(y,i)
                    )
                ref = h5['G_dict'][y]['0']
                err = np.average(np.abs(G[:]-ref[:]))/np.average(np.abs(ref[:]))
                self.assertLess(
                    err, 0.04, 
                    msg='\*** G[%s][0][:] is off by more than 4%%'%(y)
                )

    @weight(6.25)
    def test_scaled_backward(self):
        with h5py.File('solutions.hdf5','r') as h5:
            for k,y in enumerate(h5['B_dict'].keys()):
                Beta_Hat = submitted.scaled_backward(h5['Lambda'][y]['0'][:], h5['B_dict'][y]['0'][:])
                ref = h5['Beta_dict'][y]['0']
                (nframes,nstates) = Beta_Hat.shape
                for i in range(nstates):
                    err = np.average(np.abs(Beta_Hat[:,i]-ref[:,i]))/np.average(np.abs(ref[:,i]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Beta_Hat[%s][0][:,%d] is off by more than 4%%'%(y,i)
                    )
                    
    @weight(6.25)
    def test_posteriors(self):
        with h5py.File('solutions.hdf5','r') as h5:
            for k,y in enumerate(h5['Alpha_dict'].keys()):
                Gamma, Xi = submitted.posteriors(h5['Lambda'][y]['0'][:], h5['B_dict'][y]['0'][:],
                                                 h5['Alpha_dict'][y]['0'][:], h5['Beta_dict'][y]['0'][:])
                ref = h5['Gamma_dict'][y]['0'][:]
                nstates = ref.shape[1]
                for i in range(nstates):
                    err = np.average(np.abs(Gamma[:,i]-ref[:,i]))/np.average(np.abs(ref[:,i]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Gamma[%s][0][:,%d] is off by more than 4%%'%(y,i)
                    )
                ref = h5['Xi_dict'][y]['0']
                for i in range(nstates):
                    err = np.average(np.abs(Xi[:,i,:]-ref[:,i,:]))/np.average(np.abs(ref[:,i,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Xi[%s][0][:,%d,:] is off by more than 4%%'%(y,i)
                    )

    @weight(6.25)
    def test_E_step(self):
        train_cepstra, dev_cepstra = get_data()
        with h5py.File('solutions.hdf5','r') as h5:
            for k,y in enumerate(h5['Gamma_dict'].keys()):
                X = np.concatenate([ train_cepstra[y][n][:] for n in range(len(train_cepstra[y])) ])
                Gamma = np.concatenate([ h5['Gamma_dict'][y][str(n)][:] for n in range(len(train_cepstra[y])) ])
                Xi = np.concatenate([ h5['Xi_dict'][y][str(n)][:] for n in range(len(train_cepstra[y])) ])
                expectations=submitted.E_step(X, Gamma, Xi)
                names=['A_num','A_den','Mu_num','Mu_den','Sigma_num','Sigma_den']
                for k in range(len(names)):
                    ref = h5['expectations'][y][str(k)]
                    err = np.average(np.abs(expectations[k]-ref[:]))/np.average(np.abs(ref[:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** %s[%s] is off by more than 4%%'%(names[k],y)
                    )

    @weight(6.25)
    def test_M_step(self):
        with h5py.File('solutions.hdf5','r') as h5:
            for k,y in enumerate(h5['expectations'].keys()):
                expectations = (h5['expectations'][y]['0'][:],h5['expectations'][y]['1'][:],h5['expectations'][y]['2'][:],h5['expectations'][y]['3'][:],h5['expectations'][y]['4'][:],h5['expectations'][y]['5'][:])
                Lambda_new = submitted.M_step(*expectations, 1)
                A = Lambda_new[0]
                ref = h5['Lambda_new'][y]['0']
                nstates = A.shape[0]
                for i in range(nstates):
                    err = np.average(np.abs(A[i,:]-ref[i,:]))/np.average(np.abs(ref[i,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** A_new[%s][%d,:] is off by more than 4%%'%(y,i)
                    )
                Mu = Lambda_new[1]
                ref = h5['Lambda_new'][y]['1']
                for i in range(nstates):
                    err = np.average(np.abs(Mu[i,:]-ref[i,:]))/np.average(np.abs(ref[i,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Mu[%s][%d,:] is off by more than 4%%'%(y,i)
                    )
                Sigma = Lambda_new[2]
                ref = h5['Lambda_new'][y]['2']
                for i in range(nstates):
                    err = np.average(np.abs(np.diag(Sigma[i,:,:]-ref[i,:,:])))/np.average(np.abs(ref[i,:,:]))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** Diagonals of Sigma_new[%s][%d,:,:] off by more than 4%%'%(y,i)
                    )

    @weight(6.25)
    def test_recognize(self):
        train_cepstra, dev_cepstra = get_data()
        with h5py.File('solutions.hdf5','r') as h5:
            Lambda_ref = {}
            for y in h5['Lambda_new'].keys():
                Lambda_ref[y] = ( h5['Lambda_new'][y]['0'][:], h5['Lambda_new'][y]['1'][:], h5['Lambda_new'][y]['2'][:])
            for k,y in enumerate(train_cepstra.keys()):
                logprob, Y_hat = submitted.recognize(dev_cepstra[y], Lambda_ref)
                for yhat in train_cepstra.keys():
                    ref = h5['logprob_dict'][y][yhat]
                    err = np.average(np.abs(logprob[yhat]-ref))/np.average(np.abs(ref))
                    self.assertLess(
                        err, 0.04, 
                        msg='\*** logprob[%s][%s] is off by more than 4%%'%(y,yhat)
                    )
