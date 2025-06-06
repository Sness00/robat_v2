import numpy as np
from scipy.signal import stft
from matplotlib import pyplot as plt

def music(y, fs, nch, d, bw, theta=np.linspace(-90, 90, 73), c=343, wlen=64, ns=2, show=False):    
  """
  Simple multiband MUltiple SIgnal Classification spatial filter implementation.

  Parameters:
  
    y: mic array signals

    fs: sampling rate

    nch: number of mics in the array

    ns: number of sources

    d: mic spacing

    bw: (low freq, high freq)

    theta: angle vector

    c: sound speed

    wlen: window length for stft

    ns: expected number of sources

    show: plot the pseudospectrum for each band

  Returns: 
    
    theta: angle axis
    
    mag_p: magnitude of average spatial energy distribution estimation across bands
  """
  f_spec_axis, _, spectrum = stft(y, fs=fs, window=np.ones((wlen, )), nperseg=wlen, noverlap=wlen-1, axis=0)
  bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
  
  p = np.zeros_like(theta, dtype=complex)
  p_i = np.zeros((len(theta), 1), dtype=complex)
  
  if show:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

  for f_c in bands:
    w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
    a = np.exp(np.outer(np.linspace(0, nch-1, nch), -1j*w_s))
    a_H = a.T.conj()     
    spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
    cov_est = np.cov(spec, bias=True)
    lambdas, V = np.linalg.eig(cov_est)
    indices = np.argsort(lambdas)[::-1]
    V_sorted = V[indices]
    V_n = V_sorted[:, ns:]
    V_n_H = V_n.T.conj()

    for i in range(len(theta)):
      p_i[i] = 1/(a_H[i, :] @ V_n @ V_n_H @ a[:, i]) 
      p[i] += p_i[i]

    if show:
      ax.plot(np.deg2rad(theta), 20*np.log10(np.abs(p_i)), label=f'{f_c} Hz')
  
  if show:
    ax.set_xlim((-np.pi/2, np.pi/2))
    ax.set_title('Pseudospectra')
    ax.set_theta_offset(np.pi/2)
    plt.legend()
    plt.show()
  mag_p = np.abs(p)**2/len(bands)
      
  return theta, mag_p