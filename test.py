from utilities import padding2s, extract_RMS, extract_mfcc, extract_Wavelet, extract_stpsd, extract_autocorr
from utilities import extract_zero_crosings, extract_spectral_rolloff, extract_spectral_centroid
import librosa

audio_path = r"D:\PyCharm\underwater-data\shipsEar2s\classA\15__10_07_13_radaUno_Pasa_segment_8.wav"

y, sr = librosa.load(audio_path, sr=7000)
#print(y.shape)
y = padding2s(y, sr=7000, s=2)
#print(y.shape)


spectral_rolloff = extract_spectral_rolloff(y, sr, target_length=28) # 28
#print(spectral_rolloff.shape)
zero_crossings = extract_zero_crosings(y, target_length=28) # 28
#print(zero_crossings.shape)
spectral_centroid = extract_spectral_centroid(y, sr, target_length=28) # 28
#print(spectral_centroid.shape)
rms = extract_RMS(y, target_length=28) # 28
#print(rms.shape)
autocorr = extract_autocorr(y, sr, target_length=28) # 28
#print(autocorr.shape)
stpsd  = extract_stpsd(y, n_components=28) # 28
#print(stpsd.shape)
mfcc = extract_mfcc(y, sr, s=2) # 28没归一化
#print(mfcc.shape)
wavelet = extract_Wavelet(y, target_length=28) # 28没归一化
#print(wavelet.shape)



