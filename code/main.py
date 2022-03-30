import numpy as np
from scipy.io import wavfile
import plotly.express as px

def read_file(file_path):
    sampling_rate, samples = wavfile.read(file_path)
    return sampling_rate, samples

def draw_audio(samples):
    fig = px.line(samples)
    fig.show()

def volume(samples):
    return np.sqrt(np.sum(samples*samples)/len(samples))

def ste(samples):
    return np.sum(samples*samples)/len(samples)

def zcr(sampling_rate, samples):
    return np.sum(np.sign(samples[1:]) - np.sign(samples[:-1])) * sampling_rate / len(samples) / 2

def sr(sampling_rate, samples):
    return volume(samples) < 0.02 and zcr(sampling_rate, samples) < 50

if __name__ == '__main__':
    rate, samples = read_file('../wyewoluowac.wav')
    print(volume(samples))
    print(zcr(rate, samples))
    print(sr(rate, samples))
    draw_audio(samples)