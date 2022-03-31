from random import sample
import numpy as np
from scipy.io import wavfile
import plotly.express as px
import math


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


def vdr(samples):
    v = volume(samples)
    return (np.max(v) - np.min(v)) / np.max(v)


def mean(samples):
    return np.mean(volume(samples))


def std(samples):
    return np.std(volume(samples))

# frame_length w ms


def lster(samples, rate, frame_length):

    sum = 0
    avSTE = []

    for i in range(math.ceil(len(samples)/rate)):
        avSTE.append(ste(samples[i*rate: min((i+1)*rate, len(samples))]))

    print(avSTE)
    for i in range(math.ceil(len(samples)*1000/(frame_length*rate))):
        av_index = math.floor(i*frame_length / 1000)
        0.5*avSTE[av_index]

        sample[i * frame_length: min((i+1)*frame_length, len(samples))]
        sum = sum + (0.5*avSTE[av_index] - sample[i *
                     frame_length: min((i+1)*frame_length, len(samples))]) > 0

    return sum/(2*i)


if __name__ == '__main__':
    rate, samples = read_file('./chrzaszcz.wav')
    print('dlugosc: ', len(samples)/rate, ' s')
    print('volume: ', volume(samples))
    print('zcr: ', zcr(rate, samples))
    print('sr: ', sr(rate, samples))

    print('vdr: ', vdr(samples))
    print('mean: ', mean(samples))
    print('std: ', std(samples))
    print('lster: ', lster(samples, rate, 100))
    # draw_audio(samples)
