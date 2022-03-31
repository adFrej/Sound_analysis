import base64
import io
import numpy as np
from scipy.io import wavfile
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import webbrowser
import math


def read_file(file_path):
    sampling_rate, samples = wavfile.read(file_path)
    return sampling_rate, samples


def draw_audio(samples):
    fig = px.line(samples)
    return fig


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


def debug:
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


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Dźwięk - projekt 1',
            style={'color': 'blue'}),

    dcc.Upload(
        id='upload-file',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '90%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'margin-top': '10px',
            'margin-bottom': '10px',
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    dcc.Graph(
        id='time-graph',),
])

time_graph = {}


@app.callback(
    Output('time-graph', 'figure'),
    Input('upload-file', 'contents'),
    State('upload-file', 'filename'),
    State('upload-file', 'last_modified'),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    global time_graph
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(',')
        file = base64.b64decode(content_string)
        file = io.BytesIO(file)
        rate, samples = read_file(file)
        time_graph = draw_audio(samples)
    return time_graph


port = 8050


def open_browser():
    webbrowser.open_new("http://localhost:{}".format(port))


app.title = 'dzwiek'

if __name__ == '__main__':
    # rate, samples = read_file('../wyewoluowac.wav')
    # print(volume(samples))
    # print(zcr(rate, samples))
    # print(sr(rate, samples))
    # draw_audio(samples)

    open_browser()
    app.run_server(debug=True)
