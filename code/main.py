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
import statistics
import csv


def read_file(file_path):
    sampling_rate, samples = wavfile.read(file_path)
    samples = samples.astype('int64')
    return sampling_rate, samples


def draw_audio(samples, name, markers=None):
    fig = px.line(samples)
    if markers is not None:
        for m in markers:
            fig.add_vline(x=m)
    fig.update_layout(title=name, yaxis_title="Amplitude",
                      xaxis_title="Time", showlegend=False, margin=dict(t=40))
    return fig


def draw_plot(values, name):
    fig = px.line(values)
    fig.update_layout(title=name, yaxis_title="Value",
                      xaxis_title="Time", showlegend=False, margin=dict(t=40))
    return fig


def volume(samples, rate=22000, frame_length=100, no_samples=False):

    if no_samples:
        return np.sqrt(np.sum(samples*samples)/len(samples))

    n_rfames = math.ceil(len(samples)*1000/rate / frame_length)

    vol = [0] * n_rfames

    l = len(samples)
    for i in range(n_rfames-1):
        scope = samples[i*(l//n_rfames): min((i+1) *
                                             (l//n_rfames), len(samples))]
        vol[i] = np.sqrt(np.sum(np.square(scope))/len(scope))

    scope = samples[n_rfames*(l//n_rfames):]
    vol[-1] = np.sqrt(np.sum(np.square(scope))/len(scope))

    return vol


def ste(samples, rate=22000, frame_length=100, no_samples=False):

    if no_samples:
        return np.sum(samples*samples)/len(samples)

    n_rfames = math.ceil(len(samples)*1000/rate / frame_length)
    ste = [0] * n_rfames

    l = len(samples)
    for i in range(n_rfames-1):

        scope = samples[i*(l//n_rfames): min((i+1) *
                                             (l//n_rfames), len(samples))]
        ste[i] = np.sum(np.square(scope))/len(scope)
    scope = samples[n_rfames*(l//n_rfames):]
    ste[-1] = np.sum(scope*scope)/len(scope)

    return ste


def zcr(samples, rate, frame_length=100, no_samples=False):

    if no_samples:
        return np.sum(np.sign(np.subtract(samples[1:]), np.sign(samples[:-1]))) * rate / len(samples) / 4

    n_rfames = math.ceil(len(samples)*1000/rate / frame_length)

    zcr = [0] * n_rfames

    l = len(samples)
    for i in range(n_rfames-1):

        scope = samples[i*(l//n_rfames): min((i+1) *
                                             (l//n_rfames), len(samples))]
        zcr[i] = np.sum(np.abs(np.subtract(np.sign(scope[1:]), np.sign(scope[:-1]))
                               )) * rate / len(scope) / 4

    scope = samples[n_rfames*(l//n_rfames):]
    zcr[-1] = np.sum(np.abs(np.subtract(np.sign(scope[1:]), np.sign(scope[:-1]))
                            )) * rate / len(scope) / 4

    return zcr


def sr(samples, rate, frame_length):

    n_rfames = math.ceil(len(samples)*1000/rate / frame_length)

    sr = [0] * n_rfames

    for i in range(n_rfames):
        scope = samples[i*frame_length: min((i+1)*frame_length, len(samples))]
        sr[i] = volume(scope, rate, frame_length, no_samples=True) < 0.02 and zcr(
            scope, rate, frame_length, no_samples=True) < 50

    return sr

#      VVVV Klip scope VVVV


def vdr(samples, rate, frame_length):
    v = volume(samples, rate, frame_length)
    return (max(v) - min(v)) / max(v)


def mean(samples, rate, frame_length):
    vol = volume(samples, rate, frame_length)
    return sum(vol)/len(vol)


def std(samples, rate, frame_length):
    return statistics.stdev(volume(samples, rate, frame_length))


def lster(samples, rate, frame_length):

    sum = 0
    avSTE = []

    for i in range(math.ceil(len(samples)/rate)):
        avSTE.append(
            ste(samples[i*rate: min((i+1)*rate, len(samples))], no_samples=True))

    for i in range(math.ceil(len(samples)*1000/(frame_length*rate))):
        av_index = math.floor(i*frame_length / 1000)

        sum = sum + (0.5*avSTE[av_index] - ste(samples[i *
                                                       frame_length: min((i+1)*frame_length, len(samples))], no_samples=True) > 0)

    return sum/(2*i)


def saveCSV(samples, rate, frame_length, path):
    header = ['frame_start', 'volume', 'ste', 'zcr', 'isSilent']
    start = np.arange(0, len(samples)/rate*1000, frame_length)
    vol = volume(samples, rate, frame_length)
    ste_ = ste(samples, rate, frame_length)
    zcr = volume(samples, rate, frame_length)
    silent = sr(samples, rate, frame_length)

    data = np.array([start, vol, ste_, zcr, silent]).T

    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
    return 0


def debug():
    rate, samples = read_file('./america.wav')
    samples = samples.astype('int64')

    print('dlugosc: ', len(samples)/rate, ' s')
    print('volume: ', volume(samples, rate, 100))
    print('zcr: ', zcr(samples, rate, 100))
    print('sr: ', sr(samples, rate, 100))
    print('ste: ', ste(samples, rate, 100))

    print('vdr: ', vdr(samples, rate, 100))
    print('mean: ', mean(samples, rate, 100))
    print('std: ', std(samples, rate, 100))
    print('lster: ', lster(samples, rate, 100))
    print('speach?: ', lster(samples, rate, 100) > 0.15)

    saveCSV(samples, rate, 100, './test.csv')
    # draw_audio(samples)


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Dźwięk - projekt 1'),

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

    html.Div(children=['Input frame length [ms]: ',
                       dcc.Input(
                           id='frame-size-in',
                           value='20',
                           style={'width': '50px'},),
                       ]),

    html.H3(id='frame-size-out'),

    dcc.Graph(
        id='time-graph',),

    dcc.Slider(
        min=0,
        max=0,
        step=1,
        value=0,
        id='frame-slider'
    ),

    dcc.Dropdown(['Volume', 'ZCR', 'STE'], id='param-dropdown'),

    dcc.Graph(
        id='param-graph',
    ),
])

sample_rate, samples = None, None


@ app.callback(
    Output('time-graph', 'figure'),
    Output('frame-size-out', 'children'),
    Output('frame-slider', 'max'),
    Input('upload-file', 'contents'),
    State('upload-file', 'filename'),
    State('upload-file', 'last_modified'),
    Input('frame-slider', 'value'),
    Input('frame-size-in', 'value'),
)
def draw_graph_from_file(list_of_contents, list_of_names, list_of_dates, frame_pos, frame_size):
    global sample_rate, samples
    frame_pos = int(frame_pos)
    frame_size = int(frame_size)
    time_graph = {}
    n_frames = 0
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(',')
        file = base64.b64decode(content_string)
        file = io.BytesIO(file)
        sample_rate, samples = read_file(file)

        n_frames = math.ceil(len(samples) * 1000 /
                             sample_rate / int(frame_size))
        frame_size_samp = len(samples) / n_frames

        time_graph = draw_audio(samples, list_of_names, [
            frame_pos*frame_size_samp, (frame_pos+1)*frame_size_samp])
    return time_graph, 'Selected frame length: ' + str(frame_size) + ' ms.', n_frames-1


@ app.callback(
    Output('param-graph', 'figure'),
    Input('param-dropdown', 'value'),
    Input('frame-size-in', 'value'),
)
def draw_param_graph(value, frame_size):
    global sample_rate, samples
    graph = {}
    frame_size = int(frame_size)
    if sample_rate is not None and samples is not None and value is not None:
        dict = {'Volume': volume, 'ZCR': zcr, 'STE': ste}
        graph = draw_plot(dict[value](samples, sample_rate, frame_size), value)
    return graph


port = 8050


def open_browser():
    webbrowser.open_new("http://localhost:{}".format(port))


app.title = 'dzwiek'

if __name__ == '__main__':
    # rate, samples = read_file('../wyewoluowac.wav')
    # print(volume(samples))
    # print(zcr(rate, samples))
    # print(sr(rate, samples))
    # draw_audio(samples).show()
    # debug()

    # open_browser()
    app.run_server(debug=True)
