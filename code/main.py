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


def draw_audio(samples, sample_rate, name, markers=None):
    fig = px.line(x=1000*np.arange(0, len(samples), 1)/sample_rate, y=samples)
    if markers is not None:
        for m in markers:
            fig.add_vline(x=m)
    fig.update_layout(title=name, yaxis_title="Amplitude",
                      xaxis_title="Time [ms]", showlegend=False, margin=dict(t=40))
    return fig


def draw_plot(values, name, marker=None):
    fig = px.line(values)
    if marker is not None:
        fig.add_vline(x=marker)
    fig.update_layout(title=name, yaxis_title="Value",
                      xaxis_title="Frames", showlegend=False, margin=dict(t=40))
    fig.update_xaxes(dtick=1)
    return fig


def volume(samples, rate=22000, frame_length=100, frame_overlap=0, no_samples=False):

    if no_samples:
        return np.sqrt(np.sum(samples*samples)/len(samples))

    n_rfames = math.ceil(
        (len(samples)*1000/rate - frame_overlap) // (frame_length-frame_overlap))

    vol = [0] * n_rfames

    for i in range(n_rfames-1):
        scope = samples[i*(frame_length-frame_overlap)*rate //
                        1000: ((i+1)*frame_length-i*frame_overlap)*rate//1000]
        vol[i] = np.sqrt(np.sum(np.square(scope))/len(scope))

    scope = samples[n_rfames*(frame_length-frame_overlap)*rate //
                    1000:]
    vol[-1] = np.sqrt(np.sum(np.square(scope))/len(scope))
    return vol


def ste(samples, rate=22000, frame_length=100, frame_overlap=0, no_samples=False):

    if no_samples:
        return np.sum(samples*samples)/len(samples)

    n_rfames = math.ceil(
        (len(samples)*1000/rate - frame_overlap) // (frame_length-frame_overlap))
    ste = [0] * n_rfames

    for i in range(n_rfames-1):

        scope = samples[i*(frame_length-frame_overlap)*rate //
                        1000: ((i+1)*frame_length-i*frame_overlap)*rate//1000]
        ste[i] = np.sum(np.square(scope))/len(scope)
    scope = samples[n_rfames*(frame_length-frame_overlap)*rate //
                    1000:]
    ste[-1] = np.sum(scope*scope)/len(scope)

    return ste


def zcr(samples, rate, frame_length=100, frame_overlap=0, no_samples=False):

    if no_samples:
        return np.sum(np.abs(np.subtract(np.sign(samples[1:]), np.sign(samples[:-1]))
                             )) * rate / len(samples) / 4

    n_rfames = math.ceil(
        (len(samples)*1000/rate - frame_overlap) // (frame_length-frame_overlap))

    zcr = [0] * n_rfames

    for i in range(n_rfames-1):

        scope = samples[i*(frame_length-frame_overlap)*rate //
                        1000: ((i+1)*frame_length-i*frame_overlap)*rate//1000]
        zcr[i] = np.sum(np.abs(np.subtract(np.sign(scope[1:]), np.sign(scope[:-1]))
                               )) * rate / len(scope) / 4

    scope = samples[n_rfames*(frame_length-frame_overlap)*rate //
                    1000:]
    zcr[-1] = np.sum(np.abs(np.subtract(np.sign(scope[1:]), np.sign(scope[:-1]))
                            )) * rate / len(scope) / 4

    return zcr


def sr(samples, rate, frame_length, frame_overlap=0):

    n_rfames = math.ceil(
        (len(samples)*1000/rate - frame_overlap) // (frame_length-frame_overlap))

    sr = [0] * n_rfames

    for i in range(n_rfames-1):
        scope = samples[i*(frame_length-frame_overlap)*rate //
                        1000: ((i+1)*frame_length-i*frame_overlap)*rate//1000]
        sr[i] = volume(scope, rate, frame_length, no_samples=True) < 100 and zcr(
            scope, rate, frame_length, no_samples=True) > 300

    scope = samples[n_rfames*(frame_length-frame_overlap)*rate //
                    1000:]
    sr[-1] = volume(scope, rate, frame_length, no_samples=True) < 100 and zcr(
        scope, rate, frame_length, no_samples=True) > 300

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


def saveCSV(samples, rate, frame_length, frame_overlap=0, path = None):
    header = ['frame_start', 'frame_end', 'volume', 'ste', 'zcr', 'isSilent']

    start = np.arange(0,  len(samples)/rate*1000 -
                      frame_length+1, frame_length-frame_overlap)
    end = np.append(np.arange(frame_length, len(samples)/rate*1000 - frame_length +
                    frame_overlap,  frame_length-frame_overlap), len(samples)/rate*1000)

    vol = volume(samples, rate, frame_length, frame_overlap)
    ste_ = ste(samples, rate, frame_length, frame_overlap)
    zcr = volume(samples, rate, frame_length, frame_overlap)
    silent = sr(samples, rate, frame_length, frame_overlap)

    data = np.array([start, end, vol, ste_, zcr, silent]).T

    if path is not None:
        with open(path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write multiple rows
            writer.writerows(data)
        return None

    with io.StringIO() as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

        content = f.getvalue()
    return content


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Wav files analysis',
            style={'text-align': 'center'}),

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

    html.H2('Clip-level statistics:'),

    dash.dash_table.DataTable(
        id='table-gen',
        data=[],
        style_cell={'textAlign': 'center', 'width': '25%'},
    ),

    html.H2('Frame-level statistics:'),

    dash.dash_table.DataTable(
        id='table-frame',
        data=[],
        style_cell={'textAlign': 'center', 'width': '25%'},
    ),

    html.H2('Properties:'),

    html.Div(children=['Input frame length [ms]: ',
                       dcc.Input(
                           id='frame-size-in',
                           value='20',
                           type='number',
                           min=5,
                           style={'width': '50px'},),
                       ]),

    html.Div(children=['Input frame overlap [ms]: ',
                       dcc.Input(
                           id='frame-overlap',
                           value='0',
                           type='number',
                           min=0,
                           style={'width': '50px'}, ),
                       ]),

    html.H3(id='frame-size-out'),

    html.H3('Choose frame:'),

    dcc.Slider(
        min=0,
        max=0,
        step=1,
        value=0,
        id='frame-slider',
        included=False,
    ),

    html.H2('Signal over time:'),

    dcc.Graph(
        id='time-graph',),

    html.H2('Frame-level statistics over time:'),

    dcc.Dropdown(['Volume', 'ZCR', 'STE'], id='param-dropdown'),

    dcc.Graph(
        id='param-graph',
    ),

    html.Button(
        'Download statistics to csv',
        id='download-button',
        style={
            'width': '50%',
            'height': '40px',
            'lineHeight': '40px',
            'borderWidth': '1px',
            'borderRadius': '5px',
            'textAlign': 'center',
            'display': 'block',
            'margin-left': 'auto',
            'margin-right': 'auto',
            'margin-top': '10px',
            'margin-bottom': '10px',
        }
    ),
    dcc.Download(id="download-csv"),
])


sample_rate, samples = None, None
frame_size_global = 20
frame_overlap_global = 0
file_name_global = None


@app.callback(
    Output('time-graph', 'figure'),
    Output('frame-size-out', 'children'),
    Output('frame-slider', 'max'),
    Output('table-frame', 'data'),
    Output('frame-slider', 'value'),
    Output('table-gen', 'data'),
    Input('upload-file', 'contents'),
    State('upload-file', 'filename'),
    State('upload-file', 'last_modified'),
    Input('frame-slider', 'value'),
    Input('frame-size-in', 'value'),
    Input('frame-overlap', 'value'),
)
def draw_graph_from_file(list_of_contents, list_of_names, list_of_dates, frame_pos, frame_size, frame_overlap):
    global sample_rate, samples, frame_size_global, file_name_global, frame_overlap_global
    frame_pos = int(frame_pos)
    frame_size = int(frame_size)
    frame_overlap = int(frame_overlap)
    time_graph = {}
    n_frames = 0
    table_frame_data = []
    table_gen_data = []
    if list_of_contents is not None:
        file_name_global = list_of_names
        content_type, content_string = list_of_contents.split(',')
        file = base64.b64decode(content_string)
        file = io.BytesIO(file)
        sample_rate, samples = read_file(file)

        if frame_size > 1000 * len(samples) // sample_rate // 2:
            frame_size = 1000 * len(samples) // sample_rate // 2
        if frame_overlap > frame_size - 1:
            frame_overlap = frame_size - 1
        frame_size_global = frame_size
        frame_overlap_global = frame_overlap

        n_frames = math.ceil(
            (len(samples) * 1000 / sample_rate - frame_overlap) // (frame_size - frame_overlap))
        if frame_pos > n_frames:
            frame_pos = 0

        time_graph = draw_audio(samples, sample_rate, list_of_names, [
            frame_pos*(frame_size-frame_overlap), frame_pos*(frame_size-frame_overlap)+frame_size  if frame_pos<n_frames-1 else 1000*len(samples)/sample_rate])

        table_frame_data = [{'Volume': volume(samples, sample_rate, frame_size, frame_overlap)[frame_pos],
                       'STE - Short Time Energy': ste(samples, sample_rate, frame_size, frame_overlap)[frame_pos],
                       'ZCR - Zero Crossing Rate': zcr(samples, sample_rate, frame_size, frame_overlap)[frame_pos],
                       'Silent': sr(samples, sample_rate, frame_size, frame_overlap)[frame_pos]}]

        lster_param = lster(samples, sample_rate, frame_size)

        table_gen_data = [{'VDR - Volume Dynamic Range': vdr(samples, sample_rate, frame_size),
                       'Mean Volume': mean(samples, sample_rate, frame_size),
                       'VSTD': std(samples, sample_rate, frame_size),
                       'LSTR - Low Short Time Energy Ratio': str(lster_param) + ' >= 0.15 -> speech' if lster_param >= 0.15 else str(lster_param) + ' < 0.15 -> music'}]
    return time_graph, 'Selected frame length: ' + str(frame_size) + ' ms with overlap: ' + str(frame_overlap) + ' ms.', \
           n_frames-1, table_frame_data, frame_pos, table_gen_data


@app.callback(
    Output('param-graph', 'figure'),
    Input('param-dropdown', 'value'),
    Input('frame-size-in', 'value'),
    Input('frame-overlap', 'value'),
    Input('frame-slider', 'value'),
)
def draw_param_graph(value, frame_size, frame_overlap, frame_pos):
    global sample_rate, samples, frame_size_global, frame_overlap_global
    graph = {}
    frame_size = int(frame_size)
    frame_pos = int(frame_pos)
    frame_overlap = int(frame_overlap)
    if sample_rate is not None and samples is not None and value is not None:
        if frame_size > 1000 * len(samples) // sample_rate // 2:
            frame_size = 1000 * len(samples) // sample_rate // 2
        if frame_overlap > frame_size - 1:
            frame_overlap = frame_size - 1
        frame_size_global = frame_size
        frame_overlap_global = frame_overlap
        dict = {'Volume': volume, 'ZCR': zcr, 'STE': ste}
        graph = draw_plot(dict[value](samples, sample_rate, frame_size, frame_overlap), value, frame_pos)
    return graph


@app.callback(
    Output('download-csv', 'data'),
    Input('download-button', 'n_clicks'),
    prevent_initial_call=True,
)
def button_on_click(n_clicks):
    global sample_rate, samples, frame_size_global, file_name_global, frame_overlap_global
    download_data = None
    if samples is not None and sample_rate is not None:
        file = saveCSV(samples, sample_rate, frame_size_global, frame_overlap_global)
        download_data = dict(content=file, filename=str.replace(file_name_global, '.wav', '.csv'))
    return download_data


port = 8050


def open_browser():
    webbrowser.open_new("http://localhost:{}".format(port))


app.title = 'Sound analysis'

if __name__ == '__main__':
    open_browser()
    app.run_server(debug=False)
