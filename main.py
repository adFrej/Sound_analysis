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


class AudioFile:
    def __init__(self, file_name, file_path, frame_length, frame_overlap):
        self.file_name = file_name
        self.file_path = file_path
        self.samples, self.sample_rate = self.read_file()
        self.frame_length = frame_length
        self.frame_overlap = frame_overlap

    def read_file(self):
        sampling_rate, samples = wavfile.read(self.file_path)
        samples = samples.astype('int64')
        return samples, sampling_rate

    def draw_audio(self, name, markers=None):
        fig = px.line(x=1000*np.arange(0, len(self.samples), 1)/self.sample_rate, y=self.samples)
        if markers is not None:
            for m in markers:
                fig.add_vline(x=m)
        fig.update_layout(title=name, yaxis_title="Amplitude",
                          xaxis_title="Time [ms]", showlegend=False, margin=dict(t=40))
        return fig


    def draw_plot(self, values, name, marker=None):
        fig = px.line(values)
        if marker is not None:
            fig.add_vline(x=marker)
        fig.update_layout(title=name, yaxis_title="Value",
                          xaxis_title="Frames", showlegend=False, margin=dict(t=40))
        fig.update_xaxes(dtick=1)
        return fig


    def volume(self, samples_scope=None, no_samples=False):
        if samples_scope is None:
            samples_scope=self.samples

        if no_samples:
            return np.sqrt(np.sum(samples_scope*samples_scope)/len(samples_scope))

        n_frames = math.ceil(
            (len(samples_scope)*1000/self.sample_rate - self.frame_overlap) // (self.frame_length-self.frame_overlap))

        vol = [0] * n_frames

        for i in range(n_frames-1):
            scope = samples_scope[i*(self.frame_length-self.frame_overlap)*self.sample_rate //
                            1000: ((i+1)*self.frame_length-i*self.frame_overlap)*self.sample_rate//1000]
            vol[i] = np.sqrt(np.sum(np.square(scope))/len(scope))

        scope = samples_scope[n_frames*(self.frame_length-self.frame_overlap)*self.sample_rate //
                        1000:]
        vol[-1] = np.sqrt(np.sum(np.square(scope))/len(scope))
        return vol


    def ste(self, samples_scope=None, no_samples=False):
        if samples_scope is None:
            samples_scope = self.samples

        if no_samples:
            return np.sum(samples_scope*samples_scope)/len(samples_scope)

        n_frames = math.ceil(
            (len(samples_scope)*1000/self.sample_rate - self.frame_overlap) // (self.frame_length-self.frame_overlap))
        ste = [0] * n_frames

        for i in range(n_frames-1):

            scope = samples_scope[i*(self.frame_length-self.frame_overlap)*self.sample_rate //
                            1000: ((i+1)*self.frame_length-i*self.frame_overlap)*self.sample_rate//1000]
            ste[i] = np.sum(np.square(scope))/len(scope)
        scope = samples_scope[n_frames*(self.frame_length-self.frame_overlap)*self.sample_rate //
                        1000:]
        ste[-1] = np.sum(scope*scope)/len(scope)

        return ste


    def zcr(self, samples_scope=None,no_samples=False):
        if samples_scope is None:
            samples_scope=self.samples

        if no_samples:
            return np.sum(np.abs(np.subtract(np.sign(samples_scope[1:]), np.sign(samples_scope[:-1]))
                                 )) * self.sample_rate / len(samples_scope) / 4

        n_frames = math.ceil(
            (len(samples_scope)*1000/self.sample_rate - self.frame_overlap) // (self.frame_length-self.frame_overlap))

        zcr = [0] * n_frames

        for i in range(n_frames-1):

            scope = samples_scope[i*(self.frame_length-self.frame_overlap)*self.sample_rate //
                            1000: ((i+1)*self.frame_length-i*self.frame_overlap)*self.sample_rate//1000]
            zcr[i] = np.sum(np.abs(np.subtract(np.sign(scope[1:]), np.sign(scope[:-1]))
                                   )) * self.sample_rate / len(scope) / 4

        scope = samples_scope[n_frames*(self.frame_length-self.frame_overlap)*self.sample_rate //
                        1000:]
        zcr[-1] = np.sum(np.abs(np.subtract(np.sign(scope[1:]), np.sign(scope[:-1]))
                                )) * self.sample_rate / len(scope) / 4

        return zcr


    def sr(self):

        n_frames = math.ceil(
            (len(self.samples)*1000/self.sample_rate - self.frame_overlap) // (self.frame_length-self.frame_overlap))

        sr = [0] * n_frames

        for i in range(n_frames-1):
            scope = self.samples[i*(self.frame_length-self.frame_overlap)*self.sample_rate //
                            1000: ((i+1)*self.frame_length-i*self.frame_overlap)*self.sample_rate//1000]
            sr[i] = self.volume(samples_scope=scope, no_samples=True) < 100 and self.zcr(
                samples_scope=scope, no_samples=True) > 300

        scope = self.samples[n_frames*(self.frame_length-self.frame_overlap)*self.sample_rate //
                        1000:]
        sr[-1] = self.volume(samples_scope=scope, no_samples=True) < 100 and self.zcr(
            samples_scope=scope, no_samples=True) > 300

        return sr

    #      VVVV Klip scope VVVV


    def vdr(self):
        v = self.volume()
        return (max(v) - min(v)) / max(v)


    def mean(self):
        vol = self.volume()
        return sum(vol)/len(vol)


    def std(self):
        return statistics.stdev(self.volume())


    def lster(self):

        sum = 0
        avSTE = []

        for i in range(math.ceil(len(self.samples)/self.sample_rate)):
            avSTE.append(
                self.ste(samples_scope=self.samples[i*self.sample_rate: min((i+1)*self.sample_rate, len(self.samples))], no_samples=True))

        for i in range(math.ceil(len(self.samples)*1000/(self.frame_length*self.sample_rate))):
            av_index = math.floor(i*self.frame_length / 1000)

            sum = sum + (0.5*avSTE[av_index] - self.ste(samples_scope=self.samples[i *
                                                           self.frame_length: min((i+1)*self.frame_length, len(self.samples))], no_samples=True) > 0)

        return sum/(2*i)


    def saveCSV(self, path = None):
        header = ['frame_start', 'frame_end', 'volume', 'ste', 'zcr', 'isSilent']

        start = np.arange(0,  len(self.samples)/self.sample_rate*1000 -
                          self.frame_length+1, self.frame_length-self.frame_overlap)
        end = np.append(np.arange(self.frame_length, len(self.samples)/self.sample_rate*1000 - self.frame_length +
                        self.frame_overlap,  self.frame_length-self.frame_overlap), len(self.samples)/self.sample_rate*1000)

        vol = self.volume()
        ste_ = self.ste()
        zcr = self.volume()
        silent = self.sr()

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
server = app.server

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

audio_file = None

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
    global audio_file
    frame_pos = int(frame_pos)
    frame_size = int(frame_size)
    frame_overlap = int(frame_overlap)
    time_graph = {}
    n_frames = 0
    table_frame_data = []
    table_gen_data = []
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(',')
        file = base64.b64decode(content_string)
        file = io.BytesIO(file)
        audio_file = AudioFile(list_of_names, file, frame_size, frame_overlap)

        if frame_size > 1000 * len(audio_file.samples) // audio_file.sample_rate // 2:
            frame_size = 1000 * len(audio_file.samples) // audio_file.sample_rate // 2
        if frame_overlap > frame_size - 1:
            frame_overlap = frame_size - 1

        n_frames = math.ceil(
            (len(audio_file.samples) * 1000 / audio_file.sample_rate - frame_overlap) // (frame_size - frame_overlap))
        if frame_pos > n_frames:
            frame_pos = 0

        time_graph = audio_file.draw_audio(list_of_names, [
            frame_pos*(frame_size-frame_overlap), frame_pos*(frame_size-frame_overlap)+frame_size  if frame_pos<n_frames-1 else 1000*len(audio_file.samples)/audio_file.sample_rate])

        table_frame_data = [{'Volume': audio_file.volume()[frame_pos],
                       'STE - Short Time Energy': audio_file.ste()[frame_pos],
                       'ZCR - Zero Crossing Rate': audio_file.zcr()[frame_pos],
                       'Silent': audio_file.sr()[frame_pos]}]

        lster_param = audio_file.lster()

        table_gen_data = [{'VDR - Volume Dynamic Range': audio_file.vdr(),
                       'Mean Volume': audio_file.mean(),
                       'VSTD': audio_file.std(),
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
    global audio_file
    graph = {}
    frame_size = int(frame_size)
    frame_pos = int(frame_pos)
    frame_overlap = int(frame_overlap)
    if audio_file is not None and value is not None:
        if frame_size > 1000 * len(audio_file.samples) // audio_file.sample_rate // 2:
            frame_size = 1000 * len(audio_file.samples) // audio_file.sample_rate // 2
        if frame_overlap > frame_size - 1:
            frame_overlap = frame_size - 1
        audio_file.frame_length = frame_size
        audio_file.frame_overlap = frame_overlap
        dict = {'Volume': audio_file.volume, 'ZCR': audio_file.zcr, 'STE': audio_file.ste}
        graph = audio_file.draw_plot(dict[value](), value, frame_pos)
    return graph


@app.callback(
    Output('download-csv', 'data'),
    Input('download-button', 'n_clicks'),
    prevent_initial_call=True,
)
def button_on_click(n_clicks):
    global audio_file
    download_data = None
    if audio_file is not None:
        file = audio_file.saveCSV()
        download_data = dict(content=file, filename=str.replace(audio_file.file_name, '.wav', '.csv'))
    return download_data


port = 8050


def open_browser():
    webbrowser.open_new("http://localhost:{}".format(port))


# VVVVVVVVVVVVVVVVVVVVVVV projekt 2 VVVVVVVVVVVVVVVVVVVVVVVVVV


def widmo(samples, okno=""):
    okno = okno.lower()
    if okno=="hamming":
        return np.fft.rfft(samples*np.hamming(len(samples)))
    if okno=="hann":
        return np.fft.rfft(samples*np.hanning(len(samples)))
    if okno=="blackman":
        return np.fft.rfft(samples*np.blackman(len(samples)))
    
    return np.fft.rfft(samples)


def freq(samples,scale):
    return np.fft.rfftfreq(samples,scale)



app.title = 'Sound analysis'

if __name__ == '__main__':
    # open_browser()
    #app.run_server(debug=False)
    pp = AudioFile('dun','./dun.wav',None,None)
    [samples, sampling_rate] = pp.read_file()
    print(len(samples),"samples ",samples)
    w = np.abs(widmo(samples, "blackman"))/len(samples)*2
    f = freq(len(samples), 1/sampling_rate)
    print(len(w),"widmo ",w)
    print(len(f),"frequ ",f)
    fig = px.line(x=f, y=w)
    fig.update_layout(title="Widmo rzeczywiste", yaxis_title="amplituda widma",
                      xaxis_title="częstotliwość [Hz]", showlegend=False, margin=dict(t=40))
    fig.show()
