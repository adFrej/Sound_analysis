import numpy as np
from scipy.io import wavfile
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import webbrowser

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

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Dźwięk - projekt 1',
            style={'color': 'blue'}),
    dcc.Graph(
        id='time_graph',
        figure=draw_audio(read_file('../wyewoluowac.wav')[1])),
])

# @app.callback(
#     Output('', ''),
#     Input('', ''),
# )
# def update_output():
#

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