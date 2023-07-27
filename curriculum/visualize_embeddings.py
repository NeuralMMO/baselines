from pdb import set_trace as T
import json
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
try:
    import dash
except:
    print('pip install dash to use this script')

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pickle

import pandas as pd
import inspect


def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    eval_fn_array = []
    embedding_array = []
    for d in data:
        func_name = d[1].__name__
        func_args = d[2]
        func_src = inspect.getsource(d[1])
        embeddings = d[3]['embedding'].tolist()

        # Combine function name, args and source code
        eval_fn = f"{func_name} {func_args}\n\n{func_src}"
        
        eval_fn_array.append(eval_fn)
        embedding_array.append(embeddings)

    # Convert to numpy arrays
    eval_fn_array = np.array(eval_fn_array)
    embedding_array = np.array(embedding_array)

    return eval_fn_array, embedding_array


class TaskEmbeddingVisualizer:
    def __init__(self, json_file_path):
        self.eval_fns, self.embeddings = load_data_from_pickle(json_file_path)

    def visualize(self, dims=2):
        unique_eval_fns = list(set(self.eval_fns))
        colors = [f"#00{format(int(val), '02x')}ff" for val in np.linspace(0, 255, len(unique_eval_fns))]
        eval_fn_to_color = {eval_fn: color for eval_fn, color in zip(unique_eval_fns, colors)}

        tsne = TSNE(n_components=dims, random_state=42, perplexity=23)
        embeddings_tsne = tsne.fit_transform(self.embeddings)

        x, y, z, colors, eval_fns = [], [], [], [], []
        for i, emb in enumerate(embeddings_tsne):
            x.append(emb[0])
            y.append(emb[1])
            z.append(emb[2] if dims==3 else 0)
            colors.append(eval_fn_to_color[self.eval_fns[i]])
            eval_fns.append(self.eval_fns[i].replace('\n', '<br>'))

        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                color=colors, 
                size=6, 
                symbol='circle', 
                opacity=0.8
            ),
            hovertemplate='%{hovertext}',
            hovertext=eval_fns,
            name="Embeddings"
        )

        return [trace]

# usage
visualizer = TaskEmbeddingVisualizer("pickled_task_with_embs.pkl")
traces = visualizer.visualize(dims=3)

# Create Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='3d-scatter', 
        figure=go.Figure(
            data=traces,
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                    yaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="white", showbackground=True, zerolinecolor="white"),
                    zaxis=dict(backgroundcolor="rgba(0, 0, 0,0)", gridcolor="white", showbackground=True, zerolinecolor="white")
                ),
                legend=dict(yanchor="top", y=1, xanchor="left", x=0),
                margin=dict(l=0, r=0, b=0, t=0),
                plot_bgcolor='rgba(6,26,26,1)',
                paper_bgcolor='rgba(6,26,26,1)',
                scene_bgcolor='rgba(6,26,26,1)', 
                font=dict(color='white')
            )
        )
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)