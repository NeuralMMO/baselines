from pdb import set_trace as T
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import dill
try:
    import dash
    from dash import dcc, html
except:
    print('pip install dash to use this script')

CURRICULUM_FILE_PATH = "reinforcement_learning/curriculum_with_embedding.pkl"


class TaskEmbeddingVisualizer:
    def __init__(self, curriculum_file_path):
        with open(curriculum_file_path, 'rb') as f:
            # TODO: de-duplication. Although, the manual curriculum doesn't have duplicates.
            self.curriculum = dill.load(f)
        self.embeddings = np.array([single_spec.embedding for single_spec in self.curriculum])

    def visualize(self, dims=2):
        task_names = [single_spec.name for single_spec in self.curriculum]
        colors = [f"#00{format(int(val), '02x')}ff" for val in np.linspace(0, 255, len(task_names))]
        task_to_color = {eval_fn: color for eval_fn, color in zip(task_names, colors)}

        tsne = TSNE(n_components=dims, random_state=42, perplexity=23)
        embeddings_tsne = tsne.fit_transform(self.embeddings)

        x, y, z, colors = [], [], [], []
        for i, emb in enumerate(embeddings_tsne):
            x.append(emb[0])
            y.append(emb[1])
            z.append(emb[2] if dims==3 else 0)
            colors.append(task_to_color[task_names[i]])

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
            hovertext=task_names,
            name="Embeddings"
        )

        return [trace]

# usage
visualizer = TaskEmbeddingVisualizer(CURRICULUM_FILE_PATH)
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