import plotly.express as px
import pandas as pd
import os
from embeddings import words
from dimensionality_reduction import embedding_2d, word_categories

# Create a DataFrame with the 2D embeddings and words
df = pd.DataFrame({
    'x': embedding_2d[:, 0],
    'y': embedding_2d[:, 1],
    'word': words,
    'category': word_categories
})

# Create interactive plot
fig = px.scatter(df, x='x', y='y',
                color='category',
                text='word',
                title="Interactive Word Embeddings",
                labels={'x': 'UMAP dimension 1',
                       'y': 'UMAP dimension 2',
                       'category': 'Word Category'},
                hover_data=['word', 'category'])

# Update layout
fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_layout(
    showlegend=True,
    hovermode='closest',
    title_x=0.5,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Create graphs directory if it doesn't exist
os.makedirs("../../graphs", exist_ok=True)

# Save and show the plot
fig.write_html("../../graphs/interactive_word_embeddings.html")
fig.show()