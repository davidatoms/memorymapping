import umap
import matplotlib.pyplot as plt
import numpy as np
import os
from embeddings import embeddings, words  # Import from our embeddings script

# Define word categories
categories = {
    'common': ['the', 'and', 'but'],
    'tech': ['computer', 'algorithm', 'data'],
    'actions': ['run', 'think', 'learn'],
    'concepts': ['intelligence', 'knowledge', 'wisdom'],
    'adjectives': ['smart', 'fast', 'complex'],
    'related': ['neural', 'network', 'brain']
}

# Create category labels for each word
word_categories = []
for word in words:
    for cat, cat_words in categories.items():
        if word in cat_words:
            word_categories.append(cat)
            break

# Reduce dimensionality to 2D
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=5)
embedding_2d = reducer.fit_transform(embeddings)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot points with different colors for each category
for cat in categories.keys():
    mask = np.array(word_categories) == cat
    if np.any(mask):
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], label=cat, alpha=0.7)

# Add labels for each point
for i, (word, cat) in enumerate(zip(words, word_categories)):
    plt.annotate(word, (embedding_2d[i, 0], embedding_2d[i, 1]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

plt.title("Word Embeddings Visualization\nColored by Category")
plt.xlabel("UMAP dimension 1")
plt.ylabel("UMAP dimension 2")
plt.grid(True, alpha=0.3)
plt.legend()

# Create graphs directory if it doesn't exist
os.makedirs("../../graphs", exist_ok=True)

# Save the plot
plt.savefig('../../graphs/word_embeddings_2d.png', dpi=100, bbox_inches='tight')
plt.show()
plt.close()