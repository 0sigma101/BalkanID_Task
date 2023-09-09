from flask import Flask, request, jsonify
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import math

app = Flask(__name__)
df = pd.read_csv("Dataset.csv")

loaded_dict = torch.load('model.pt', map_location=torch.device('cpu'))

# Example model class
class GCNRecommender(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCNRecommender, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

df3 = pd.read_csv("coauthored_relationships.csv")
coauthored_relationships = []
for i in df3.index:
    relation = []
    relation.append(df3['coauthored_relationship_from'][i])
    relation.append(df3["coauthored_relationship_to"][i])
    coauthored_relationships.append(relation)

# Load author features and IDs
# You can load these from a file or database as needed
df2 = df.drop(["author_id","data"],axis=1)
lis = []
for i in df2.columns:
    lis.append(i)

author_features = {}
for index, row in df.iterrows():
    author_id = row['author_id']
    features = row[lis].tolist()
    features.remove(features[0])
    author_features[author_id] = features

author_ids = df['author_id']

# Create a mapping of author IDs to their corresponding indices
x = {author_id: idx for idx, author_id in enumerate(author_ids)}

# Create the edge_index from coauthored_relationships
edge_index = torch.tensor([[x[author_id1], x[author_id2]]
                           for author_id1, author_id2 in coauthored_relationships],
                          dtype=torch.long).t().contiguous()

# Instantiate the model
input_size = len(lis) - 1  # Number of features
hidden_size = 64
output_size = len(author_ids)  # Number of authors in the dataset
model = GCNRecommender(input_size, hidden_size, output_size)

# Load the model weights
model.conv1.lin.weight.data = loaded_dict['conv1.lin.weight']
model.conv1.bias.data = loaded_dict['conv1.bias']
model.conv2.lin.weight.data = loaded_dict['conv2.lin.weight']
model.conv2.bias.data = loaded_dict['conv2.bias']

model.eval()

author_features_tensor = torch.tensor([author_features[author_id] for author_id in author_ids], dtype=torch.float)

# Define a route for the GET request
@app.route('/')
def recommend_authors():
    global author_features_tensor, author_ids
    query_author_id = request.args.get('id')  # Get author ID from query parameter
    query_author_idx = x[query_author_id]
    num_recommendations = 5

    model.eval()
    with torch.no_grad():
        out = model(author_features_tensor, edge_index)
        query_embedding = out[query_author_idx]
        distances = torch.norm(out - query_embedding, dim=1)
        likeliness = 1.0 / (1.0 + distances)  # Likeliness inversely proportional to distance
        sorted_indices = torch.argsort(likeliness, descending=True)
        recommendations = [author_ids[idx.item()] for idx in sorted_indices if idx != query_author_idx][:num_recommendations]
        response = []
        for idx, author in enumerate(recommendations, start=1):
            likeness = 1.0 / (1.0 + math.log(1 + idx))
            response.append({
                "author": author,
                "likeness": likeness,
                "rank": idx
            })
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)