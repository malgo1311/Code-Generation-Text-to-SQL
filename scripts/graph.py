import transformers
import dgl
import torch
import numpy as np


'''
Entire file made to create GNN but networks were not successful hence not using it.
'''

# Load T5 tokenizer and model
tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
t5_model = transformers.T5Model.from_pretrained('t5-base')

# Define function to create graph from NL statements
def create_graph(embeddings):
    '''

    :param embeddings: Normal input embeddings

    :return: graph which can be given as an input for GNN
    '''
    # # Tokenize NL statements and add special tokens
    # input_ids = tokenizer.batch_encode_plus(nl_statements, add_special_tokens=True, return_tensors='pt')['input_ids']
    # # Generate T5 embeddings for NL statements
    # with torch.no_grad():
    #     embeddings = t5_model(input_ids)[0].squeeze(1)
    # Compute similarity matrix based on cosine similarity
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    # Threshold similarity matrix to create edges
    threshold = torch.mean(similarity_matrix) + torch.std(similarity_matrix)
    edges = torch.nonzero(similarity_matrix > threshold)
    # Create graph from edges
    g = dgl.graph((edges[:, 0], edges[:, 1]))
    return g

# Define GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dgl.nn.GraphConv(hidden_feats, out_feats)

    def forward(self, g, h):
        h = torch.relu(self.conv1(g, h))
        h = self.conv2(g, h)
        return h

# Define function to generate GNN embeddings for NL statements
def generate_gnn_embeddings(input_ids,embeddings, gnn_model):
    g = create_graph(embeddings)
    h = t5_model(input_ids)['last_hidden_state'].squeeze(1)
    g.ndata['h'] = h
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    h = gnn_model(g, h)
    return h

# Define function to generate SQL skeleton using T5 decoder
def generate_sql_skeleton(nl_statements, gnn_embeddings):
    input_ids = tokenizer.batch_encode_plus(nl_statements, add_special_tokens=True, return_tensors='pt')['input_ids']
    input_ids = torch.cat([input_ids, gnn_embeddings], dim=1)
    output_ids = t5_model.generate(input_ids=input_ids, max_length=128)
    skeleton_sql = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return skeleton_sql
