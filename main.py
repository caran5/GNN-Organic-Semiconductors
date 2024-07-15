import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.utils import add_self_loops

crystallization_df = pd.read_csv('datasets/Dataset_dHm.csv')
melting_point_df = pd.read_csv('datasets/Dataset_Tm.csv')

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    nodes = torch.tensor(node_feats, dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))
        edge_attr.append(bond.GetBondTypeAsDouble())
        edge_attr.append(bond.GetBondTypeAsDouble())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=nodes.view(-1, 1), edge_index=edge_index, edge_attr=edge_attr)

graph_data_list_dHm = [smiles_to_graph(smiles) for smiles in crystallization_df['SMILES'] if smiles_to_graph(smiles) is not None]
graph_data_list_Tm = [smiles_to_graph(smiles) for smiles in melting_point_df['SMILES'] if smiles_to_graph(smiles) is not None]

data_dHm = graph_data_list_dHm[0]
data_Tm = graph_data_list_Tm[0]

graph_dHm = nx.Graph()
graph_Tm = nx.Graph()

for i, node_feat in enumerate(data_dHm.x):
    graph_dHm.add_node(i, atomic_num=int(node_feat.item()))

edge_indices_dHm = data_dHm.edge_index.t().tolist()
for start, end in edge_indices_dHm:
    graph_dHm.add_edge(start, end)

for i, node_feat in enumerate(data_Tm.x):
    graph_Tm.add_node(i, atomic_num=int(node_feat.item()))

edge_indices_Tm = data_Tm.edge_index.t().tolist()
for start, end in edge_indices_Tm:
    graph_Tm.add_edge(start, end)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(graph_dHm)
node_labels = nx.get_node_attributes(graph_dHm, 'atomic_num')
nx.draw(graph_dHm, pos, with_labels=True, labels=node_labels, node_color='skyblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
plt.title("Molecular Graph: Crystallization Driving Force (dHm) Dataset")
plt.show()

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(graph_Tm)
node_labels = nx.get_node_attributes(graph_Tm, 'atomic_num')
nx.draw(graph_Tm, pos, with_labels=True, labels=node_labels, node_color='skyblue', node_size=500, font_size=10, font_color='black', edge_color='gray')
plt.title("Molecular Graph: Melting Temperature (Tm) Dataset")
plt.show()

def calculate_psi6(pos):
    N = len(pos)
    psi6_sum = 0
    for i in range(N):
        neighbors = pos[np.linalg.norm(pos - pos[i], axis=1) < 3.0]
        
        neighbors = neighbors[~np.all(neighbors == pos[i], axis=1)]
        
        if len(neighbors) == 0:
            continue
        
        angles = np.arctan2(neighbors[:, 1] - pos[i, 1], neighbors[:, 0] - pos[i, 0])
        psi6_sum += np.sum(np.exp(6j * angles))
    
    psi6 = psi6_sum / N
    return np.abs(psi6)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    node_features = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    
    edge_indices = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attr.append(bond.GetBondTypeAsDouble())
        edge_attr.append(bond.GetBondTypeAsDouble())
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    
    if AllChem.EmbedMolecule(mol) == -1:
        return None
    
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    
    psi6 = calculate_psi6(pos.numpy())
    psi6_tensor = torch.tensor([psi6], dtype=torch.float)
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos, psi6=psi6_tensor)

def process_data(df, target_column):
    data_list = []
    for _, row in df.iterrows():
        smiles = row['SMILES']
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph.y = torch.tensor([row[target_column]], dtype=torch.float)
            data_list.append(graph)
    return data_list

crystallization_data_list = process_data(crystallization_df, 'dHm')
melting_point_data_list = process_data(melting_point_df, 'Tm')

class AdvancedSymmetryAwareGNN(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super(AdvancedSymmetryAwareGNN, self).__init__(aggr='add')
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        
        self.symmetry_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        
        self.output_mlp = torch.nn.Sequential(
            torch.nn.Linear(192, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_attr, pos, psi6):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos, psi6=psi6)

    def message(self, x_j, edge_attr, pos_i, pos_j, psi6):
        distance = (pos_i - pos_j).norm(dim=1).view(-1, 1)
        
        node_message = self.node_mlp(x_j)
        edge_message = self.edge_mlp(edge_attr.view(-1, 1))
        symmetry_message = self.symmetry_mlp(psi6.view(-1, 1))
        
        message = node_message + edge_message + distance + symmetry_message
        return message

    def update(self, aggr_out):
        return self.output_mlp(aggr_out)

def calculate_symmetry_metric(molecule):
    return molecule.psi6

def symmetry_invariant_loss(pred, target, molecule):
    mse_loss = F.mse_loss(pred, target)
    symmetry_loss = calculate_symmetry_metric(molecule)
    lambda_symmetry = 0.1  
    total_loss = mse_loss + lambda_symmetry * symmetry_loss
    return total_loss

def calculate_metrics(predictions, targets):
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, r2

def evaluate_model(model, data_list, target_column):
    model.eval()
    predictions = []
    targets = []
    psi6_values = []

    for data in data_list:
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_attr, data.pos, data.psi6)
            predictions.append(pred.item())
            targets.append(data.y.item())
            psi6_values.append(data.psi6.item())

    mse, r2 = calculate_metrics(predictions, targets)
    psi6_mean = np.mean(psi6_values)

    return mse, r2, psi6_mean

node_dim = 32  
edge_dim = 32 

crystallization_model = AdvancedSymmetryAwareGNN(node_dim, edge_dim)
melting_point_model = AdvancedSymmetryAwareGNN(node_dim, edge_dim)

optimizer = torch.optim.Adam(crystallization_model.parameters(), lr=0.01)
loss_function = symmetry_invariant_loss

num_epochs = 5  
for epoch in range(num_epochs):
    crystallization_model.train()
    total_loss = 0
    for data in crystallization_data_list:
        optimizer.zero_grad()
        pred = crystallization_model(data.x, data.edge_index, data.edge_attr, data.pos, data.psi6)
        loss = loss_function(pred, data.y, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(crystallization_data_list):.4f}')

optimizer = torch.optim.Adam(melting_point_model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    melting_point_model.train()
    total_loss = 0
    for data in melting_point_data_list:
        optimizer.zero_grad()
        pred = melting_point_model(data.x, data.edge_index, data.edge_attr, data.pos, data.psi6)
        loss = loss_function(pred, data.y, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(melting_point_data_list):.4f}')

crystallization_mse, crystallization_r2, crystallization_psi6_mean = evaluate_model(crystallization_model, crystallization_data_list, 'dHm')
melting_point_mse, melting_point_r2, melting_point_psi6_mean = evaluate_model(melting_point_model, melting_point_data_list, 'Tm')

print("Crystallization Force Metrics:")
print(f"MSE: {crystallization_mse:.4f}, R2: {crystallization_r2:.4f}, Mean Psi_6: {crystallization_psi6_mean:.4f}")

print("Melting Point Metrics:")
print(f"MSE: {melting_point_mse:.4f}, R2: {melting_point_r2:.4f}, Mean Psi_6: {melting_point_psi6_mean:.4f}")