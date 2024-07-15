# **Description**
Organic molecules are molecules that are made of carbon and hydrogen, and can include other elements. An organic semiconductor refers to a type of material that is increasingly used in electronic device manufacturing, such as metal complex pigments like hexa-deca-fluoro copper phthalocyanine, for electron transport. Organic semiconductors, unlike conventional semiconductors, are made of polymers, which makes them more environmentally friendly and less costly to produce. New organic conductive materials, which can be mixed into inks and printed onto substrates, are now being used to produce flexible electronics. 

A factor in what makes an Organic Semiconductor is the the crystal lattices and the melting point of the molecule. In a perfect crystal at low temperatures, all electrons are in filled shells and there are no conduction electrons. However, impurities or thermal excitations can provide some conduction electrons. At high temperatures, thermal fluctuations can cause large ion vibrations that knock electrons out of filled shells, turning them into conduction electrons that contribute to conductivity.

The object of this product is to optimize the conductivity of an organic semiconductor based on the symmetry of its crystal lattices. This is because symmetrical molecules in crystalline form have higher melting temperatures and exhibit lower solubilities compared with molecules of similar structure but with lower symmetry.

## **Evaluation**
* **Category**: Computational Chemistry
* **Industry**: Semiconductor Manufacturing
* **Scope**:
  * Version 1: Using a Graph Neural Network. The rationale of utilizing a Graph Neural Network is based on the dynamic geometry of an organic molecule; where, the nodes represent the atoms of the molecule, the edges represent the bond of the molecule, and the symmetry being a feature of the graph. These organic molecules will be associated with the global features of its crystallization force and its melting point. These features of the GNN will help incorporate the relationship between conductivity and stability.

## Code Overview

### Reading CSV Files

| Function               | Description                                         |
|------------------------|-----------------------------------------------------|
| read_crystallization_dataset     | Reads the crystallization dataset (Dataset_dHm.csv).|
| read_melting_point_dataset       | Reads the melting point dataset (Dataset_Tm.csv).   |

### Bond Orientation Order Parameter

| Function               | Description                                         |
|------------------------|-----------------------------------------------------|
| calculate_psi6    | It measures the degree of symmetry in the arrangement of atoms or molecules within the lattice. One commonly used bond orientation order parameter is the Psi_6 parameter, which is particularly useful for detecting hexagonal symmetry.|



| Function               | Description                                         |
|------------------------|-----------------------------------------------------|
| read_crystallization_dataset     | Reads the crystallization dataset (Dataset_dHm.csv).|
| read_melting_point_dataset       | Reads the melting point dataset (Dataset_Tm.csv).   |

### Converting SMILES to Graph

#### smiles_to_graph Function

| Step                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| convert_smiles_to_graph       | Converts a SMILES string into a graph representation.                     |
| add_hydrogens        | Adds hydrogen atoms to the molecule.                                        |
| extract_features     | Extracts atomic numbers as node features and bond types as edge attributes. |
| create_tensors       | Creates tensors for node features (node_features), edge indices (edge_index), and edge attributes (edge_attr). |
| embed_in_3d_space    | Embeds the molecule in 3D space and retrieves atomic positions (pos).       |
| return_data_object   | Returns a data object containing the graph representation.|

### Processing Data

#### process_data Function

| Step                      | Description                                                                            |
|---------------------------|----------------------------------------------------------------------------------------|
| iterate_rows             | Iterates over each row in the dataframe.                                               |
| convert_smiles_to_graph  | Converts each SMILES string to a graph representation.                          |
| add_target_as_label      | Adds the target value (e.g., dHm or Tm) as a label to the graph.                       |
| collect_graph_data       | Collects valid graph data objects into a list.                                         |

#### Creating Data Lists

| Data List                   | Description                                                           |
|-----------------------------|-----------------------------------------------------------------------|
| crystallization_data_list   | List of graph data objects for crystallization data.                  |
| melting_point_data_list     | List of graph data objects for melting point data.                    |

### Custom Message-Passing Layer

#### SymmetryAwareMessagePassing Class

| Method                    | Description                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| Inherits MessagePassing   | Inherits from MessagePassing in PyTorch Geometric.                                                  |
| forward                   | Adds self-loops and propagates messages.                                                            |
| message                   | Computes messages based on neighboring node features, edge attributes, and distances between atoms. |
| update                    | Updates node features after message aggregation.                                                    |

### Symmetry-Aware Loss Function

#### symmetry_invariant_loss Function

| Step                        | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| combine_mse_with_symmetry  | Combines mean squared error (MSE) loss with a symmetry loss.                                    |
| calculate_symmetry_metric   | Placeholder function to compute a symmetry-related metric for a molecule.                        |
| total_loss                  | Sum of MSE loss and symmetry loss, weighted by lambda_symmetry.                                 |

### Visualization

#### visualize_graph Function

| Step                      | Description                                             |
|---------------------------|---------------------------------------------------------|
| convert_to_networkx      | Converts a data object to a NetworkX graph. |
| visualize_graph          | Visualizes the graph using Matplotlib.                  |

### Visualization Execution

| Task                      | Description                                                         |
|---------------------------|---------------------------------------------------------------------|
| visualize_first_graph    | Visualizes the first valid molecular graph in crystallization_data_list and melting_point_data_list. |

### Advanced GNN

#### AdvancedSymmetryAwareGNN Class

| Step                      | Description                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| Advanced GNN              | The AdvancedSymmetryAwareGNN class integrates node features, edge features, distances, and a symmetry metric Psi_6 into the message passing process using multi-layer perceptrons (MLPs) to enhance the model's performance by leveraging symmetry information. |

### Symmetry Metric Calculation

#### calculate_symmetry_metric Function

| Step                      | Description                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| def calculate_symmetry_metric(molecule)    | Function that returns the metric of symmetry generated by the GNN|
