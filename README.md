# **Description**
(**DISCLAIMER**: I am in no means a Condensed Matter Physicist. I am an undergraduate Chemical Engineering major having fun with machine learning.)

Organic molecules, composed of carbon and hydrogen, often include other elements. Organic semiconductors, made from polymers, are increasingly used in electronic devices due to their environmental benefits and lower production costs compared to conventional semiconductors. Examples include metal complex pigments like hexa-deca-fluoro copper phthalocyanine for electron transport. These materials can be mixed into inks and printed onto substrates for flexible electronics.

The crystal lattice structure and melting point of an organic semiconductor are crucial factors. In a perfect crystal at low temperatures, electrons are in filled shells with no conduction electrons. Impurities or thermal excitations can provide some conduction electrons. At high temperatures, thermal fluctuations can knock electrons out of filled shells, increasing conductivity.

The goal of this project is to optimize the conductivity of an organic semiconductor by enhancing the symmetry of its crystal lattices, as symmetrical molecules have higher melting temperatures and lower solubilities compared to less symmetrical molecules.

## **Evaluation**
* **Category**: Computational Chemistry
* **Industry**: Semiconductor Manufacturing
* **Scope**:
  * Version 1: Using a Graph Neural Network. The rationale of utilizing a Graph Neural Network is based on the dynamic geometry of an organic molecule; where, the nodes represent the atoms of the molecule, the edges represent the bond of the molecule, and the symmetry being a feature of the graph. These organic molecules will be associated with the global features of its crystallization force and its melting point. These features of the GNN will help incorporate the relationship between conductivity and stability. This model utilizes the data from "Discovery of Crystallizable Organic Semiconductors with Machine Learning" by Holly M. Johnson, Filipp Gusev, Jordan T. Dull, Yejoon Seo, Rodney D. Priestley, Olexandr Isayev, and Barry P. Rand.

## Version 1
<p align="center">
  <img width="650" alt="image" src="https://github.com/user-attachments/assets/2583399b-7ff9-467e-aaa5-66288697735c">
</p>

<p align="center">
   <img width="653" alt="image" src="https://github.com/user-attachments/assets/92c134aa-4113-4135-b724-11130a96f873">
</p>

In order to establish a metric of symmetries in these organic molecules, we need to psi-six formula. The psi-six formula is used to determine the bond-orientational order parameter in two dimensional systems. It is commonly applied in the analysis of phase transitions and crystallization processes. 

<p align="center">
  <img width="351" alt="image" src="https://github.com/user-attachments/assets/bec57683-4f58-4d0a-8272-1b6802cd54a7">
</p>

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

## Citations
Shen, Z. S. “Exploring the Synthesis and Properties of Novel Semiconductor Materials.” ChemRxiv, 2023, chemrxiv.org/engage/chemrxiv/article-details/661eb94491aefa6ce1b1f0c9.

Pantel, Guillaume. “2D Bond Orientational Order Analysis Method.” Guillaume Pantel, 2023, gpantel.github.io/analysis-method/2D_boo/.










