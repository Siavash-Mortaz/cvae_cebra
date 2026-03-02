import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from cvae_model import CVAE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import cebra
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm



# import pandas as pd

# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

def load_data(dataset_path):
    dataset=dataset_path
    with open(dataset,'rb') as data_file:
        data = pickle.load(data_file)

    hand_train, hand_val, hand_test = data['hand_train'], data['hand_val'], data['hand_test']
    obj_train, obj_val, obj_test = data['obj_train'], data['obj_val'], data['obj_test']
    obj_names_train, obj_names_val, obj_names_test = np.array(data['obj_names'])[data['train_indices']],np.array(data['obj_names'])[data['val_indices']],np.array(data['obj_names'])[data['test_indices']]

    return hand_train, hand_val, hand_test,obj_train, obj_val, obj_test ,obj_names_train, obj_names_val, obj_names_test

def load_tensor_data(obj_train,hand_train,obj_val,hand_val,obj_test,hand_test):
    batch_size = 64
    train_dataset = TensorDataset(torch.tensor(obj_train, dtype=torch.float32),torch.tensor(hand_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(obj_val, dtype=torch.float32), torch.tensor(hand_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(obj_test, dtype=torch.float32),torch.tensor(hand_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader,val_loader,test_loader

def compute_latent(model, data_loader, model_weight):
    model.load_state_dict(torch.load(model_weight))
    model.eval()
    latent_list = []
    with torch.no_grad():
        for obj, hand in data_loader:
            # obj = obj.view(-1, 2623, 3)

            c = model.object_encoder(obj)
            h = model.hand_encoder(hand)
            # Use the posterior mean as the latent representation.
            mu_q, _ = model.posterior_net(h, c)
            latent_list.append(mu_q.cpu().numpy())
    latent_all = np.concatenate(latent_list, axis=0)
    return latent_all

def plot_pca_explained_variance_ratio(pca):
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance Ratio")
    plt.grid(True)
    plt.show()

    explained_variance = pca.explained_variance_ratio_

    # Print cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    for i, var in enumerate(cumulative_variance):
        print(f"First {i + 1} components explain {var:.2%} of the variance")


def plot_3Dpca(pca_results,object_name, x,y,z):
    # Assuming 'pca_results' and 'object_names' are already defined
    pca_df = pd.DataFrame(pca_results[:, :], columns=[f'PC{i+1}' for i in range(pca_results.shape[1])])
    pca_df['Object Name'] = object_name  # Add object names for grouping


    # Create a 3D scatter plot with Plotly Express
    fig = px.scatter_3d(
        pca_df,
        x=f'PC{x}',
        y=f'PC{y}',
        z=f'PC{z}',
        color='Object Name',  # Different colors by object name
        hover_name='Object Name',  # Show object name on hover
        title="3D PCA of Latent Space Grouped by Object Names"
    )

    # Update axis labels
    fig.update_layout(height=1000,
                      scene=dict(
                          xaxis_title=f'Principal Component {x}',
                          yaxis_title=f'Principal Component {y}',
                          zaxis_title=f'Principal Component {z}'
                      )
                      )

    fig.show()


def plot_2Dpca(pca_results,object_name, x,y):
    pca_df = pd.DataFrame(pca_results[:, :], columns=[f'PC{i+1}' for i in range(pca_results.shape[1])])
    pca_df['Object Name'] = object_name  # Add object names for grouping

    # print(pca_df)

    # Unique object names for plotting
    unique_objects = np.unique(obj_names_test)

    # Plot PCA results for each object
    plt.figure(figsize=(12, 8))
    for obj_name in unique_objects:
        # if obj_name=='035_power_drill'or obj_name=='003_cracker_box':

        subset = pca_df[pca_df['Object Name'] == obj_name]
        plt.scatter(subset[f'PC{x}'], subset[f'PC{y}'], label=obj_name, alpha=0.6)

    # Add plot legend and labels
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Object Names")
    plt.xlabel(f"Principal Component {x}")
    plt.ylabel(f"Principal Component {y}")
    plt.title("PCA of Latent Space Grouped by Object Names")
    plt.tight_layout()
    plt.show()


def compute_cebra_embeddings(latent_vectors, object_names, output_dim=3, max_iterations=5000):
    """
    Compute CEBRA embeddings from latent vectors using object names as discrete labels.
    
    Parameters:
    - latent_vectors: numpy array of shape (n_samples, n_features)
    - object_names: numpy array of object names (discrete labels)
    - output_dim: dimensionality of the output embedding (default: 3)
    - max_iterations: maximum number of training iterations
    
    Returns:
    - cebra_embeddings: numpy array of shape (n_samples, output_dim)
    - cebra_model: trained CEBRA model
    - label_encoder: LabelEncoder used to convert object names to integers
    """
    # Convert object names to integer labels for CEBRA
    label_encoder = LabelEncoder()
    discrete_labels = label_encoder.fit_transform(object_names)
    
    print(f"Training CEBRA model with {len(np.unique(discrete_labels))} unique object classes...")
    print(f"Input dimension: {latent_vectors.shape[1]}, Output dimension: {output_dim}")
    print(f"Number of samples: {latent_vectors.shape[0]}")
    
    # Initialize CEBRA model with discrete labels
    # For discrete labels, we use the discrete mode
    cebra_model = cebra.CEBRA(
        model_architecture='offset10-model',
        time_offsets=10,
        max_iterations=max_iterations,
        batch_size=512,
        learning_rate=3e-4,
        temperature=1,
        output_dimension=output_dim,
        distance='cosine',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    # Fit the model using discrete labels
    # CEBRA expects discrete labels as integers
    cebra_model.fit(latent_vectors, discrete_labels)
    
    # Transform the data to get embeddings
    cebra_embeddings = cebra_model.transform(latent_vectors)
    
    print(f"CEBRA embeddings computed! Shape: {cebra_embeddings.shape}")
    
    return cebra_embeddings, cebra_model, label_encoder


def compute_cebra_embeddings_with_time(latent_vectors, object_names, output_dim=3, max_iterations=5000):
    """
    Compute CEBRA embeddings from latent vectors using time as a continuous variable.
    Time is represented as the sample index (assuming sequential frames).
    
    Parameters:
    - latent_vectors: numpy array of shape (n_samples, n_features)
    - object_names: numpy array of object names (for visualization only)
    - output_dim: dimensionality of the output embedding (default: 3)
    - max_iterations: maximum number of training iterations
    
    Returns:
    - cebra_embeddings: numpy array of shape (n_samples, output_dim)
    - cebra_model: trained CEBRA model
    - time_index: numpy array of time indices used for training
    """
    # Create time index based on sample order (assuming sequential frames)
    time_index = np.arange(latent_vectors.shape[0]).reshape(-1, 1).astype(np.float32)
    
    print(f"Training CEBRA model with TIME as continuous variable...")
    print(f"Input dimension: {latent_vectors.shape[1]}, Output dimension: {output_dim}")
    print(f"Number of samples: {latent_vectors.shape[0]}")
    print(f"Time range: {time_index.min()} to {time_index.max()}")
    
    # Initialize CEBRA model for continuous time
    cebra_model = cebra.CEBRA(
        model_architecture='offset10-model',
        time_offsets=10,
        max_iterations=max_iterations,
        batch_size=512,
        learning_rate=3e-4,
        temperature=1,
        output_dimension=output_dim,
        distance='cosine',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True
    )
    
    # Fit the model using time as continuous variable
    # CEBRA expects continuous labels as a 2D array (n_samples, n_features)
    cebra_model.fit(latent_vectors, time_index)
    
    # Transform the data to get embeddings
    cebra_embeddings = cebra_model.transform(latent_vectors)
    
    print(f"CEBRA embeddings with time computed! Shape: {cebra_embeddings.shape}")
    
    return cebra_embeddings, cebra_model, time_index


def plot_3Dcebra(cebra_embeddings, object_names):
    """
    Create a 3D scatter plot of CEBRA embeddings colored by object names.
    """
    cebra_df = pd.DataFrame(cebra_embeddings, columns=['CEBRA_1', 'CEBRA_2', 'CEBRA_3'])
    cebra_df['Object Name'] = object_names
    
    # Create a 3D scatter plot with Plotly Express
    fig = px.scatter_3d(
        cebra_df,
        x='CEBRA_1',
        y='CEBRA_2',
        z='CEBRA_3',
        color='Object Name',
        hover_name='Object Name',
        title="3D CEBRA Embedding of Latent Space Grouped by Object Names"
    )
    
    # Update axis labels
    fig.update_layout(
        height=1000,
        scene=dict(
            xaxis_title='CEBRA Dimension 1',
            yaxis_title='CEBRA Dimension 2',
            zaxis_title='CEBRA Dimension 3'
        )
    )
    
    fig.show()


def plot_2Dcebra(cebra_embeddings, object_names, dim1=0, dim2=1):
    """
    Create a 2D scatter plot of CEBRA embeddings colored by object names.
    
    Parameters:
    - cebra_embeddings: numpy array of CEBRA embeddings
    - object_names: numpy array of object names
    - dim1: first dimension to plot (default: 0)
    - dim2: second dimension to plot (default: 1)
    """
    cebra_df = pd.DataFrame(cebra_embeddings, columns=[f'CEBRA_{i+1}' for i in range(cebra_embeddings.shape[1])])
    cebra_df['Object Name'] = object_names
    
    # Unique object names for plotting
    unique_objects = np.unique(object_names)
    
    # Plot CEBRA results for each object
    plt.figure(figsize=(12, 8))
    for obj_name in unique_objects:
        subset = cebra_df[cebra_df['Object Name'] == obj_name]
        plt.scatter(subset[f'CEBRA_{dim1+1}'], subset[f'CEBRA_{dim2+1}'], label=obj_name, alpha=0.6)
    
    # Add plot legend and labels
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Object Names")
    plt.xlabel(f"CEBRA Dimension {dim1+1}")
    plt.ylabel(f"CEBRA Dimension {dim2+1}")
    plt.title("CEBRA Embedding of Latent Space Grouped by Object Names")
    plt.tight_layout()
    plt.show()


def plot_3Dcebra_time(cebra_embeddings, time_index, object_names=None):
    """
    Create a 3D scatter plot of CEBRA embeddings colored by time.
    
    Parameters:
    - cebra_embeddings: numpy array of CEBRA embeddings
    - time_index: numpy array of time indices
    - object_names: optional numpy array of object names for hover info
    """
    cebra_df = pd.DataFrame(cebra_embeddings, columns=['CEBRA_1', 'CEBRA_2', 'CEBRA_3'])
    cebra_df['Time'] = time_index.flatten()
    
    if object_names is not None:
        cebra_df['Object Name'] = object_names
        hover_name = 'Object Name'
    else:
        hover_name = None
    
    # Create a 3D scatter plot with Plotly Express, colored by time
    fig = px.scatter_3d(
        cebra_df,
        x='CEBRA_1',
        y='CEBRA_2',
        z='CEBRA_3',
        color='Time',
        hover_name=hover_name,
        color_continuous_scale='viridis',  # or 'plasma', 'magma', 'inferno'
        title="3D CEBRA Embedding of Latent Space Colored by Time"
    )
    
    # Update axis labels
    fig.update_layout(
        height=1000,
        scene=dict(
            xaxis_title='CEBRA Dimension 1',
            yaxis_title='CEBRA Dimension 2',
            zaxis_title='CEBRA Dimension 3'
        ),
        coloraxis_colorbar=dict(title="Time Index")
    )
    
    fig.show()


def plot_2Dcebra_time(cebra_embeddings, time_index, object_names=None, dim1=0, dim2=1):
    """
    Create a 2D scatter plot of CEBRA embeddings colored by time.
    
    Parameters:
    - cebra_embeddings: numpy array of CEBRA embeddings
    - time_index: numpy array of time indices
    - object_names: optional numpy array of object names for grouping
    - dim1: first dimension to plot (default: 0)
    - dim2: second dimension to plot (default: 1)
    """
    cebra_df = pd.DataFrame(cebra_embeddings, columns=[f'CEBRA_{i+1}' for i in range(cebra_embeddings.shape[1])])
    cebra_df['Time'] = time_index.flatten()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot colored by time
    scatter = ax.scatter(
        cebra_df[f'CEBRA_{dim1+1}'], 
        cebra_df[f'CEBRA_{dim2+1}'], 
        c=cebra_df['Time'], 
        cmap='viridis',  # or 'plasma', 'magma', 'inferno'
        alpha=0.6,
        s=20
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Index', rotation=270, labelpad=20)
    
    # Add labels and title
    ax.set_xlabel(f"CEBRA Dimension {dim1+1}")
    ax.set_ylabel(f"CEBRA Dimension {dim2+1}")
    ax.set_title("CEBRA Embedding of Latent Space Colored by Time")
    
    # Optionally overlay object names if provided
    if object_names is not None:
        unique_objects = np.unique(object_names)
        for obj_name in unique_objects:
            mask = object_names == obj_name
            if np.sum(mask) > 0:
                # Plot a marker at the centroid of each object cluster
                centroid_x = cebra_df.loc[mask, f'CEBRA_{dim1+1}'].mean()
                centroid_y = cebra_df.loc[mask, f'CEBRA_{dim2+1}'].mean()
                ax.annotate(obj_name, (centroid_x, centroid_y), 
                           fontsize=8, alpha=0.7, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


dataset_path=r'C:\Users\Siava\PycharmProjects\pythonProject\cvae_cebra\dataset\hand_object_data.pkl'

hand_train, hand_val, hand_test,obj_train, obj_val, obj_test ,obj_names_train, obj_names_val, obj_names_test=load_data(dataset_path)
print('Dataset is Loaded!')
print(obj_names_train)
train_loader, val_loader, test_loader=load_tensor_data(obj_train,hand_train,obj_val,hand_val,obj_test,hand_test)
print('DataLoader is Loaded!')

model_weight = r'C:\Users\Siava\PycharmProjects\pythonProject\cvae_cebra\src\cvae_weight.pth'
latent_dim = 32  # You can try different values (e.g., 32 or 128)
model = CVAE(latent_dim=latent_dim)

latent_vectors = compute_latent(model, test_loader,model_weight)
print('LatentVector is created!')
print(latent_vectors.shape)


# Perform PCA on latent space
pca = PCA(n_components=latent_vectors.shape[1])  # Keep all components
pca_results = pca.fit_transform(latent_vectors)

print('PCA is created!')
print(pca_results.shape)

plot_pca_explained_variance_ratio(pca)

plot_3Dpca(pca_results,obj_names_test,7,8,9)

plot_2Dpca(pca_results,obj_names_test,7,8)

# Perform CEBRA on latent space
print('\n' + '='*50)
print('Computing CEBRA embeddings...')
print('='*50)

# Compute CEBRA embeddings (3D for visualization)
cebra_embeddings, cebra_model, label_encoder = compute_cebra_embeddings(
    latent_vectors, 
    obj_names_test, 
    output_dim=3, 
    max_iterations=5000
)

print('CEBRA embeddings created!')
print(cebra_embeddings.shape)

# Visualize CEBRA embeddings (with object labels)
plot_3Dcebra(cebra_embeddings, obj_names_test)
plot_2Dcebra(cebra_embeddings, obj_names_test, dim1=0, dim2=1)

# Alternative: Use CEBRA's built-in plotting function
# Convert object names to numeric labels for coloring
label_encoder_plot = LabelEncoder()
numeric_labels = label_encoder_plot.fit_transform(obj_names_test)
print("\nUsing CEBRA's built-in plotting function...")
cebra.plot_embedding(
    cebra_embeddings, 
    embedding_labels=numeric_labels, 
    cmap='viridis',
    markersize=5,
    alpha=0.6,
    title="CEBRA Embedding (Built-in Plot)"
)

# Perform CEBRA with TIME as continuous variable
print('\n' + '='*50)
print('Computing CEBRA embeddings with TIME factor...')
print('='*50)

# Compute CEBRA embeddings using time as continuous variable
cebra_embeddings_time, cebra_model_time, time_index = compute_cebra_embeddings_with_time(
    latent_vectors, 
    obj_names_test, 
    output_dim=3, 
    max_iterations=5000
)

print('CEBRA embeddings with time created!')
print(cebra_embeddings_time.shape)

# Visualize CEBRA embeddings colored by time
plot_3Dcebra_time(cebra_embeddings_time, time_index, object_names=obj_names_test)
plot_2Dcebra_time(cebra_embeddings_time, time_index, object_names=obj_names_test, dim1=0, dim2=1)

# Alternative: Use CEBRA's built-in plotting function with time
print("\nUsing CEBRA's built-in plotting function with time...")
cebra.plot_embedding(
    cebra_embeddings_time, 
    embedding_labels=time_index.flatten(), 
    cmap='magma',
    markersize=5,
    alpha=0.6,
    title="CEBRA Embedding with Time (Built-in Plot)"
)