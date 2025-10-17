import os, glob, torch, pandas as pd
from datetime import datetime
from tqdm import tqdm
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from simulacra.iterable_csv_dataset import PythaeIterableDataset, DNAmDataset
from simulacra.log import _log

def find_model_path(model_dir):
    """Find the actual model path within the timestamped subdirectory."""
    if not os.path.exists(model_dir):
        return None
    
    # Look for VAE_training_* directories
    training_dirs = glob.glob(os.path.join(model_dir, "VAE_training_*"))
    if not training_dirs:
        return None
    
    # Get the most recent training directory
    latest_training_dir = max(training_dirs, key=os.path.getctime)
    final_model_path = os.path.join(latest_training_dir, "final_model")
    
    if os.path.exists(final_model_path):
        return final_model_path
    return None

def generate_embeddings_with_pythae(accession: str, n_dnam_cols: int = 100, latent_dim: int = 512, epochs: int = 20):
    """
    Generate embeddings using Pythae VAE on GSE data.
    
    Args:
        accession (str): GSE accession number (e.g., 'GSE42861')
        n_dnam_cols (int): Number of DNAm columns to use for training
        latent_dim (int): Latent dimension for VAE
        epochs (int): Number of training epochs
    
    Returns:
        str: Path to the generated embeddings CSV file
    """
    _log(f"Starting Pythae embedding generation for {accession}...")
    
    # Create local data directory
    data_dir = "2_poc_simulacra"
    os.makedirs(data_dir, exist_ok=True)
    
    # Define paths - use the actual file names that exist
    embeddings_path = os.path.join(data_dir, f"{accession}_embeddings.csv")
    model_dir = os.path.join(data_dir, f"{accession}_vae_model")
    dnam_path = os.path.join(data_dir, "dnam.csv")
    metadata_path = os.path.join(data_dir, "metadata.csv")
    
    # Check if embeddings already exist
    if os.path.exists(embeddings_path):
        _log(f"Loading existing embeddings from {embeddings_path}...")
        embeddings_df = pd.read_csv(embeddings_path, index_col=0)
        _log(f"Loaded embeddings: {embeddings_df.shape[0]} rows, {embeddings_df.shape[1]} columns")
        return embeddings_path
    
    # Check if model already exists
    model_path = find_model_path(model_dir)
    if model_path:
        _log(f"Loading existing VAE model from {model_path}...")
        model = VAE.load_from_folder(model_path)
        _log("VAE model loaded successfully")
    else:
        _log("Training new VAE model (LONG OPERATION)...")
        train_start = datetime.now()
        
        # Create train/validation split
        ds = {}
        ds['train'], ds['val'] = DNAmDataset.split_dataset(
            split_points=0.8,
            dnam_path=dnam_path,
            metadata_path=metadata_path,
            n_dnam_cols=n_dnam_cols,
            target_metadata_col="disease",
            target_metadata_type="categorical",
            shuffle=True,
            seed=42,
            on_unmatched='warn',
            join_on='Unnamed: 0',
        )
        
        _log(f"Dataset split created - Train: {len(ds['train'])}, Val: {len(ds['val'])}")
        
        # Create VAE model
        model_config = VAEConfig(
            input_dim=(n_dnam_cols,),
            latent_dim=latent_dim
        )
        
        # Create training config
        training_config = BaseTrainerConfig(
            output_dir=model_dir,
            learning_rate=1e-3,
            per_device_train_batch_size=512,
            per_device_eval_batch_size=512,
            num_epochs=epochs,
            keep_best_on_train=True
        )
        
        # Create training pipeline
        pipeline = TrainingPipeline(
            model=VAE(model_config=model_config),
            training_config=training_config
        )
        
        # Train the model
        pipeline(
            train_data=ds['train'],
            eval_data=ds['val']
        )
        
        train_duration = datetime.now() - train_start
        _log(f"VAE training completed in {train_duration.total_seconds():.2f} seconds")
        
        # Find and load the trained model
        model_path = find_model_path(model_dir)
        if model_path:
            model = VAE.load_from_folder(model_path)
            _log("VAE model loaded successfully")
        else:
            raise FileNotFoundError(f"Could not find trained model in {model_dir}")
    
    # Generate embeddings for all data
    _log("Generating embeddings for all data (LONG OPERATION)...")
    embed_start = datetime.now()
    
    # Create dataset for all data
    all_ds = PythaeIterableDataset(
        dnam_path=dnam_path,
        metadata_path=metadata_path,
        n_dnam_cols=n_dnam_cols,
        target_metadata_col="disease",
        target_metadata_type="categorical",
        join_on="Unnamed: 0",
    )
    
    # Get the list of indices from metadata
    index_list = list(all_ds.metadata.index)
    
    # Generate embeddings
    model.eval()
    buffer = []
    with torch.no_grad():
        with open(embeddings_path, 'w') as fout:
            header = ','.join([f'emb_{i}' for i in range(model.latent_dim)])
            fout.write(f'id,{header}\n')
            
            for idx, (X, _) in tqdm(enumerate(all_ds), desc="Generating embeddings"):
                X_tensor = X.to(dtype=torch.float32).unsqueeze(0)
                embedding = model.encoder(X_tensor)['embedding'].cpu().numpy().squeeze()
                
                # Use the original index from metadata
                row_id = index_list[idx] if idx < len(index_list) else idx
                row = ','.join([str(x) for x in embedding])
                buffer.append(f'{row_id},{row}\n')
                
                # Write buffer periodically
                if len(buffer) >= 64:
                    fout.writelines(buffer)
                    buffer.clear()
            
            # Write remaining buffer
            if buffer:
                fout.writelines(buffer)
    
    embed_duration = datetime.now() - embed_start
    _log(f"Embedding generation completed in {embed_duration.total_seconds():.2f} seconds")
    
    # Load and display the embeddings
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    _log(f"Generated embeddings: {embeddings_df.shape[0]} rows, {embeddings_df.shape[1]} columns")
    
    return embeddings_path

def create_embeddings_with_target(accession: str, target_col: str = "disease"):
    """
    Create a new dataframe combining embeddings with target column.
    
    Args:
        accession (str): GSE accession number (e.g., 'GSE42861')
        target_col (str): Name of the target column from metadata
    
    Returns:
        pd.DataFrame: Combined dataframe with target as first column, followed by embeddings
    """
    _log(f"Creating combined dataframe for {accession} with target column '{target_col}'...")
    
    # Define paths
    data_dir = "2_poc_simulacra"
    embeddings_path = os.path.join(data_dir, f"{accession}_embeddings.csv")
    metadata_path = os.path.join(data_dir, "metadata.csv")
    combined_path = os.path.join(data_dir, f"{accession}_embeddings_with_target.csv")
    
    # Check if combined file already exists
    if os.path.exists(combined_path):
        _log(f"Loading existing combined dataframe from {combined_path}...")
        combined_df = pd.read_csv(combined_path, index_col=0)
        _log(f"Loaded combined dataframe: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        return combined_df
    
    # Load embeddings and metadata
    _log("Loading embeddings and metadata...")
    embeddings_df = pd.read_csv(embeddings_path, index_col=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    
    _log(f"Embeddings shape: {embeddings_df.shape}")
    _log(f"Metadata shape: {metadata_df.shape}")
    
    # Check if target column exists in metadata
    if target_col not in metadata_df.columns:
        available_cols = list(metadata_df.columns)
        raise ValueError(f"Target column '{target_col}' not found in metadata. Available columns: {available_cols}")
    
    # Join on index to combine the dataframes
    _log("Combining embeddings with target column...")
    combined_df = pd.concat([metadata_df[[target_col]], embeddings_df], axis=1, join='inner')
    
    _log(f"Combined dataframe shape: {combined_df.shape}")
    _log(f"Target column '{target_col}' unique values: {combined_df[target_col].unique()}")
    
    # Save the combined dataframe
    _log(f"Saving combined dataframe to {combined_path}...")
    combined_df.to_csv(combined_path)
    _log("Combined dataframe saved successfully")
    
    return combined_df