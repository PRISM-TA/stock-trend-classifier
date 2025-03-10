import streamlit as st
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_history(filename=None, model_name=None, ticker=None, feature_set=None):
    """
    Plot loss history from a JSON file or search for a specific combination
    
    Args:
        filename: Path to a specific loss history file
        model_name: Model name to search for
        ticker: Ticker symbol to search for
        feature_set: Feature set name or acronym to search for
    """
    if filename is None:
        # Try to find a matching file based on criteria
        if not os.path.exists('loss_record'):
            st.error("No loss_record directory found")
            return None
            
        loss_files = os.listdir('loss_record')
        if not loss_files:
            st.error("No loss records found")
            return None
        
        # Filter files based on provided criteria
        if model_name:
            loss_files = [f for f in loss_files if model_name in f]
        if ticker:
            loss_files = [f for f in loss_files if f"_{ticker}_" in f or f"{ticker}_" in f]
        if feature_set:
            # We'll just search for the feature set string directly
            loss_files = [f for f in loss_files if feature_set in f]
        
        if not loss_files:
            criteria = []
            if model_name: criteria.append(f"model={model_name}")
            if ticker: criteria.append(f"ticker={ticker}")
            if feature_set: criteria.append(f"feature_set={feature_set}")
            
            st.error(f"No loss records found matching criteria: {', '.join(criteria)}")
            return None
            
        # Sort by timestamp (newest first)
        loss_files.sort(reverse=True)
        filename = os.path.join('loss_record', loss_files[0])
    
    # Load the loss data
    try:
        with open(filename, 'r') as f:
            loss_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading loss data: {e}")
        return None
    
    # Extract data
    train_loss = loss_data['train_loss']
    val_loss = loss_data['val_loss'] if loss_data.get('val_loss') else None
    model_name = loss_data['model_name']
    ticker = loss_data.get('ticker', 'Unknown')
    feature_set_name = loss_data.get('feature_set', 'Unknown')
    
    # Get hyperparameters for the title
    hyperparams = loss_data['hyperparameters']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot data
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'b-', linewidth=1.5, label='Training Loss')
    
    if val_loss:
        ax.plot(epochs, val_loss, 'r-', linewidth=1.5, label='Validation Loss')
    
    # Set title and labels
    title = f'Training Loss: {model_name}'
    if ticker != 'Unknown':
        title += f' - {ticker}'
    if feature_set_name != 'Unknown':
        title += f' - {feature_set_name}'
        
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Add a text box with hyperparameters
    params_text = ""
    for k, v in hyperparams.items():
        if v is not None:  # Only include non-None parameters
            params_text += f"{k}: {v}\n"
    
    # Position the text box in the upper right corner
    if params_text:
        plt.figtext(0.75, 0.85, params_text.strip(), 
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                    fontsize=9)
    
    plt.tight_layout()
    
    return fig

def loss_visualizer_app():
    """Streamlit app for visualizing training loss history"""
    st.title("Training Loss Visualizer")
    
    # Get available loss files
    if not os.path.exists('loss_record'):
        st.warning("No loss_record directory found. Train a model first to generate loss records.")
        return
        
    loss_files = os.listdir('loss_record')
    if not loss_files:
        st.warning("No loss records found in the loss_record directory.")
        return
    
    # Extract unique model names, tickers, and feature sets from filenames
    model_names = sorted(list(set([f.split('_')[0] for f in loss_files if '_' in f])))
    
    # For tickers, look at the second part of the filename
    tickers = []
    for f in loss_files:
        parts = f.split('_')
        if len(parts) >= 2:
            tickers.append(parts[1])
    tickers = sorted(list(set(tickers)))
    
    # For feature sets, it's more complex as they can be different formats
    # We'll extract the third part if it exists
    feature_sets = []
    for f in loss_files:
        parts = f.split('_')
        if len(parts) >= 3:
            # The feature set is the third part, but remove any timestamp
            feature_part = parts[2]
            # If the part contains numbers only, it might be a timestamp
            if not feature_part.isdigit():
                feature_sets.append(feature_part)
    feature_sets = sorted(list(set(feature_sets)))
    
    # Create filter selectors
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_model = st.selectbox("Select Model", ["All"] + model_names)
    
    with col2:
        selected_ticker = st.selectbox("Select Ticker", ["All"] + tickers)
        
    with col3:
        selected_feature = st.selectbox("Feature Set", ["All"] + feature_sets)
    
    # Filter loss files based on selections
    filtered_files = loss_files
    
    if selected_model != "All":
        filtered_files = [f for f in filtered_files if f.startswith(f"{selected_model}_")]
        
    if selected_ticker != "All":
        filtered_files = [f for f in filtered_files if f.split('_')[1] == selected_ticker]
        
    if selected_feature != "All":
        filtered_files = [f for f in filtered_files if selected_feature in f]
    
    # Format function for displaying files
    def format_file_display(filename):
        parts = filename.split('_')
        # Extract timestamp from the last part
        timestamp = parts[-1].split('.')[0]
        
        # Format the display string
        display = f"{parts[0]}"  # Model name
        
        if len(parts) > 1:
            display += f" - {parts[1]}"  # Ticker
            
        if len(parts) > 2:
            # Feature set (might be the third part or could include more)
            feature_part = parts[2]
            display += f" - {feature_part}"
            
        # Add timestamp
        if timestamp.isdigit() and len(timestamp) >= 8:
            date = timestamp[:8]
            time = timestamp[9:] if len(timestamp) > 8 else ""
            display += f" ({date} {time})"
            
        return display
    
    # Display the filtered files for selection
    if filtered_files:
        # Sort by timestamp (newest first)
        filtered_files.sort(reverse=True)
        
        selected_file = st.selectbox(
            "Select Training Run", 
            filtered_files,
            format_func=format_file_display
        )
        
        # Plot the selected loss history
        file_path = os.path.join('loss_record', selected_file)
        fig = plot_loss_history(file_path)
        
        if fig:
            st.pyplot(fig)
        else:
            st.error("Error generating the loss plot.")
    else:
        st.warning("No training runs found matching the selected criteria.")

if __name__ == "__main__":
    loss_visualizer_app()