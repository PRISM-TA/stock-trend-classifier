import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MaxNLocator
import glob
import re


def plot_consolidated_loss_chart(json_files, model_name, ticker=None, feature_set=None, save_path='loss_charts'):
    """
    Creates a consolidated chart of all training sessions for a specific model/ticker/feature set
    combination and saves it as a PNG.
    
    Args:
        json_files: List of JSON file paths with loss data
        model_name: Name of the model
        ticker: Stock ticker symbol
        feature_set: Feature set object or name
        save_path: Directory to save the chart
    
    Returns:
        The path to the saved chart file
    """
    # Ensure save_path is a string
    if not isinstance(save_path, str):
        print(f"Warning: save_path is not a string. Using default 'loss_charts' instead of {save_path}")
        save_path = 'loss_charts'
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Get the feature set name
    feature_set_name = None
    if feature_set:
        if hasattr(feature_set, 'set_name'):
            feature_set_name = feature_set.set_name
        else:
            # If feature_set is a string
            feature_set_name = str(feature_set)
    
    # Create base filename for the chart - use full names for consistency
    filename_parts = [model_name]
    if ticker:
        filename_parts.append(str(ticker))  # Ensure ticker is a string
    if feature_set_name:
        # Replace characters that might cause issues in filenames
        safe_feature_name = feature_set_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename_parts.append(safe_feature_name)
    
    base_pattern = '_'.join(filename_parts)
    
    if not json_files:
        print(f"No loss files found for {base_pattern}")
        return None
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Use only two colors: one for training loss, one for validation loss
    train_color = 'blue'
    val_color = 'red'
    
    # Plot each training session
    for i, file_path in enumerate(sorted(json_files)):
        try:
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    
                    # Extract loss data
                    train_losses = data.get('train_loss', [])
                    val_losses = data.get('val_loss', [])
                    
                    # Get the window number for labeling if available
                    window_num = data.get('window_num', i)
                    
                    # Plot training loss - always using the same color (blue)
                    plt.plot(train_losses, color=train_color, linestyle='-', alpha=0.7, 
                             label=f'Train Window' if i == 0 else None)
                    
                    # Plot validation loss if available - always using the same color (red)
                    if val_losses:
                        plt.plot(val_losses, color=val_color, linestyle='--', alpha=0.7, 
                                 label=f'Val Window' if i == 0 else None)
                    
                except json.JSONDecodeError:
                    print(f"Error loading JSON from {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Set up plot aesthetics
    title_parts = [f'Loss History: {model_name}']
    if ticker:
        title_parts.append(str(ticker))
    if feature_set_name:
        title_parts.append(feature_set_name)
    
    plt.title(' - '.join(title_parts))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Add a legend to explain the colors
    plt.legend(loc='upper right')
    
    # Add a single line of explanatory text
    plt.figtext(0.5, 0.01, 
                f'Chart shows {len(json_files)} training windows. Blue: Training Loss | Red: Validation Loss', 
                ha='center', fontsize=10)
    
    # Force integer epochs on x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add a tight layout - adjusted to leave room for text at the bottom
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the chart
    chart_filename = f"{base_pattern}_consolidated.png"
    chart_path = os.path.join(save_path, chart_filename)
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    print(f"Consolidated loss chart saved to {chart_path}")
    return chart_path


def generate_all_consolidated_charts():
    """
    Generate consolidated charts for all unique model/ticker/feature set combinations
    found in the loss_record directory.
    """
    if not os.path.exists('loss_record'):
        print("No loss_record directory found")
        return
    
    # Track files by their combinations
    combinations = {}
    
    # Process all JSON files
    for filename in os.listdir('loss_record'):
        if not filename.endswith('.json'):
            continue
        
        # Extract model name, ticker, and feature set info (excluding window part)
        parts = filename.split('_')
        
        if len(parts) >= 2:  # At minimum model_something.json
            model_name = parts[0]
            
            # Extract ticker if present (typically the second part)
            ticker = parts[1] if len(parts) > 1 else None
            
            # Extract feature set if present
            feature_set_part = None
            
            # Check if there's a window part to exclude
            window_index = -1
            for i, part in enumerate(parts):
                if part.startswith('window'):
                    window_index = i
                    break
            
            # Extract feature set part (anything between ticker and window)
            if window_index > 2:  # If we have parts between ticker and window
                feature_set_part = '_'.join(parts[2:window_index])
            elif len(parts) > 2 and window_index == -1:  # No window, but we have more parts
                feature_set_part = '_'.join(parts[2:])
            
            # Use the combined key to group files
            key = (model_name, ticker, feature_set_part)
            
            if key not in combinations:
                combinations[key] = []
            
            combinations[key].append(os.path.join('loss_record', filename))
    
    # Generate charts for each combination
    for (model_name, ticker, feature_set_part), files in combinations.items():
        print(f"Generating chart for {model_name}, {ticker}, {feature_set_part}")
        
        # Pass the files, model name, ticker, and feature set to the plotting function
        plot_consolidated_loss_chart(
            json_files=files,
            model_name=model_name,
            ticker=ticker,
            feature_set=feature_set_part,
            save_path='loss_charts'
        )
    
    print(f"Generated {len(combinations)} consolidated charts")


# If run as a script
if __name__ == "__main__":
    generate_all_consolidated_charts()