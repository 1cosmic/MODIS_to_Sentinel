# %matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize


def plot_data(data, mask=None):
    data = resize(data, (1000, 1000), preserve_range=True, anti_aliasing=True)
    plt.style.use('dark_background')
    if mask is not None:
        mask = resize(mask, (1000, 1000), preserve_range=True, anti_aliasing=False)
        masked_data = data * (mask == 1)
        plt.imshow(masked_data, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    plt.show()


def plot_confusion(matrix):
    # TODO: setup plot
    plt.clf()

    matrix = matrix / matrix.sum(axis=1).reshape(-1, 1)  # normalise it

    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), z in np.ndenumerate(matrix):
        if z > 0.5:
            c = 'white'
            f = 'bold'
        else:
            c = 'black'
            f = 'normal'
        plt.text(
            j,
            i,
            f"{z:.2f}",
            ha="center",
            va="center",
            color=c,
            fontsize=10,
            fontweight=f,
        )
    plt.clim(0, 1)
    plt.colorbar()
    # plt.show()


def spyder_eye(datasets, ml_sets):
    plt.clf()
    if len(datasets) != ml_sets:
        Exception("Lens of datasets != ml_sets")

    fig, axes = plt.subplots(2, len(datasets), figsize=(4 * len(datasets), 8))

    for i in range(len(datasets)):
        # draw 1st line: conf_matrix
        ds = datasets[i]
        heatmap = ml_sets[i]['cf_matrix']
        heatmap = heatmap / heatmap.sum(axis=1).reshape(-1, 1)  # normalise it

        percent = ds.get('percent', '')
        mask_mode = ds.get('mask_mode', '')
        layer_mode = ds.get('layer_mode', '')
        rows = ml_sets[i].get('r', '')
        # resize_val = ml_sets[i].get('resize', '')
        im = axes[0, i].imshow(heatmap, cmap="Blues")
        axes[0, i].set_title("Confusion Matrix")
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("True")
        for (r, c), z in np.ndenumerate(heatmap):
            if z > 0.3:
                axes[0, i].text(
                    c,
                    r,
                    f"{z:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        im.set_clim(0, 1)
        # plt.colorbar(im, ax=axes[0, i])

        hp = ds.get('homogen_percent', '')
        radius = ds.get('r', '')

        f1_score = ds.get('f1_score', None)
        if f1_score is not None:
            f1_score_str = f"{round(f1_score  * 100, 1)}"
        else:
            f1_score_str = ""

        title = f"f1: {f1_score_str}%, mask: {mask_mode}, layer: {layer_mode},\nsize: {percent*100:.02f}% | std: {hp:.02f} or r={radius}"
        axes[0, i].set_title(title)
        plt.colorbar(im, ax=axes[0, i])

        # draw 2nd line: hist of classes

        classes = range(ds['classes'].size)
        axes[1, i].bar(classes, ds['classes'])
        axes[1, i].set_xlabel('Class')
        axes[1, i].set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def draw_median_sign(data):
    data = np.array(data)
    def normalize_rgb(values):
        """Normalize RGB values to [0, 1] range for matplotlib"""

        # min_val = np.min(values)
        # max_val = np.max(values)
        # if max_val > min_val:
        #     return (values - min_val) / (max_val - min_val)

        # values[values > 2000] = 2000
        values[values > 1500] = 1500
        values = values / 1500
        return values

    n_classes = len(data)
    n_samples = 3  # 3 RGB sets per class
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for class_idx in range(n_classes):
        rgb_values = []
        for sample in range(n_samples):
            base_idx = sample * 4
            b = data[class_idx, base_idx]
            g = data[class_idx, base_idx + 1]
            r = data[class_idx, base_idx + 3]
            rgb_values.append([r, g, b])  # matplotlib uses RGB order
        
        rgb_values = np.array(rgb_values)
        
        # Normalize each color channel separately for better visualization
        rgb_normalized = np.zeros_like(rgb_values, dtype=float)
        for channel in range(3):
            rgb_normalized[:, channel] = normalize_rgb(rgb_values[:, channel])
            # print(rgb_normalized[:, channel])
        
        # Create color patches
        for sample_idx, rgb in enumerate(rgb_normalized):
            # Create a small rectangle for each color sample
            rect = plt.Rectangle((sample_idx * 0.3, 0), 0.275, 0.8, 
                            facecolor=rgb, edgecolor='black', linewidth=1)
            axes[class_idx].add_patch(rect)
        
        # Set subplot properties
        axes[class_idx].set_xlim(0, 1)
        axes[class_idx].set_ylim(0, 1)
        axes[class_idx].set_aspect('equal')
        axes[class_idx].axis('off')
        axes[class_idx].set_title(f'Class {class_idx + 1}', fontsize=12, fontweight='bold', verticalalignment='top')
        
        # Add RGB values as text
        for sample_idx, rgb in enumerate(rgb_values):
            rgb_int = rgb.astype(int)
            axes[class_idx].text(sample_idx * 0.3 + 0.05, 0.4, 
                            f'R:{rgb_int[0]}\nG:{rgb_int[1]}\nB:{rgb_int[2]}', 
                            fontsize=8, verticalalignment='center', horizontalalignment='center', fontweight='bold')

    plt.suptitle('Median class for each class (RGB, NIR skipped)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def draw_hist_cosine_dists(data):
    counts = [x[0] for x in data]
    bins = [x[1] for x in data]

    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Ensure we have exactly 6 classes
    assert len(data) == 6, print("Expected data for 6 classes, skip of histogram")
    
    for class_idx in range(6):
        counts, bins = data[class_idx]
        
        # Calculate bin centers for bar plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        
        # Plot histogram
        axes[class_idx].bar(bin_centers, counts, 
                          width=bin_width * 0.9,  # Slight gap between bars
                          edgecolor='black', 
                          linewidth=0.5,
                          alpha=0.7,
                          color=f'C{class_idx}')  # Different color for each class
        
        # Customize subplot
        axes[class_idx].set_title(f'Class {class_idx + 1}', fontsize=12, fontweight='bold')
        axes[class_idx].set_xlabel('Cosine Distance', fontsize=10)
        axes[class_idx].set_ylabel('Frequency', fontsize=10)
        axes[class_idx].grid(True, alpha=0.3, linestyle='--')
        
        # Set x-axis limits from 0 to 1
        axes[class_idx].set_xlim(0, 1)
        
        # Add some statistics
        total_samples = np.sum(counts)
        mean_dist = np.sum(bin_centers * counts) / total_samples
        axes[class_idx].text(0.05, 0.95, f'n={total_samples}\nμ={mean_dist:.3f}', 
                           transform=axes[class_idx].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    plt.suptitle('Cosine Distance Distributions by Class', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()