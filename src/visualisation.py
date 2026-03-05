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
        if z > 0.3:
            plt.text(
                j,
                i,
                f"{z:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
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