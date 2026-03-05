# For preparation of dataset

from time import time
import gc
import os
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter, sobel
from osgeo import gdal

from utils import init, load_tif, cut_tif_by, save_tif, DEFAULT_PATH, GEO_DATA

import itertools
from tqdm import tqdm

init()


def load_resized_data_labels(signs, labels, force=False, resize='by_label', verbose=True):
    px_size = 230 if resize == 'by_label' else 10
    if verbose:
        print("Loading & preparing image data...")

    if verbose:
        print("\nCropping & loading labels by 1st image:")
    ref_image = load_tif(signs[0], only_first=True, verbose=verbose)
    cropped_labels = []
    for label in labels:
        name = os.path.basename(label)
        out = DEFAULT_PATH['cropped_labels'] + name
        if not os.path.exists(out) or force:
            cut_tif_by(label, ref_image, out, resize=px_size, mode='nearest', verbose=verbose)
        else:
            if verbose:
                print("Loading from cache...")
        cropped_labels.append(load_tif(out, only_first=True, verbose=verbose))
        
    if verbose:
        print("Crop labels is done.")

    # Resizing by: label/image.
    if resize == 'by_label':
        if verbose:
            print("\nResizing images by 1st label:")
        sources = signs
        by = cropped_labels[0]
        resize_path = 'resized_images'

    elif resize == 'by_sign':
        if verbose:
            print("\nResizing labels by 1st image:")
        sources = labels
        by = load_tif(signs[0], only_first=True, verbose=verbose)
        resize_path = 'resized_labels'
    else:
        raise ValueError("resize can be only [by_sign, by_label]")

    resized = []
    switch_mode = 'bilinear' if resize == 'by_sign' else 'nearest'
    for src in sources:
        name = os.path.basename(src)
        out = DEFAULT_PATH[resize_path] + name
        if not os.path.exists(out) or force:
            cut_tif_by(src, by, out, resize=px_size, mode=switch_mode, verbose=verbose)
        else:
            if verbose:
                print("Loading from cache...")
        resized.append(load_tif(out, only_first=True, verbose=verbose))
    if verbose:
        print(f"Resizing {resize} is done.")

    if verbose:
        print("\nData and labels prepared.\n")
    if resize == 'by_label':
        return resized, cropped_labels
    elif resize == 'by_sign':
        loaded_signs = [load_tif(s, verbose=verbose, only_first=True) for s in signs]
        return loaded_signs, resized
    else:
        return [], []


def create_texture_layer(r, nir, out=None, resize_by=None, force=False, order=None, verbose=False):
    if out is None:
        suffix = f'{order}_' if order is not None else ''
        out = DEFAULT_PATH['processing'] + f'texture_{suffix}' + os.path.basename(r)

    res = None
    if os.path.exists(out) and not force:
        print("Finded texture layer. Load it.")
        res = load_tif(out, only_first=True, verbose=verbose)

    else:
        r_obj = load_tif(r, only_first=True, verbose=verbose)
        nir_obj = load_tif(nir, only_first=True, verbose=verbose)
        print("Retyping...")
        r = r_obj['array'].astype(np.float32)
        nir = nir_obj['array'].astype(np.float32)
        del r_obj['array'], nir_obj
        gc.collect()

        Mpx = 1000000
        chunk = 10 * 1024^2
        N = r.size / Mpx  # r and nir have same shape
        size = r.shape

        print("NDVI…")
        t0 = time()
        homogeneous_array = (r - nir) / (r + nir + 1e-6)
        dt = time() - t0
        print(f"NDVI done: {dt:.3f}s   | per Mpx: {dt/N:.3f}s")

        t1 = time()
        gx = sobel(homogeneous_array, axis=1)
        dt = time() - t1
        print(f"gx done: {dt:.3f}s     | per Mpx: {dt/N:.3f}s")

        t2 = time()
        gy = sobel(homogeneous_array, axis=0)
        dt = time() - t2
        print(f"gy done: {dt:.3f}s     | per Mpx: {dt/N:.3f}s")

        t3 = time()
        gx = gx.ravel(); gy = gy.ravel()
        homogeneous_array = np.zeros_like(gx)
        for i in tqdm(range(0, gy.size, chunk)):
            homogeneous_array[i:i+chunk] = np.hypot(gx[i:i+chunk], gy[i:i+chunk])
        homogeneous_array = homogeneous_array.reshape(size)
        dt = time() - t3
        del gx, gy
        gc.collect()
        print(f"hypot done: {dt:.3f}s  | per Mpx: {dt/N:.3f}s")

        t4 = time()
        homogeneous_array = uniform_filter(homogeneous_array, size=23)
        dt = time() - t4
        print(f"uniform_filter done: {dt:.3f}s | per Mpx: {dt/N:.3f}s")
        total = time() - t0
        print(f"total: {total:.3f}s | total per Mpx: {total/N:.3f}s")

        r_obj['array'] = homogeneous_array
        save_tif(r_obj, out, dtype=gdal.GDT_Float32, verbose=verbose)
        res = r_obj

    if resize_by is not None:
        resize_by = load_tif(resize_by, only_first=True)
        print("Start resizing of homoheneous layer.")
        resized_out = DEFAULT_PATH['resized_layers'] + 'homogeneous_layer.tif'
        if not os.path.exists(resized_out) or force:
            cut_tif_by(out, resize_by, resized_out, resize=True, mode='bilinear',verbose=verbose)
        res = load_tif(resized_out, only_first=True)

    return res


def create_homogeneous_layer(sentinel_red, sentinel_nir, modis_label, resize='by_label', force=False, order=None, verbose=True):

    # modis_path = os.path.basename(modis_label['path'])
    sentinel_red_path = os.path.basename(sentinel_red)
    suffix = f'{order}_' if order is not None else ''
    out = DEFAULT_PATH['resized_layers'] + f'homogeneous_{suffix}{sentinel_red_path}'

    if verbose:
        print("\nStart of create the homogeneous layer by STD...")

    if os.path.exists(out) and not force:
        if verbose:
            print("Loading cached std layer from", out)
        return load_tif(out, only_first=True, verbose=verbose)

    if verbose:
        print("Loading red & nir & labels for NDVI...")
    sentinel_red = load_tif(sentinel_red, only_first=True, verbose=verbose)
    sentinel_nir = load_tif(sentinel_nir, only_first=True, verbose=verbose)

    r = sentinel_red["array"].astype(np.float32).ravel()
    n = sentinel_nir["array"].astype(np.float32).ravel()

    if verbose:
        print("Creating MODIS index layer and resizing to Sentinel...")
        

    if verbose:
        print("create idx for group std on Sentinel-NDVI (10m) by MODIS px (230m)")

    cut_tif_by(modis_label['path'], sentinel_nir, out, resize=230, verbose=verbose)
    modis_label = load_tif(out, only_first=True, verbose=verbose)
    l = modis_label['array']
    shape = l.shape
    idx = np.arange(l.size, dtype=np.uint32)

    if verbose:
        print("Convert 230m idx-layer to 10m, grouped like 230m")

    modis_label['array'] = idx.reshape(l.shape)
    save_tif(modis_label, out, dtype=gdal.GDT_UInt32, verbose=verbose)
    cut_tif_by(out, sentinel_nir, out, resize=10, verbose=verbose)
    idx = load_tif(out, only_first=True, verbose=verbose)['array'].ravel()

    if verbose:
        print("Calculating NDVI layer...")
    # NDVI = np.zeros_like(r, dtype=np.float32)
    NDVI = (n - r) / (r + n + 1e-6)
    del r, n, l
    gc.collect()
    print('NDVI min max shape:', NDVI.min(), NDVI.max(), NDVI.shape)
    
    if verbose:
        print("Preparing data for std calculation...")
    count = np.bincount(idx)

    # std = sqrt( sum(x_i^2)/n - mean_i^2 )
    if verbose:
        print("Calculating std layer...")
    sum = np.bincount(idx, weights=NDVI).astype(np.float64, copy=False)
    sum_squares = np.bincount(idx, weights=NDVI**2).astype(np.float64, copy=False)
    del NDVI, idx; gc.collect()
    std = sum_squares / count - (sum / count)**2
    del sum, sum_squares; gc.collect()
    std = np.sqrt(std)
    std[std > 1] = 0
    std = std.reshape(shape)

    if verbose:
        print(f"Std statistic: min: {std.min()} | max: {std.max()} | mean: {std.mean()} | shape: {std.shape}")
        print("Saving std layer...")
    
    modis_label['array'] = std.astype(np.float32)
    save_tif(modis_label, out, dtype=gdal.GDT_Float32, verbose=verbose)
    
    # If MODIS px to Sentinel, resize grouped std to it.
    if resize == 'by_sign':
        cut_tif_by(out, sentinel_nir, out, resize=10, mode='bilinear', verbose=verbose)
        modis_label = load_tif(out, only_first=True, verbose=verbose)
        print("Resizing done.")

    if verbose:
        print("Std layer created and saved to", out)
    return modis_label


def create_mask(label: dict, image: dict, percent: float = 0.01, r=1,
                mode:str = 'random', save_mask=False,
                homogen_layer=None, homogen_percent=0.1, stratify=False, verbose=True) -> np.ndarray:

    if verbose:
        print("Creating mask...")
    image_array = image['array']
    label_array = label['array']

    label_array[image_array <= 0] = 0              # work with ONLY FILLED pixels
    mask = np.zeros_like(label_array, dtype=bool)  # create a mask from the labels

    # percent_by_class = np.bincount(label_array[image_array > 0].ravel()) / label_array[label_array > 0].size
    # print(percent_by_class)

    # if stratify:
    #     count_signs = label_array[label_array > 0].size
    # else:
    count_signs = int(label_array[label_array > 0].size * percent)

    if mode == 'random':
        rows, cols = np.where(label_array > 0)
        idx = np.random.choice(rows.size, size=count_signs, replace=False)
        idx = (rows[idx], cols[idx])
        mask[idx] = True

    elif mode == 'secure':
        min_in_neighborhood = minimum_filter(label_array, size=(r*2 + 1))
        max_in_neighborhood = maximum_filter(label_array, size=(r*2 + 1))
        mask_secure_px = (min_in_neighborhood == max_in_neighborhood) & (label_array > 0)
        idx = np.where(mask_secure_px)
        count = count_signs if count_signs <= idx[0].size else idx[0].size
        print("count of value in secure mask: ", count)
        i = np.random.choice(idx[0].size, size=count, replace=False)
        idx = (idx[0][i], idx[1][i])
        mask[idx] = True

    elif mode in ['texture', 'homogeneous']:
        if homogen_layer is None:
            raise Exception("Sorry, you don't provide homogeneous mask.")
        homogen_layer = homogen_layer['array']
        if homogen_layer.shape != label_array.shape:
            raise Exception(f"Homogeneous shape: {homogen_layer.shape} != label: {label_array.shape}")

        homogen_mask = label_array > 0

        if mode == 'texture':
            mean_homogeneous = np.bincount(label_array[homogen_mask], weights=homogen_layer[homogen_mask])[1:]
            sums = np.bincount(label_array[homogen_mask])[1:]
            mean_homogeneous = mean_homogeneous / sums

            homogen_mask = np.zeros_like(label_array, dtype=bool)
            for cl in range(len(mean_homogeneous)):
                mean_class = mean_homogeneous[cl]
                min_mean = mean_class - mean_class * homogen_percent / 2
                max_mean = mean_class + mean_class * homogen_percent / 2
                
                cursor = ((homogen_layer >= min_mean) &
                        (homogen_layer <= max_mean) & 
                        (label_array == cl + 1))
                homogen_mask[cursor] = 1

        elif mode == 'homogeneous':
            classes = np.unique(label_array[label_array > 0])
            homogen_mask = np.zeros_like(label_array, dtype=bool)

            for cl in classes:
                true_px = (label_array > 0) & (label_array == cl)
                q = np.quantile(homogen_layer[true_px], homogen_percent)
                print(f"q of {cl}: ", q)
                true_px = (homogen_layer > 0) & (homogen_layer <= q) & true_px

                idx = np.where(true_px)
                count = int(idx[0].size * percent)
                i = np.random.choice(idx[0].size, size=count, replace=False)
                idx = (idx[0][i], idx[1][i])
                homogen_mask[idx] = 1

        if not stratify:
            rows, cols = np.where(homogen_mask)
            count_signs = int(rows.size * percent)
            i = np.random.choice(rows.size, size=count_signs, replace=False)
            idx = (rows[i], cols[i])
            mask[idx] = 1
        else:
            mask[homogen_mask] = 1

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # OPTIMISE THIS FUNC! 
    if stratify:
        classes, counts = np.unique(label_array[mask], return_counts=True)
        if verbose:
            print("Start stratify. Before output:", classes, counts)
        balanced = np.zeros_like(label_array, dtype=bool)
        min_c = counts.min()

        for c in classes:
            # if c == 6: continue
            rows, cols = np.where((label_array == c) & mask)
            idx = np.random.choice(rows.size, size=min_c, replace=False)
            idx = (rows[idx], cols[idx])
            balanced[idx] = True
        mask[~balanced] = False

    if verbose:
        print(f"After: mode: {mode} & stratify {stratify}:", np.unique(label_array[mask], return_counts=True))

    if save_mask:
        output = DEFAULT_PATH['output']
        name = f'mask_{mode}_r{r}_{percent}'
        if mode == "homogeneous":
            name = name + f'hp_{homogen_percent}'
        if stratify:
            name = name + '_stratified.tif'
        else:
            name = name + '.tif'
        out = label.copy()
        out['array'] = mask
        save_tif(out, output + name, with_bg=True, verbose=verbose)
    if verbose:
        print("Mask created.\n")

    return mask


def stack_and_zip(signs: list, labels: list, mask: np.ndarray, verbose=True):

    if verbose:
        print("Zipping dataset by mask...")
        print("create tensor by bands")


    zip_signs = [s['array'] for s in signs]
    zip_signs = np.moveaxis(np.array(zip_signs), 0, -1)

    zip_labels = [l['array'] for l in labels]
    zip_labels = np.moveaxis(np.array(zip_labels), 0, -1)

    print(f"mask:", mask.shape)

    zip_signs = zip_signs[mask]
    zip_labels = zip_labels[mask].reshape(-1)

    if verbose:
        print(f"Size dataset before -> after zip by mask:")
        print(f"shape of signs:", zip_signs.shape)
        print(f"shape of labels:", zip_labels.shape)
    
    return zip_signs, zip_labels


# Iteration of dataset for training with transform data to tensor.
def generate_dataset(signs, labels, percent, force=False, mask_mode='random', layer_mode=None, layer_type='static', save_mask=False, r=1, homogen_percent=0.01, stratify=False, resize='by_label', verbose=True):

    resized_signs, resized_labels = load_resized_data_labels(signs['path'].to_list(), 
                                                             labels['path'].to_list(), 
                                                             resize=resize, force=force, verbose=verbose)

    red_layers = signs.query("band == 'r'")['path'].to_list()
    nir_layers = signs.query("band == 'n'")['path'].to_list()
    modis_label = resized_labels[0]

    if len(red_layers) != len(nir_layers):
        Exception("Count of red != nir")

    layer_count = 1
    if layer_type == 'dynamic':
        layer_count = len(red_layers)
        print(f"Dynamic layer type. Create {layer_count} layers for each pair of red and nir.")

    if 'homogeneous' in [mask_mode, layer_mode]:
        # homogen_layer = []
        for i in range(layer_count):
            homogeneous = create_homogeneous_layer(red_layers[i], nir_layers[i], modis_label, resize=resize, force=force, order=i, verbose=verbose)
            resized_signs.append(homogeneous)

    elif 'texture' in [mask_mode, layer_mode]:
        # homogen_layer = []
        resize_by = resized_labels[0]['path'] if resize == 'by_label' else None
        for i in range(layer_count):
            texture = create_texture_layer(red_layers[i], nir_layers[i], out=None, resize_by=resize_by, force=force, order=i, verbose=verbose)
            resized_signs.append(texture)

    mask = create_mask(resized_labels[0], resized_signs[0], mode=mask_mode, 
                       percent=percent, stratify=stratify, 
                       r=r, save_mask=save_mask, homogen_percent=homogen_percent, verbose=verbose)
    
    if layer_mode in ['texture', 'homogeneous']: 
        print(f"Append to stack in signs: {layer_mode}")
        # resized_signs.append(homogen_layer)

    zip_signs, zip_labels = stack_and_zip(resized_signs, resized_labels, mask, verbose=verbose)
    return zip_signs, zip_labels, resized_signs, resized_labels