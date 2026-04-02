# For preparation of dataset

from time import time
import gc
import os
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter, sobel
from osgeo import gdal

from utils import init, load_tif, cut_tif_by, save_tif, DEFAULT_PATH, GEO_DATA
from visualisation import draw_median_sign, draw_hist_cosine_dists

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

    elif resize == 'all_signs':
        if verbose:
            print("\nResizing labels by 1st image:")
        sources = signs
        by = load_tif(signs[0], only_first=True, verbose=verbose)
        resize_path = 'resized_labels'
    else:
        raise ValueError("resize can be only [by_sign, by_label, all_signs]")

    i = 0
    resized = []
    switch_mode = 'bilinear' if resize == 'by_sign' else 'nearest'
    for src in sources:
        name = os.path.basename(src)
        out = DEFAULT_PATH[resize_path] + f'{i}_' + name
        i += 1
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
    if resize in ['by_label', 'all_signs']:
        return resized, cropped_labels
    elif resize == 'by_sign':
        loaded_signs = [load_tif(s, verbose=verbose, only_first=True) for s in signs]
        return loaded_signs, resized
    else:
        return [], []


def mask_cosine_dist_to_median_sign(signs, label, take_q, r=20, median_mode='similar', draw=True, verbose=True):
    if median_mode not in ['similar', 'not_similar']:
        raise ValueError("median_mode can be only [similar, not_similar]")

    x = np.array([s['array'] for s in signs])
    x = np.moveaxis(x, 0, -1)
    label = label['array']
    classes = np.unique(label[label > 0])

    r = 20

    print("Calculating secure area for selecting...")
    min_in_neighborhood = minimum_filter(label, size=(r*2 + 1))
    max_in_neighborhood = maximum_filter(label, size=(r*2 + 1))
    secure = (min_in_neighborhood == max_in_neighborhood)

    t_0 = time()

    median_signs = []
    bins_median_signs = []
    result_mask = np.ones_like(label, dtype=bool)
    result_mask[~secure] = 0

    print("Calculating median sign for each class...")
    # for cl in classes[:4]:
    for cl in classes:
        to_class = (label == cl) & result_mask
        print(f"Statistic for class: {cl}")
 
        if median_mode == 'not_similar':
            y = np.median(x[to_class], axis=0)
            print(f"median of sign:\n{y}")

            norm_x = np.linalg.norm(x[result_mask], axis=1)
            norm_y = np.linalg.norm(y)
            scalar_x_y = x[result_mask].dot(y)
            print(f"shapes: x: {norm_x.shape}, y: {norm_y.shape}, scalar: {scalar_x_y.shape}")

            cosine_dist = 1 - scalar_x_y / (norm_x * norm_y)
            to_class = (label == cl)[result_mask]
            dist_max = np.quantile(cosine_dist[to_class], take_q)
            print(f"max of dist to class == {cl}: {dist_max}")

            idx = np.where(result_mask)
            another = (label != cl)[result_mask] & (cosine_dist < dist_max)

            mask_remove = (
                idx[0][another],
                idx[1][another],
            )

            print(
                f"removed signatures for [label > {cl}]: {result_mask[mask_remove].size}"
            )
            result_mask[mask_remove] = 0

        elif median_mode == 'similar':
            y = np.median(x[to_class], axis=0)
            print(f"median of sign:\n{y}")

            norm_x = np.linalg.norm(x[to_class], axis=1)
            norm_y = np.linalg.norm(y)
            scalar_x_y = x[to_class].dot(y)
            print(f"shapes: x: {norm_x.shape}, y: {norm_y.shape}, scalar: {scalar_x_y.shape}")

            cosine_dist = 1 - scalar_x_y / (norm_x * norm_y)
            dist_max = np.quantile(cosine_dist, take_q)
            print(f"max of dist to class == {cl}: {dist_max}")

            idx = np.where(to_class)
            not_similar = (cosine_dist > dist_max)

            mask_remove = (
                idx[0][not_similar],
                idx[1][not_similar],
            )

            print(
                f"removed signatures for [label == {cl}]: {result_mask[mask_remove].size}"
            )
            result_mask[mask_remove] = 0


            if draw:
                median_signs.append(y)
                print("calcs of cosine dists for draw...")
                counts, bins = np.histogram(cosine_dist, bins=np.arange(0, 1.1, 0.025))
                bins_median_signs.append((counts, bins))

        else:
            raise Exception(f"Unknown median mode: {median_mode}")

    if draw:
        draw_median_sign(median_signs)
        # draw_hist_cosine_dists(bins_median_signs)

    print(f"Cosine distance to median sign is calculated. Time: {time() - t_0:0.3f}s")
    return result_mask


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
    if resize in ['by_sign', 'all_signs']:
        cut_tif_by(out, sentinel_nir, out, resize=10, mode='bilinear', verbose=verbose)
        modis_label = load_tif(out, only_first=True, verbose=verbose)
        print("Resizing done.")

    if verbose:
        print("Std layer created and saved to", out)
    return modis_label


def create_mask(label: list[dict], signs: list[dict], count_signs:int = 5000, r=20,
                mode:str = 'random', save_mask=False,
                feature_layer=None, feature_percent=0.1, stratify=False, draw=False, verbose=True, median_mode='similar') -> np.ndarray:

    if verbose:
        print("Creating mask...")
    image_array = signs[0]['array']
    label_array = label[0]['array']

    label_array[image_array <= 0] = 0              # work with ONLY FILLED pixels
    mask = np.zeros_like(label_array, dtype=bool)  # create a mask from the labels

    if mode == 'random':
        rows, cols = np.where(label_array > 0)
        idx = np.random.choice(rows.size, size=count_signs, replace=False)
        idx = (rows[idx], cols[idx])
        mask[idx] = True



    if mode == 'unique':
        t0 = time()
        x = np.array([s['array'] for s in signs])
        x = np.moveaxis(x, 0, -1)

        h, w, c = x.shape
        pixels = x.reshape(-1, c)

        # Храним хеши и индексы первых вхождений
        seen = {}
        idx = np.zeros(len(pixels), dtype=bool)

        for i in tqdm(range(len(pixels)), desc="Hashing"):
            h = hash(pixels[i].tobytes())
            if h not in seen:
                seen[h] = i
                idx[i] = 1
        
        mask = idx.reshape(h, w)


    elif mode == 'secure':
        min_in_neighborhood = minimum_filter(label_array, size=(r*2 + 1))
        max_in_neighborhood = maximum_filter(label_array, size=(r*2 + 1))
        mask_secure_px = (min_in_neighborhood == max_in_neighborhood) & (label_array > 0)
        idx = np.where(mask_secure_px)

        if stratify:
            count = idx[0].size
        else:
            count = count_signs if count_signs < idx[0].size else idx[0].size

        print("count of value in secure mask: ", idx[0].size)
        i = np.random.choice(idx[0].size, size=count, replace=False)
        idx = (idx[0][i], idx[1][i])
        mask[idx] = True


    elif mode == 'median':
        mask = mask_cosine_dist_to_median_sign(signs, label[0], feature_percent, median_mode=median_mode, r=r, draw=draw)
    

    elif mode in ['texture', 'homogeneous']:
        if feature_layer is None:
            raise Exception("Sorry, you don't provide homogeneous mask.")
        feature_layer = feature_layer['array']
        if feature_layer.shape != label_array.shape:
            raise Exception(f"Homogeneous shape: {feature_layer.shape} != label: {label_array.shape}")

        if mode == 'texture':
            classes = np.unique(label_array[label_array > 0])
            homogen_mask = np.zeros_like(label_array, dtype=bool)
            for cl in classes:

                l_mean = np.quantile(feature_layer[label_array == cl], 0.5 - feature_percent / 2)
                mean = np.quantile(feature_layer[label_array == cl], 0.5)
                r_mean = np.quantile(feature_layer[label_array == cl], 0.5 + feature_percent / 2)

                cursor = (((feature_layer >= l_mean)  &
                           (feature_layer <= r_mean)) & 
                           (label_array == cl))
                homogen_mask[cursor] = 1

                print(f"class: {cl}, l_mean, mean, r_mean", l_mean, mean, r_mean)

        elif mode == 'homogeneous':
            classes = np.unique(label_array[label_array > 0])
            homogen_mask = np.zeros_like(label_array, dtype=bool)

            for cl in classes:
                true_px = (label_array == cl)
                q = np.quantile(feature_layer[true_px], feature_percent)
                q2 = np.quantile(feature_layer[true_px], 1 - feature_percent)
                q_mean = np.quantile(feature_layer[true_px], 0.5)
                q_mean_l = np.quantile(feature_layer[true_px], 0.5 - feature_percent / 2)
                q_mean_r = np.quantile(feature_layer[true_px], 0.5 + feature_percent / 2)
                print(f"class {cl}: q_left: {q}\tq_right: {q2}\tmean: {q_mean}\tmean_l: {q_mean_l}\tmean_r: {q_mean_r}")

                true_px = (feature_layer > 0) & (feature_layer <= q) & true_px  # MODE 1: take only homogen
                print("Homogen mode: only homogen")
                # true_px = ((homogen_layer <= q) | (homogen_layer >= q2)) & true_px    # MODE 2: take homogen & extragen
                # print("Homogen mode: homogen & extragen")
                # true_px = ((feature_layer >= q_mean_l) & (feature_layer <= q_mean_r)) & true_px    # MODE 2: take window of mean 
                # print("Homogen mode: in window around mean")

                idx = np.where(true_px)
                # count = int(idx[0].size * percent)
                i = np.random.choice(idx[0].size, size=count_signs, replace=False)
                idx = (idx[0][i], idx[1][i])
                homogen_mask[idx] = 1

        if not stratify:
            rows, cols = np.where(homogen_mask)
            # count_signs = int(rows.size * percent)
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
        min_c = counts.min() if counts.min() < count_signs else count_signs

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
        name = f'mask_{mode}_r{r}_{count_signs}'
        if mode == "homogeneous":
            name = name + f'_hp_{feature_percent}'
        if stratify:
            name = name + '_stratified.tif'
        else:
            name = name + '.tif'
        out = label[0].copy()
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
def generate_dataset(signs, labels, count, force=False, mask_mode='random', layer_mode=None, layer_type='static', median_mode='similar', save_mask=False, r=20, feature_percent=0.1, stratify=False, resize='by_label', draw=False, verbose=True):

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

    feature = []
    if 'homogeneous' == layer_mode:
        for i in range(layer_count):
            homogeneous = create_homogeneous_layer(red_layers[i], nir_layers[i], modis_label, resize=resize, force=force, order=i, verbose=verbose)
            feature.append(homogeneous)

    elif 'texture' == layer_mode:
        resize_by = resized_labels[0]['path'] if resize == 'by_label' else None
        for i in range(layer_count):
            texture = create_texture_layer(red_layers[i], nir_layers[i], out=None, resize_by=resize_by, force=force, order=i, verbose=verbose)
            feature.append(texture)


    feature_layer = None
    if 'homogeneous' == mask_mode:
        feature_layer = create_homogeneous_layer(red_layers[0], nir_layers[0], modis_label, resize=resize, force=force, verbose=verbose)

    elif 'texture' == mask_mode:
        resize_by = resized_labels[0]['path'] if resize == 'by_label' else None
        feature_layer = create_texture_layer(red_layers[0], nir_layers[0], out=None, resize_by=resize_by, force=force, order=0, verbose=verbose)

    mask = create_mask(resized_labels, resized_signs, mode=mask_mode, 
                       count_signs=count, stratify=stratify, feature_layer=feature_layer, median_mode=median_mode,
                       r=r, save_mask=save_mask, feature_percent=feature_percent, draw=draw, verbose=verbose)

    if layer_mode in ['texture', 'homogeneous']: 
        print(f"Append to stack in signs: {layer_mode}")
        resized_signs.extend(feature)

    zip_signs, zip_labels = stack_and_zip(resized_signs, resized_labels, mask, verbose=verbose)
    return zip_signs, zip_labels, resized_signs, resized_labels