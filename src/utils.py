# Utils for meta-use in other modules.
import os
import glob
import re
from osgeo import gdal
import pandas as pd
import numpy as np

# Constants
GEO_DATA = 1

# Get the directory of the current module
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

color_palette = {
    0: (0, 0, 0),         # Class None: #000000
    1: (1, 143, 51),      # Class Forest: #018f33
    2: (252, 155, 1),     # Class Land Ownership (землевладения): #fc9b01
    3: (254, 249, 1),     # Class Meadows (луговые): #fef901
    4: (31, 5, 254),      # Class Water: #1f05fe
    5: (220, 16, 16),     # Class Urbanization: #dc1010
    6: (1, 247, 254)      # Class Other: #01f7fe
}
# convert 0–255 RGB → 0–1 RGB
color_palette_plot = {
    k: tuple(np.array(v)/255.0)
    for k, v in color_palette.items()
}


def rel_path(path):
    return os.path.join(MODULE_DIR, path)

DEFAULT_PATH = {
    'images': rel_path('../data/input/images_10m/Sentinel/'),
    'labels': rel_path('../data/input/labels_230m/'),
    'etalons': rel_path('../data/input/etalons/'),
    'processing': rel_path('../data/processing/'),
    'cropped_labels': rel_path('../data/processing/cropped/labels/'),
    'resized_images': rel_path('../data/processing/resized/images/'),
    'resized_labels': rel_path('../data/processing/resized/labels/'),
    'resized_layers': rel_path('../data/processing/resized/layers/'),
    'cropped_etalons': rel_path('../data/processing/cropped/etalons/'),
    'output': rel_path('../data/output/'),
}


def parse_tifs_from(path:str, typeof:str, force=False, verbose=True, cache_csv=None):

    if cache_csv is not None:
        out = DEFAULT_PATH['processing'] + cache_csv
    else:
        out = DEFAULT_PATH['processing'] + f'path2tif_{typeof}.csv'

    if os.path.exists(out) and not force:
        if verbose:
            print(f"Load '{typeof}' from cached DataFrame to path: ", out)
        return pd.read_csv(out)

    if typeof == 'sign':
        def template_sign(struct, fullpath):
            return {
                    'type': typeof,
                    'reg': int(struct[1]),
                    'year': int(struct[2]),
                    'month': int(struct[3]),
                    'season': struct[4],
                    'band': struct[6],
                    'path': fullpath,
                }
        template = template_sign

    elif typeof == 'label':
        def template_label(struct, fullpath):
            return {
                    'type': typeof,
                    'name': struct[0],
                    'year': int(struct[2]),
                    'path': fullpath,
                }
        template = template_label

    elif typeof == 'tile':
        def template_tile(struct, fullpath):
            # struct: ['comp', 's2', '90d', 'b8', 'l2a', 'med', 'frag', '0', 'tif']
            band_map = {
                'b2': 'b',
                'b3': 'g',
                'b4': 'r',
                'b8': 'n',
                'b11': 'swir1',
                'b12': 'swir2'
            }
            # Extract date from fullpath: .../ZONE/2020_06_01/comp_s2_90d_b4_l2a_med.frag.0.tif
            date_folder = os.path.basename(os.path.dirname(fullpath))
            date_parts = date_folder.split('_')
            year = int(date_parts[0]) if len(date_parts) > 0 else None
            month = int(date_parts[1]) if len(date_parts) > 1 else None
            return {
                'type': typeof,
                'period': struct[2],
                'band': band_map[struct[3]],
                'year': year,
                'month': month,
                'fragment': struct[7],
                'path': fullpath,
            }
        template = template_tile

    elif typeof == 'etalon':
        def template_etalon(struct, fullpath):
            # struct: ['r36000', '2020', 'wc', 'cor', 'tif']
            return {
                'type': typeof,
                'year': struct[1],
                'map': struct[2],
                'path': fullpath,
            }
        template = template_etalon
    else:
        raise ValueError('typeof can be only [sign, label, tile, etalon]')

    data = []
    if typeof == 'tile':
        tifs = glob.glob(os.path.join(path, '**', '*.tif'), recursive=True)
    else:
        tifs = glob.glob(os.path.join(path, '*.tif'), recursive=True)

    for t in tifs:
        name = os.path.basename(t)
        struct = re.split(r'[_.]', name)
        data.append(template(struct, t))
    
    df = pd.DataFrame(data)
    df.to_csv(out, index=False)
    if verbose:
        print(f"Parsed {len(df)} tifs of type '{typeof}' and saved to {out}")
    return df


def check_output():
    pass

def create_output():
    pass


def init(verbose=True):
    os.environ["OPENBLAS_NUM_THREADS"] = "6"  # Для M1
    os.environ["OMP_NUM_THREADS"] = "6"
    if verbose:
        print("Initialization paths...")
    for path in DEFAULT_PATH.values():
        if not os.path.exists(path):
            os.makedirs(path)
            if verbose:
                print(f"Created directory: {path}")
    else:
        if verbose:
            print("All paths was initialized.")


def cut_tif_by(src, by, out, mode='mode', resize=10, aligned=False, verbose=True):
    size_x = by['size_x']
    size_y = by['size_y']
    gt = by['transform']
    bounds = [gt[0], gt[3] + gt[5] * size_y,      # minX, minY
            gt[0] + gt[1] * size_x, gt[3]]      # maxX, maxY

    gdal.Warp(
        out,
        src,
        outputBounds=bounds,
        dstSRS=by['projection'],
        xRes=resize,
        yRes=resize,
        resampleAlg=mode,
        targetAlignedPixels=aligned if aligned else None,
        dstNodata=0)
    if verbose:
        print(f"Map cutted and saved to {out}")

    return out


def downgrade_classes(src, out, assign_class, force=False, verbose=True):
    if os.path.exists(out) and not force:
        if verbose:
            print("\nSkip downgrade the map of MODIS. If need, set force=True")
        return
    os.makedirs(out, exist_ok=True)

    if verbose:
        print("\nParse .csv of new classes...")
    new_classes = pd.read_csv(assign_class, delimiter=';')
    new_classes = new_classes[new_classes["version"] == 5.71]
    max_old_class = new_classes["class_id"].max()
    class_mapping = np.zeros(max_old_class + 1, dtype=np.int32)
    
    # Fill mapping (assuming class_id is 0-based and continuous in CSV)
    for _, row in new_classes.iterrows():
        class_mapping[row["class_id"]] = row["igce_id"]
    
    if verbose:
        print("Class mapping:")
        for old_id, new_id in enumerate(class_mapping):
            print(f"Old class {old_id} → New class {new_id}")

    # Process files
    files = glob.glob(f"{src}/*.tif")
    for src_file in files:
        output_file = os.path.join(out, os.path.basename(src_file))
        if verbose:
            print("\nProcessing:", os.path.basename(src_file))
        
        src_ds = gdal.Open(src_file)
        data = src_ds.GetRasterBand(1).ReadAsArray()
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.CreateCopy(output_file, src_ds, 0)
        
        # Apply class mapping using vectorized operation
        output_data = np.zeros_like(data)
        for old_id in range(len(class_mapping)):
            output_data[data == old_id] = class_mapping[old_id]
        unmapped = ~np.isin(data, range(len(class_mapping)))
        if np.any(unmapped) and verbose:
            print(f"Warning: {unmapped.sum()} pixels with unmapped classes")
            output_data[unmapped] = 0  # Or another default value
        
        # Write output.
        dst_ds.GetRasterBand(1).WriteArray(output_data)

        # Apply new colors.
        colors = gdal.ColorTable()
        for class_val, rgb in color_palette.items():
            colors.SetColorEntry(class_val, rgb)
        dst_ds.GetRasterBand(1).SetRasterColorTable(colors)
        dst_ds.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex)  
        dst_ds.FlushCache()
        src_ds = dst_ds = None
    
    if verbose:
        print("\nAll classes assigned successfully.")


def load_tif(data: str,  only_first=False, verbose=True):
    # Load data and geoinfo from the specified path
    if not os.path.exists(data):
        raise ValueError(f"Path not found: {data}")

    results = []
    if os.path.isfile(data):
        data_files = [data]
    else:
        data_files = glob.glob(os.path.join(data, '*.tif'))
        if only_first:
            data_files = data_files[:1]
        if not data_files:
            raise ValueError(f"No data files found in path: {data}")

    shape = None
    for tif in data_files:
        if verbose:
            print(f"Loading file: {tif}")
        src = gdal.Open(tif)
        results.append({
            "path": tif,
            "array": src.ReadAsArray(),
            "size_x": src.RasterXSize,
            "size_y": src.RasterYSize,
            "transform": src.GetGeoTransform(),
            "projection": src.GetProjection(),
        })
        shape = results[-1]['array'].shape
        src = None  # clear memory
    if verbose:
        print(f"Files was loaded with shape: {shape}")

    if only_first:
        return results[0]
    else:
        return results


def save_tif(data: dict, path: str, dtype=None, with_bg=False, color_palette=None, verbose=True):
    # Save data to the specified path
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    src = data['array']
    if len(src.shape) == 2:
        bands = 1
    elif len(src.shape) == 3:
        bands = src.shape
    else:
        raise ValueError("Data must be 2D or 3D numpy array")

    driver = gdal.GetDriverByName('GTiff')
    dtype = gdal.GDT_Byte if dtype is None else dtype
    out = driver.Create(path, data['size_x'], data['size_y'], bands, dtype)

    out.SetGeoTransform(data['transform'])
    out.SetProjection(data['projection'])

    if bands == 1:
        out.GetRasterBand(1).WriteArray(src)
        if not with_bg:
            out.GetRasterBand(1).SetNoDataValue(0)
    else:
        for i in range(bands):
            out.GetRasterBand(i + 1).WriteArray(src[i])

    if color_palette is not None:
        colors = gdal.ColorTable()
        for class_val, rgb in color_palette.items():
            colors.SetColorEntry(class_val, rgb)
        out.GetRasterBand(1).SetRasterColorTable(colors)
        out.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex) 

    out.FlushCache()
    if verbose:
        print(f"Data saved to {path}")