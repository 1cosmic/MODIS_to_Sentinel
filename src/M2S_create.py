from datetime import datetime
import gc
import re
import os
import argparse
from time import time
from typing import List, Optional

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from osgeo import gdal
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


color_palette = {
    0: (0, 0, 0),         # Class None: #000000
    1: (1, 143, 51),      # Class Forest: #018f33
    2: (252, 155, 1),     # Class Land Ownership (землевладения): #fc9b01
    3: (254, 249, 1),     # Class Meadows (луговые): #fef901
    4: (31, 5, 254),      # Class Water: #1f05fe
    5: (220, 16, 16),     # Class Urbanization: #dc1010
    6: (1, 247, 254)      # Class Other: #01f7fe
}


##############
# Basic utils
def printf(msg, verbose):
    if verbose:
        print(msg)


def init_path(path, verbose=False):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created directory {path}")    


def parse_dates(arg: str):
    def parse_range(expr: str, min_v: int, max_v: int):
        values = set()

        for part in expr.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                values.update(range(start, end + 1))
            else:
                values.add(int(part))
        return sorted(v for v in values if min_v <= v <= max_v)

    years = parse_range(arg, 1900, 2100)
    # months = parse_range(month_expr, 1, 12)

    return years


def parse_files(src, type) -> pd.DataFrame:
    def template(path):
        if type == 'signs':
            part = path.parts
            file = part[-1].split('_')
            type_data = file[-4]
            channel = file[-3]
            tile = part[-3]
            date = part[-2].split('_')
            year = int(date[0])
            month = int(date[1])
            day = int(date[2])
            return {"tile": tile, "type_data": type_data, "year": year, "month": month, "day": day, "channel": channel, "path": path}

        elif type == 'labels':
            parts = path.parts
            parts = re.split(r'[_.]', parts[-1])
            year = int(parts[-3])
            return {"year": year, "path": path}

        elif type == 'etalon':
            parts = path.parts
            name = re.split(r'[_.]', parts[-1])

            # Search of year in etalon filename, expected format: *YYYY*.tif
            year = None
            for part in name:
                if part.isdigit():
                    part = int(part)
                    if 1900 <= part <= 2100:
                        year = part
                    break
            if year is None:
                raise ValueError(f"Year in range 1900->2100 not found in etalon: {path}\nExpected format: *YYYY*.tif, please, remove invalid tiles")
            return {"year": year, "path": path}
    
    paths = []
    src_path = Path(src)
    if src_path.is_dir():
        paths.extend(src_path.rglob('*.tif'))
        if not paths:
            raise ValueError(f"No .tif files found in directory: {src}")
    data = [template(p) for p in paths if p.is_file()]
    df = pd.DataFrame(data)
    return df


def load_tif(data: str, verbose=True):
    if not os.path.exists(data):
        raise ValueError(f"File not found: {data}")
    # print(f"Loading file {data}...")

    src = gdal.Open(str(data))
    result = {
            "path": data,
            "array": src.ReadAsArray(),
            "size_x": src.RasterXSize,
            "size_y": src.RasterYSize,
            "transform": src.GetGeoTransform(),
            "projection": src.GetProjection(),
        }
    # printf("File loaded successfully", verbose)
    return result


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
    out = driver.Create(str(path), data['size_x'], data['size_y'], bands, dtype)

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


def cut_tif_by(src, by, out, mode='bilinear', resize=10, aligned=False, verbose=False) -> str:
    by = load_tif(by, verbose=verbose)
    size_x = by['size_x']
    size_y = by['size_y']
    gt = by['transform']
    bounds = [gt[0], gt[3] + gt[5] * size_y,      # minX, minY
              gt[0] + gt[1] * size_x, gt[3]]      # maxX, maxY
    gdal.Warp(
        str(out), str(src),
        outputBounds=bounds,
        dstSRS=by['projection'],
        xRes=resize, yRes=resize,
        resampleAlg=mode, targetAlignedPixels=aligned if aligned else None,
        dstNodata=0)
    if verbose:
        print(f"File {os.path.basename(src)} was cutted and saved to {out}")

    return out


##############
# Pipeline steps
def prepare_and_load_data(signs, labels, out, cache, verbose=False):
    if not signs:
        raise ValueError(f"No sign files found.")
    if not labels:
        raise ValueError(f"No label files found.")
    msg = f"Preparing and loading data for {len(signs)} signs, {len(labels)} labels"
    printf(msg, verbose)

    prepared_signs = []
    prepared_labels = []

    init_path(out)
    labels = sorted(labels, key=lambda x: os.path.basename(x))

    pbar = tqdm(labels, desc="Preparing labels") if verbose else labels
    for l in pbar:
        name = os.path.basename(l)
        status = f"Prepare & load label: {name}"
        if verbose:
            pbar.set_description(status)

        cutted_out = out.joinpath(name)
        l = cut_tif_by(l, signs[0], str(cutted_out), mode='nearest', resize=10)
        prepared_labels.append(load_tif(l))

    i = 0
    pbar = tqdm(signs, desc="Preparing signs") if verbose else signs
    for s in pbar:
        name = os.path.basename(s)
        gt = gdal.Open(str(s)).GetGeoTransform()
        size_x, size_y = gt[1], gt[5]
        if size_x != 10 or size_y != -10:
            status = f"Resizing sign: {name}"
            if verbose:
                pbar.set_description(status)
            resized_out = cache.joinpath(f"{i}_{name}")
            s = cut_tif_by(s, prepared_labels[0]['path'], str(resized_out), mode='bilinear', resize=10)
            i += 1

        if verbose:
            status = f"Loading sign: {name}"
            pbar.set_description(status)
        prepared_signs.append(load_tif(s))

    t_start = time()
    printf("Data prepared and loaded successfully. Convert to np.array...", verbose)
    signs = np.stack([s['array'] for s in prepared_signs])
    signs = np.moveaxis(signs, 0, -1)  # (C, H, W) -> (H, W, C)
    labels = np.stack([l['array'] for l in prepared_labels])
    labels = np.moveaxis(labels, 0, -1)  # (C, H, W) -> (H, W, C)
    if labels.shape[2] == 1:
        labels = np.squeeze(labels, axis=2)  # (H, W, 1) -> (H, W)
    t_end = time()
    msg = f"Converted to np.array successfully in {t_end - t_start:.2f} seconds. Signs shape: {signs.shape}, Labels shape: {labels.shape}"
    printf(msg, verbose)

    # For creating map we need template with geoinformation, but without data
    template = prepared_signs[0]
    template['array'] = None
    return signs, labels, template



def create_dataset(signs, label, r=20, sign_per_class=5000, verbose=False):
    msg = "Creating dataset from signs and labels use method 'secure'"
    printf(msg, verbose)

    mask = (label > 0) & (signs.sum(axis=-1) > 0)  # take only valid signs and labels

    t_start = time()
    printf("Generating safe zones for sampling...", verbose)
    min_neighbors = minimum_filter(label, size=(r+1) * 2)
    max_neighbors = maximum_filter(label, size=(r+1) * 2)
    safezone = (min_neighbors == max_neighbors) & (mask > 0)

    classes, per_class = np.unique(label[safezone], return_counts=True)
    sign_per_class = min(sign_per_class, per_class.min())

    mask = np.zeros_like(label, dtype=bool)
    for cl in tqdm(classes, desc="Selecting signs per class") if verbose else classes:
        idx = np.where(safezone & (label == cl))
        i = np.random.choice(len(idx[0]), size=sign_per_class, replace=False)
        idx = (idx[0][i], idx[1][i])
        mask[idx] = True
    X = signs[mask]
    y = label[mask]
    t_end = time()
    printf(f"Dataset created successfully in {t_end - t_start:.2f} seconds", verbose)
    printf(f"Sampled {len(X)} per {classes[-1]} signs for training", verbose)
    return X, y


def train_model(X, y, verbose):
    msg = f"Training model"
    printf(msg, verbose)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    printf(f"Model trained successfully with accuracy: {accuracy:.2f}", verbose)
    print(classification_report(y_test, y_pred))
    
    return model


def create_map(model, signs, out, template, chunk=100, verbose=False):
    msg = f"Creating map with model, outputting to {out}"
    printf(msg, verbose)
    
    H, W, C = signs.shape
    signs = signs.reshape(-1, C)  # (H*W, C)
    signs = signs.astype(np.float32, copy=False)  # speedup model prediction
    chunk = chunk * 1024*1024       # in MB

    size_sign = signs.dtype.itemsize * C  # (sign = type * channels)
    per_chunk = int(chunk // size_sign)

    idx = np.where(signs.sum(axis=1) > 0)[0]  # only valid signs

    res = np.zeros((H * W,), dtype=np.uint8)
    for i in tqdm(range(0, idx.shape[0], per_chunk),
                  desc="Creating map") if verbose else range(0, idx.shape[0], per_chunk):
        end = min(i + per_chunk, idx.shape[0])
        
        res[idx[i:end]] = model.predict(signs[idx[i:end]])

    res = res.reshape(H, W)
    template['array'] = res
    save_tif(template, out, 
             dtype=gdal.GDT_Byte, with_bg=True, color_palette=color_palette, verbose=verbose)
    return template


def validate_map(result, etalon, out=None, verbose=False):
    msg = f"Validating result map with etalon"
    printf(msg, verbose)

    mask = (result > 0) & (etalon > 0)
    validate = classification_report(result[mask].ravel(), etalon[mask].ravel())
    printf(validate, verbose)
    if out is not None:
        with open(out, 'w') as f:
            f.write(validate)
        printf(f"Validation report saved to {out}", verbose)


##############
# Main flow
def run_pipeline(
    src: str,
    labels: Optional[str],
    out: str,
    years: str,
    channels: List[str],
    etalons: Optional[str],
    chunk: int,
    verbose: bool,
    force: bool,
    ignore_error: bool,
):
    years = parse_dates(years)
    processed_out = Path(out).joinpath('processed')
    cache_out = Path(out).joinpath('cache')

    src = parse_files(src, type='signs')
    tiles = src['tile'].unique()
    labels = parse_files(labels, type='labels')

    validate = False
    if etalons is not None:
        etalons = parse_files(etalons, type='etalon').query("year in @years")
        if etalons.empty:
            raise ValueError(f"No etalon files found for years {years}. Please, check the path and format of etalon files.")
        validate = True

    # For each tile and year create map, validate it with etalon (if provided) and save to output
    qbar = tqdm(total=len(tiles) * len(years), desc="Processing tiles and years") if verbose else None
    for tile in tiles:
        for year in years:
            qbar.set_description(f"Processing tile {tile} for  {year}") if verbose else None
            qbar.update(1) if verbose else None

            curdir = Path(out).joinpath(tile).joinpath(str(year))
            init_path(curdir, verbose=verbose)
            save_out = curdir.joinpath(f"M2C_{tile}_{year}.tif")

            # Aggregate the signs, labels for the current tile and year
            signs = src.query("tile == @tile and year == @year and channel in @channels")
            signs = signs.sort_values(by=['year', 'month', 'day', 'channel'])
            label = labels.query("year == @year").sort_values('year')
            if signs.empty or label.empty:
                printf(f"No sign/label files found for tile {tile} and year {year}. Skipping...", verbose)
                continue
            signs = signs['path'].tolist()
            label = label['path'].tolist()

            try:
                # If result map's exist - skip double work, but if need to rewrite - use force
                if not os.path.exists(save_out) or force:
                    # Prepare & load this data to numpy from source.tif
                    signs, label, template = prepare_and_load_data(signs, label, 
                                                                out=processed_out.joinpath(tile),
                                                                cache=cache_out,
                                                                verbose=verbose)
                    X, y = create_dataset(signs, label, sign_per_class=5000, verbose=verbose)
                    m = train_model(X, y, verbose)
                    X, y, label = None, None, None
                    tile_map = create_map(m, signs, save_out, template=template, chunk=chunk, verbose=verbose)
                    signs = None
                    gc.collect()
                else:
                    printf(f"File {os.path.basename(save_out)} already exists. Skipping tile {tile}", verbose)
                    tile_map = load_tif(save_out, verbose=verbose)

                # Validate result map with etalon (if provided)
                if validate:
                    etalon = etalons.query("year == @year")['path'].tolist()
                    name = os.path.basename(etalon[0]).replace(".tif", "")
                    etalon = etalon[0]   # take only 1st etalon in this year
                    etalon = cut_tif_by(etalon, str(save_out), 
                                        curdir.joinpath(f"etalon_{name}.tif"), 
                                        mode='nearest', resize=10, aligned=True, verbose=verbose)
                    etalon = load_tif(etalon, verbose=verbose)['array']
                    tile_map = tile_map['array']
                    validate_map(tile_map, etalon,
                                    out=curdir.joinpath(f"valid_report_{name}.txt"), verbose=verbose)

            except Exception as e:
                if ignore_error:
                    printf(f"Error processing tile {tile} for year {year}: {e}", verbose=verbose)
                    continue
                else:
                    raise e


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for running the pipeline of creating a map from MODIS data"
    )

    # Required / main args
    parser.add_argument("--src", required=True, help="Path to source data (Sentinel-2)")
    parser.add_argument("--labels", required=True, help="Path to labels (MODIS)")
    parser.add_argument("--year", required=True, type=str, help="Sampling years: 2018-2020")
    parser.add_argument("--out", default=os.getcwd(), help="Path to output map")
    parser.add_argument("--chunk", default=300, type=int, help="Chunk size (in MB) for processing")
    parser.add_argument(
        "--channels",
        type=lambda s: s.split(","),
        default=["b2", "b3", "b4", "b8", "b11"],
        help="List of channels (',' separated, like --channels b2,b3,b4 ...)",
    )
    parser.add_argument(
        "--validate_by",
        default=None,
        help="Validation strategy"
    )

    # Extra prototype args
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output in terminal")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    parser.add_argument("--ignore_error", action="store_true", help="Ignore errors during processing")

    return parser.parse_args()


def main():
    args = parse_args()
    paths = ['processed', 'models', 'cache']
    for p in paths:
        init_path(Path(args.out).joinpath(p), verbose=args.verbose)

    run_pipeline(
        src=args.src,
        labels=args.labels,
        years=args.year,
        out=args.out,
        channels=args.channels,
        etalons=args.validate_by,
        chunk=args.chunk,
        force=args.force,
        verbose=args.verbose,
        ignore_error=args.ignore_error,
    )


if __name__ == "__main__":
    main()