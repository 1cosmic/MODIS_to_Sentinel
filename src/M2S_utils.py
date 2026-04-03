import os
from osgeo import gdal
import glob
import argparse
import pandas as pd
import numpy as np


class_matching = pd.DataFrame({
    'class_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 23, 11, 18, 9, 10, 15, 16, 21, 22, 12, 13, 14, 19, 20],
    'igce_id': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 5, 6, 6, 6, 6, 6],
})

color_palette = {}


def downgrade_classes(src, out, reassign=None, force=False, verbose=True):
    if os.path.exists(out) and not force:
        if verbose:
            print("\nSkip downgrade the map of MODIS. If need, use --force")
        return
    os.makedirs(out, exist_ok=True)

    # Load class mapping or use cached
    if reassign is not None:
        new_classes = pd.read_csv(reassign, delimiter=';')
    else:
        new_classes = class_matching

    max_old_class = new_classes["class_id"].max()
    class_mapping = np.zeros(max_old_class + 1, dtype=np.uint8)

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
        # dst_ds.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex)  
        dst_ds.FlushCache()
        src_ds = dst_ds = None
    
    if verbose:
        print("\nAll classes assigned successfully.")


def stick_tifs(src, out, force=False, verbose=True):

    # List your input files in desired band order
    input_files = sorted(glob.glob(f"{src}/*.tif"))
    if not input_files:
        if verbose:
            print("\nNo input files found.")
        return

    if verbose:
        print(f"\nFound {len(input_files)} files to merge:")
        for f in input_files:
            print(" -", os.path.basename(f))

    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT("temp.vrt", input_files, options=vrt_options)
    translate_options = gdal.TranslateOptions(
        format="GTiff",
        creationOptions=["BIGTIFF=YES"]  # BIGTIFF for large files
    )
    gdal.Translate(out, vrt, options=translate_options)
    vrt = None

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downgrade MODIS classes")
    parser.add_argument("--src", required=True, help="Source directory with .tif files")
    parser.add_argument("--out", required=True, help="Output directory")

    parser.add_argument("--assign_cache_classes", action="store_true", help="Use cached class assignment")
    parser.add_argument("--assign_new_classes", action="store", help="Path to class assignment CSV template")
    parser.add_argument("--stick", action="store_true", help="Stick tifs from source directory into one file")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing output")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    
    args = parser.parse_args()

    if args.assign_cache_classes or args.assign_new_classes:
        downgrade_classes(args.src, args.out, args.assign_new_classes, args.force, args.verbose)
    elif args.stick:
        stick_tifs(args.src, args.out, args.force, args.verbose)