# Module 2: create map by predict on weights
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


from utils import save_tif, color_palette, DEFAULT_PATH, load_tif
from prepare_ds import create_homogeneous_layer, create_texture_layer


def create_map(signs, model, name: str, count_chunks=2, layer_mode=None, layer_type='static'):

    if model is str:
        model = joblib.load(model)
    
    textures = []
    if isinstance(signs, pd.DataFrame):
        signs = signs.sort_values("band").sort_values("month")
        red = signs.query("band == 'b4' or band == 'r'")['path'].to_list()
        nir = signs.query("band == 'b8' or band == 'n'")['path'].to_list()
        signs = signs['path'].to_list()

        if layer_mode == 'texture':
            layer_count = 1
            if layer_type == 'dynamic':
                layer_count = len(red)
                print(f"Dynamic layer type. Create {layer_count} texture layers.")
        
            for i in range(layer_count):
                textures.append(create_texture_layer(red[i], nir[i], out=None, order=i))

        else:
            Exception("Layer mode can be only 'texture'")
    

    loaded_signs = []
    for sig in signs:
        loaded_signs.append(load_tif(sig, only_first=True))
    loaded_signs.extend(textures)

    # model = joblib.load(model)
    X = np.stack([sig['array'] for sig in loaded_signs])
    X = np.moveaxis(X, 0, -1)

    s_x = X.shape[0]
    s_y = X.shape[1]
    band = X.shape[2]
    print("Reshaping tensor-images...")

    # TOOD: X.reshape() for Full reshape's very slow.
    # D.S. say, what X{chunks}.reshape's can be fast.
    X = X.reshape(-1, band)

    print("Start create of map...")
    print(f"\nmap size: x = {s_x}, y = {s_y}, bands = {band}, total px={s_x * s_y}")

    start_time = time.time()

    MB = pow(1024, 2)
    chunk_s = int(count_chunks * MB) # adjust based on your RAM
    if len(X) < chunk_s:
        chunk_s = int(len(X) / 5)

    predict_map = np.zeros(X.shape[0], dtype=np.uint8)
    for i in tqdm(range(0, len(X), chunk_s), desc="Creating map..."):
        chunk = X[i:i + chunk_s]
        predict_map[i:i + chunk_s] = model.predict(chunk)
    predict_map = predict_map.reshape((s_x, s_y))
    print("Map is done.")
    elapsed_time = round((time.time() - start_time) / 60, 1)
    
    out_img = loaded_signs[0]
    out_img['array'] = predict_map
    save_tif(out_img, name, color_palette=color_palette)

    return name