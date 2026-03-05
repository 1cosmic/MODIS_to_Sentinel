# Module of validation the predicted maps by etalon maps.

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import glob
import os

from utils import DEFAULT_PATH, cut_tif_by, load_tif, save_tif
from utils import parse_tifs_from, color_palette


def validate_how_array(predicted, etalon):
    if predicted.shape != etalon.shape:
        raise Exception(f"Shape of 'predicted' {predicted.shape} != {etalon.shape} 'etalon'.")
    
    mask = (predicted > 0) & (etalon > 0)
    etalon = etalon[mask]
    predicted = predicted[mask]

    if len(predicted.shape) > 1:
        predicted = predicted.reshape(-1)
    if len(etalon.shape) > 1:
        etalon = etalon.reshape(-1)
    
    class_report = classification_report(etalon, predicted)
    cf_matrix = confusion_matrix(etalon, predicted)

    return {'report': class_report,
        'cf_matrix': cf_matrix,
        }

def cut_etalons(predicted, etalons, force=False):
    # etalons = glob.glob(os.path.join(etalons, '*.tif'))
    cropped_dir = DEFAULT_PATH['cropped_etalons']

    cropped_etalons = []
    for etalon in etalons:
        name = os.path.basename(etalon)
        out = os.path.join(cropped_dir, name)
        cropped_etalons.append(out)
        if not os.path.exists(out) or force:
            print(etalon, predicted)
            cut_tif_by(etalon, predicted, out, mode='mode')
        else:
            print("Etalons will be loaded from cache.")

    return cropped_etalons


def validate_how_tif(predicted_path:str, etalons_path:str, force=False, verbose=False):
    predicted = load_tif(predicted_path, only_first=True, verbose=False)
    etalons = cut_etalons(predicted, etalons_path, force=force)

    if len(etalons) < 1:
        print("Etalons not found.")
        return

    predicted_array = predicted['array'] # type: ignore
    res = {}
    for etalon in etalons:
        etalon_array = load_tif(etalon, only_first=True)['array'] # type: ignore
        r = validate_how_array(predicted_array, etalon_array)
        res[os.path.basename(etalon)] = r

    return res


def create_diff_map(predicted, etalon, mode='positive', force=False):
    predicted = load_tif(predicted, only_first=True)
    etalon = cut_etalons(predicted, etalon, force=force)[0]
    etalon = load_tif(etalon, only_first=True)

    pred_arr = predicted['array']
    etalon_arr = etalon['array']
    
    if mode == 'positive':
        predicted['array'] = np.where(pred_arr == etalon_arr, pred_arr, 0)
    elif mode == 'negative':
        predicted['array'] = np.where(pred_arr != etalon_arr, pred_arr, 0)
    print(predicted['array'].shape)
    
    out_name = DEFAULT_PATH['output'] + mode + '_' + os.path.basename(predicted['path'])
    save_tif(predicted, out_name, color_palette=color_palette)
