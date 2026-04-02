# Training model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
from prepare_ds import generate_dataset
from utils import DEFAULT_PATH

from joblib import dump, load
from tqdm import tqdm
import itertools
from sklearn.metrics import f1_score
import time


def select_method(method='RF', verbose=True):
    """With teacher: (RF, Bayers...), without teacher (segmentation)"""
    if method == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42,
                                       n_jobs=-1)
    else:
        raise ValueError(f"Method not recognized: {method}")

    if verbose:
        print("Selected model / method:", method)

    return model


def save_model(model, out=None):
    out = out if out is not None else "model.joblib"
    out = DEFAULT_PATH['output'] + out
    dump(model, out)
    print("Model was saved: " + out)


def train_model(data, labels, model='RF', save=True, verbose=True):

    mask = labels > 0
    if verbose:
        print("Split X, y -> X_train, y_train...")
    X_train, X_test, y_train, y_test = train_test_split(data[mask], labels[mask], shuffle=True, test_size=0.2, random_state=42)

    # TODO: develop it.
    if verbose:
        print("Start training model...")
    m = select_method(model, verbose)
    m.fit(X_train, y_train)
    if verbose:
        print("Model was trained. Start validate it...")
    y_pred = m.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    if save:
        if verbose:
            print("Saving model...")
        n = f"tmp_weights_{model}.joblib"
        save_model(m, n)

    return m, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred), f1


def analyse_best_model(config_grid, signs_path, labels_path, store_models=True, verbose=False, force=False):
    """
    datasets: dict with keys as param names and values as lists of possible values.
    Example:
        datasets = {
            'mask_mode': ['random', 'secure', 'homogeneous'],
            'r': [1, 2, 3],  # only for secure mode
            'percent': [0.001, 0.01, 0.1...],
            'homogen_percent': [0.01, 0.1...],
            'resize': ['by_label', 'by_sign'],
            'stratify': [False, True]
        }
    """
    keys = list(config_grid.keys())
    values = list(config_grid.values())
    results = []

    # Generate dataset for current params
    for combo in tqdm(list(itertools.product(*values)), desc="Training set of models..."):
        params = dict(zip(keys, combo))
        mm = params['mask_mode']
        lm = params['layer_mode']
        lt = params['layer_type']
        rr = params['r']
        pp = params['percent']
        rz = params['resize']
        st = params['stratify']
        hp = params['homogen_percent']

        if mm != 'secure' and rr > 1:
            if verbose:
                print(f"Skip {mm} with r={rr}")
            continue

        z_x, z_y, _, _ = generate_dataset(signs_path, labels_path,
            mask_mode=mm,
            layer_mode=lm,
            layer_type=lt,
            r=rr,
            count=pp,
            resize=rz,
            stratify=st,
            feature=hp,
            verbose=verbose,
            force=force)
        _, count_classes = np.unique(z_y, return_counts=True)

        # Train model and measure time
        start_time = time.time()
        model, report, cm, f1 = train_model(z_x, z_y, save=False, verbose=verbose)
        elapsed_time = round(((time.time() - start_time) / 60), 1)

        # Store results as flat dict
        result = {
            'mask_mode': mm,
            'layer_mode': lm,
            'r': rr,
            'percent': pp,
            'homogen_percent': hp,
            'resize': rz,
            'stratify': st,
            'classes': count_classes,
            'report': report,
            'cf_matrix': cm,
            'f1_score': f1,
            'train_time': elapsed_time,
        }
        if store_models:
            result['model'] = model
        results.append(result)



    # Sort results by f1_score descending
    results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    if verbose:
        for r in results:
            i = results.index(r)
            print(f"{i+1}: f1={r['f1_score']:.4f}, mm={r['mask_mode']}, r={r['r']}, p={r['percent']}, rz={r['resize']}, st={r['stratify']}")
    return results
