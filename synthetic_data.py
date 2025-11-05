# synthetic_data.py
import numpy as np
import pandas as pd

ACTS = ["SITTING", "WALKING", "JOGGING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "STANDING"]

def gen_window(act, fs=50, seconds=2.56):
    t = np.linspace(0, seconds, int(fs*seconds))
    if act == "SITTING":
        ax = 0.02 * np.random.randn(len(t))
        ay = 0.01 * np.random.randn(len(t))
        az = 1.0 + 0.02 * np.random.randn(len(t))
    elif act == "WALKING":
        freq = 1.8
        ax = 0.3*np.sin(2*np.pi*freq*t) + 0.1*np.random.randn(len(t))
        ay = 0.25*np.cos(2*np.pi*freq*t) + 0.1*np.random.randn(len(t))
        az = 0.9 + 0.15*np.sin(2*np.pi*freq*t + 0.5) + 0.1*np.random.randn(len(t))
    elif act == "JOGGING":
        freq = 3.0
        ax = 0.6*np.sin(2*np.pi*freq*t) + 0.2*np.random.randn(len(t))
        ay = 0.5*np.cos(2*np.pi*freq*t) + 0.2*np.random.randn(len(t))
        az = 1.1 + 0.25*np.sin(2*np.pi*freq*t + 0.2) + 0.2*np.random.randn(len(t))
    elif act == "WALKING_UPSTAIRS":
        ax = 0.35*np.sin(2*np.pi*1.7*t) + 0.15*np.random.randn(len(t))
        ay = 0.25*np.cos(2*np.pi*1.7*t) + 0.12*np.random.randn(len(t))
        az = 1.0 + 0.2*np.sin(2*np.pi*1.7*t + 0.4) + 0.12*np.random.randn(len(t))
    else:
        ax = 0.05*np.random.randn(len(t))
        ay = 0.03*np.random.randn(len(t))
        az = 1.0 + 0.05*np.random.randn(len(t))
    df = pd.DataFrame({
        "timestamp": t,
        "ax": ax,
        "ay": ay,
        "az": az
    })
    return df

def create_synthetic_dataset(n_per_class=200, fs=50, seconds=2.56):
    X_list = []
    y_list = []
    for act in ACTS:
        for _ in range(n_per_class):
            df = gen_window(act, fs=fs, seconds=seconds)
            X_list.append(df)
            y_list.append(act)
    return X_list, y_list
