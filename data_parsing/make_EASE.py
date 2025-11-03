import re
import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.signal import resample_poly

DEVICE_HZ = 1000  # Hz
TARGET_FS = 30  # Hz

WINDOW_SEC = 0.5  # seconds
WINDOW_OVERLAP_SEC = 0.25  # seconds

WINDOW_LEN = int(TARGET_FS * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(TARGET_FS * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks

DATAFILES = '/Users/kaanbecker/Documents/VSCode/EASE_pretrained_models/ssl_wearables/ssl-wearables/raw_data/labeled/full*.csv'
OUTDIR = '/Users/kaanbecker/Documents/VSCode/EASE_pretrained_models/ssl_wearables/ssl-wearables/data/downstream/EASE_1000Hz_120w_25s'

IMUS_ACC = ['Sig_IMU_RL_Acc_',                          # 'Sig_IMU_LL_Acc_', 
            'Sig_IMU_LU_Acc_', 'Sig_IMU_RU_Acc_',
            'Sig_IMU_Back_Acc_']

IMUS_GYRO = ['Sig_IMU_LL_AngVel_', 'Sig_IMU_RL_AngVel_', 
             'Sig_IMU_LU_AngVel_','Sig_IMU_RU_AngVel_',
             'Sig_IMU_Back_AngVel_']

X, Y, T, P, = [], [], [], []

for datafile in tqdm(glob.glob(DATAFILES)):
    data = pd.read_csv(datafile, parse_dates=['Time'], index_col='Time')

    up, down = TARGET_FS, int(fs)
    res_data = pd.DataFrame()

    for imu_acc in IMUS_ACC:
        for axis in ['x', 'y', 'z']:
            col = imu_acc + axis
            res_data[col] = resample_poly(data[col], up, down)
    data = res_data

    p = datafile.split('/')[-1].split('_')[2]

    for i in range(0, len(data), WINDOW_STEP_LEN):
        w = data.iloc[i:i + WINDOW_LEN]

        t = w.index[0].to_datetime64()
        x = w[['x', 'y', 'z']].values
        y = annolabel.loc[w['annotation'][0], LABEL]

        X.append(x)
        Y.append(y)
        T.append(t)
        P.append(p)

X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)

os.system(f'mkdir -p {OUTDIR}')
np.save(os.path.join(OUTDIR, 'X'), X)
np.save(os.path.join(OUTDIR, 'Y'), Y)
np.save(os.path.join(OUTDIR, 'time'), T)
np.save(os.path.join(OUTDIR, 'pid'), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:")
print(pd.Series(Y).value_counts())
