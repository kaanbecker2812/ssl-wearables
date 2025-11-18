import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly, resample
import pandas as pd
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import pathlib

DATAFILES = '/dss/dsshome1/06/ge38qav/DATA/raw/labeled/full*.csv' #'/Users/kaanbecker/Documents/VSCode/EASE_pretrained_models/DATA/raw/labeled_copy/full*.csv'
TARGET_FS = 1000
WIN_SEC   = 0.120   # window length
HOP_SEC   = 0.090  # 50% overlap
WIN_SAMP  = int(round(WIN_SEC * TARGET_FS))  # 120
HOP_SAMP_FLOAT = HOP_SEC * TARGET_FS         # 90.0
IMUS_ACC = ['Sig_IMU_RL_Acc_', 'Sig_IMU_LL_Acc_', 
            'Sig_IMU_LU_Acc_', 'Sig_IMU_RU_Acc_',
            'Sig_IMU_Back_Acc_']

def to_g(x, units="ms2"):
    if units == "ms2":
        return x.astype(np.float32) / 9.81
    elif units == "g":
        return x.astype(np.float32)
    else:
        raise ValueError("units must be 'ms2' or 'g'")

def resample_float(sig, fs_raw, fs_tgt=TARGET_FS):
    """Anti-aliased resampling using resample_poly with rational ratio."""
    frac = Fraction(fs_tgt, fs_raw).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    samples_out = int(np.ceil(len(sig) * up / down))
    return resample(sig.astype(np.float32), samples_out)

def resample_labels_nearest(labels, fs_raw, n_out, fs_tgt=TARGET_FS):
    """Nearest-neighbor label resample (no smoothing)."""
    t_out = np.arange(n_out) / fs_tgt
    idx = np.clip(np.round(t_out * fs_raw).astype(int), 0, len(labels) - 1)
    return labels.iloc[idx].astype(np.int64).values

def window_majority(lbl_win):
    # majority vote; tie-breaker = last sample
    vals, counts = np.unique(lbl_win, return_counts=True)
    winner = vals[counts == counts.max()]
    return int(winner[-1])  # pick last in case of tie

def cut_force_peaks(df, force_peak_label=9, active_labels=[1,2,3]):
    if force_peak_label not in df['labels'].values:
        return df
    force_peaks = df.index[df['labels'] == force_peak_label].to_list()
    active_indices = df.index[df['labels'].isin(active_labels)].to_list()
    if not active_indices:
        df['labels'] = 0
        return df
    first_active, last_active = active_indices[0], active_indices[-1]
    # cut peaks outside active range
    peaks_before = [p for p in force_peaks if p < first_active]
    peaks_after = [p for p in force_peaks if p > last_active]

    cut_start = max(peaks_before) + 1 if peaks_before else 0
    cut_end = min(peaks_after) if peaks_after else len(df)

    return df.iloc[cut_start:cut_end]


def make_windows(ax, ay, az, labels, fs_raw, pid, units="ms2"):
    # 1) units -> g
    ax, ay, az = [to_g(a, units=units) for a in (ax, ay, az)]

    # 2) resample accel to 30 Hz
    ax30 = resample_float(ax, fs_raw)
    ay30 = resample_float(ay, fs_raw)
    az30 = resample_float(az, fs_raw)

    # 3) resample labels (nearest) to the same length
    n_out = len(ax30)
    lab30 = resample_labels_nearest(labels, fs_raw, n_out)

    # 4) sliding windows with overlap
    starts = []
    s = 0.0
    while int(np.floor(s)) + WIN_SAMP <= n_out:
        starts.append(int(np.floor(s)))
        s += HOP_SAMP_FLOAT

    Xs, Ys, PIDs = [], [], []
    for st in starts:
        sl = slice(st, st + WIN_SAMP)
        Xs.append(np.stack([ax30[sl], ay30[sl], az30[sl]], axis=0))  # [3,15]
        Ys.append(window_majority(lab30[sl]))
        PIDs.append(int(pid))
    return np.stack(Xs).astype(np.float32), np.array(Ys, np.int64), np.array(PIDs, np.int64)

def plot_Z_with_raw_and_window_labels(
    df_raw: pd.DataFrame,
    z_col: str = "Z",
    labels_raw: np.ndarray | None = None,   # per-sample labels aligned to df_raw length
    labels_win: np.ndarray | None = None,   # per-window labels (ints)
    win_sec: float = 0.5,
    hop_sec: float | None = None,           # if None -> 50% overlap
    starts_30hz: np.ndarray | None = None,  # per-window start indices at 30 Hz
    fs_raw: float | None = None,            # required if no numeric time column / DatetimeIndex
    class_names: dict | None = None,        # {id: "name"}
    class_colors: dict | None = None        # {id: "#RRGGBB"}
):
    assert z_col in df_raw.columns, f"{z_col=} not found"

    # ---- build raw time base (seconds) ----
    if "time" in df_raw.columns and np.issubdtype(df_raw["time"].dtype, np.number):
        t_raw = df_raw["time"].to_numpy(dtype=float)
    elif isinstance(df_raw.index, pd.DatetimeIndex):
        t0 = df_raw.index[0]
        t_raw = (df_raw.index - t0).total_seconds().to_numpy()
    elif fs_raw is not None:
        t_raw = np.arange(len(df_raw), dtype=float) / float(fs_raw)
    else:
        raise ValueError("Provide a numeric 'time' column, a DatetimeIndex, or fs_raw.")

    z = df_raw[z_col].to_numpy(dtype=float)
    N = len(z)

    # ---- window centers (seconds) & per-sample window labels (for coloring the line) ----
    label_per_sample = None
    if labels_win is not None:
        if starts_30hz is not None:
            starts_sec = np.asarray(starts_30hz, dtype=float) / 30.0
        else:
            assert hop_sec is not None, "Provide hop_sec if starts_30hz is None."
            starts_sec = np.arange(len(labels_win), dtype=float) * float(hop_sec)
        centers_sec = starts_sec + win_sec / 2.0

        # map each raw sample to nearest window center
        idx_right = np.searchsorted(centers_sec, t_raw, side="left")
        idx_right = np.clip(idx_right, 0, len(centers_sec) - 1)
        idx_left = np.clip(idx_right - 1, 0, len(centers_sec) - 1)
        nearer_left = (np.abs(t_raw - centers_sec[idx_left]) <=
                       np.abs(t_raw - centers_sec[idx_right]))
        nearest = np.where(nearer_left, idx_left, idx_right)
        label_per_sample = np.asarray(labels_win, dtype=int)[nearest]

    # ---- set up colors for all classes present ----
    present_classes = set()
    if labels_raw is not None: present_classes |= set(np.unique(labels_raw.astype(int)))
    if label_per_sample is not None: present_classes |= set(np.unique(label_per_sample))
    if not present_classes: present_classes = {0}

    if class_colors is None:
        base = ['#808080','#1f77b4','#ff7f0e','#2ca02c',
                '#d62728','#9467bd','#8c564b','#e377c2',"#050000"]
        class_colors = {int(c): base[int(c) % len(base)] for c in sorted(present_classes)}

    # ---- figure layout ----
    import matplotlib.gridspec as gridspec
    rows = 2 if labels_win is None else 3  # top: signal; bottom(s): label bands
    fig = plt.figure(figsize=(13, 6 if rows==3 else 5))
    gs = gridspec.GridSpec(rows, 1, height_ratios=[3] + [1]*(rows-1), hspace=0.15)

    # ---- top: colored Z signal by window labels (if provided), else plain line ----
    ax_sig = fig.add_subplot(gs[0, 0])
    if label_per_sample is None:
        ax_sig.plot(t_raw, z, lw=1.0, label=z_col)
    else:
        # color the line segment-by-segment
        pts = np.column_stack([t_raw, z]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        colors = [class_colors[int(c)] for c in label_per_sample[:-1]]
        lc = LineCollection(segs, colors=colors, linewidths=1.0)
        ax_sig.add_collection(lc)
        ax_sig.set_xlim(t_raw[0], t_raw[-1])
        y_min, y_max = np.nanmin(z), np.nanmax(z)
        pad = 0.05 * (y_max - y_min + 1e-9)
        ax_sig.set_ylim(y_min - pad, y_max + pad)
    ax_sig.set_ylabel(z_col)
    ax_sig.set_xticklabels([])

    # Legend (class colors)
    if class_names:
        handles = [mpatches.Patch(color=class_colors[k],
                   label=class_names.get(k, str(k))) for k in sorted(class_colors)]
        ax_sig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9)

    # ---- build a ListedColormap for label bands ----
    max_id = max(class_colors)
    color_list = [class_colors.get(i, "#ffffff") for i in range(max_id+1)]
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(np.arange(max_id+2)-0.5, cmap.N)

    # ---- bottom: ORIGINAL labels band ----
    if labels_raw is not None:
        ax_raw = fig.add_subplot(gs[1, 0 if rows==2 else 0])
        # draw a 1-pixel-high colored band spanning time
        lab = np.asarray(labels_raw, dtype=int)
        if len(lab) != N:
            raise ValueError("labels_raw length must equal df_raw length.")
        ax_raw.imshow(lab[np.newaxis, :],
                      aspect="auto",
                      extent=[t_raw[0], t_raw[-1], 0, 1],
                      cmap=cmap, norm=norm, interpolation="nearest")
        ax_raw.set_yticks([])
        ax_raw.set_ylabel(f"Original\nlabels\nsamples: {N}", rotation=0, labelpad=40, va='center')
        if rows == 2:
            ax_raw.set_xlabel("Time (s)")
        else:
            ax_raw.set_xticklabels([])

    # ---- optional extra band: WINDOW labels mapped to raw (comparison) ----
    if rows == 3:
        ax_win = fig.add_subplot(gs[2, 0])
        lab_win_samples = label_per_sample
        ax_win.imshow(lab_win_samples[np.newaxis, :],
                      aspect="auto",
                      extent=[t_raw[0], t_raw[-1], 0, 1],
                      cmap=cmap, norm=norm, interpolation="nearest")
        ax_win.set_yticks([])
        ax_win.set_ylabel(f"Window\nlabels\nsamples: {len(labels_win)}", rotation=0, labelpad=40, va='center')
        ax_win.set_xlabel("Time (s)")

    plt.tight_layout()
    return fig

# --------- EXAMPLE BATCH CONVERSION ---------
# trials = list of dicts: {'ax':..., 'ay':..., 'az':..., 'labels':..., 'fs':..., 'pid':..., 'units': 'ms2' or 'g'}
def convert_trials(root_folder, save_root='data/downstream/EASE_1000Hz_120w_25s', plot=False):
    X_trial, Y_trial, PID_trial = [], [], []
    for datafile in tqdm(glob.glob(root_folder)):
        data = pd.read_csv(datafile) #, index_col='Time')
        if 9 in data['labels'].values:
            print(f"Warning: Found '9' labels in {datafile} for {imu_acc}, removing those windows.")
        data = cut_force_peaks(data, force_peak_label=9, active_labels=[1,2,3])
        assert not data['labels'].isna().any(), f"NaN labels found in {datafile}!"
        assert data['labels'].unique().max() != 9, f"'9' labels still present in {datafile} after cutting!"
        X_all = []
        y_old = None
        for imu_acc in IMUS_ACC:
            # special case: LL missing in some trials
            if f'{imu_acc}y' not in data.columns:
                print(f"Warning: Missing LL IMU in {datafile}, filling with zeros.")
                data[f'{imu_acc}y'] = 0.0
            X, Y, PID = make_windows(data[f'{imu_acc}x'], data[f'{imu_acc}y'], data[f'{imu_acc}z'],
                                    data['labels'], 1000, datafile.split('/')[-1].split('_')[2],
                                    units=data.get('units','ms2'))

            if y_old is not None:
                assert np.array_equal(Y, y_old), "Label mismatch between IMU sensors!"
            y_old = Y
            X_all.append(X)
        X_trial.append(np.concatenate(X_all, axis=1))
        Y_trial.append(Y)
        PID_trial.append(PID)

        if plot:
            print(f'Number of windows per class: {np.unique(Y, return_counts=True)}')
            fig = plot_Z_with_raw_and_window_labels(
                                                    data, z_col="Z",
                                                    labels_raw=data['labels'],
                                                    labels_win=Y,
                                                    win_sec=WIN_SEC,
                                                    hop_sec=HOP_SEC,
                                                    fs_raw=1000,
                                                    class_names={0:"idle",1:"lifting",2:"lowering",3:"carrying"}
                                                )
            plt.show()
                
    X = np.concatenate(X_trial, axis=0)
    Y = np.concatenate(Y_trial, axis=0)
    pid = np.concatenate(PID_trial, axis=0)

    os.makedirs(save_root, exist_ok=True)
    np.save(f"{save_root}/X.npy", X)
    np.save(f"{save_root}/Y.npy", Y)
    np.save(f"{save_root}/pid.npy", pid)
    print("Saved:", X.shape, Y.shape, pid.shape)
    return save_root

if __name__ == "__main__":
    save_root=f'data/downstream/EASE_{TARGET_FS}Hz_{WIN_SEC}w_{HOP_SEC}s'
    out_dir = convert_trials(root_folder=DATAFILES, save_root=save_root, plot=False)
    print("Data written to:", out_dir)
