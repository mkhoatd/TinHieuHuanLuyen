# %%
import numpy as np
import librosa
import plotly.graph_objects as go


# speech-silence and voice-unvoiced
def read_lab(lab_file_name: str):
    data = []
    with open(lab_file_name) as f:
        for line in f.readlines():
            data.append(line.split())
    return data


def get_closest(array_time, values):
    array_time = np.array(array_time)
    values = np.array(values, dtype=np.float64)
    idx = np.searchsorted(array_time, values)
    idx = np.array(idx)
    idx[array_time[idx] - values > np.diff(array_time).mean() * 0.5] -= 1
    return array_time[idx]


def get_closest_idx(array_time, values):
    array_time = np.array(array_time)
    values = np.array(values, dtype=np.float64)
    idx = np.searchsorted(array_time, values, side='left')
    idx = np.array(idx)   # idx[array_time[idx] - values > np.diff(array_time).mean() * 0.5] -= 1
    return idx


def array_norm(arr: np.ndarray):
    arr = np.copy(arr)
    max_arr = max(arr)
    min_arr = min(arr)
    return (arr - min_arr) / (max_arr - min_arr)

def array_norm_by_T(arr: np.ndarray, T):
    min_arr=min(arr)
    max_arr=max(arr)
    return np.where(arr>=T, (arr-T)/(max_arr-T), (arr-T)/(T-min_arr))

def remove_sl(array: np.ndarray, timestamp_label, t):
    new_array = np.array([])
    for line in timestamp_label:
        if line[2] != 'sil':
            try:
                idx1 = int(get_closest_idx(t, line[0]))
                idx2 = int(get_closest_idx(t, line[1]))
                new_array = np.append(new_array, array[idx1:idx2])
            except:
                print(line)
    return new_array


def load_data(audio_name: str):
    signal, sr = librosa.load(f'{audio_name}.wav')
    lab_data = read_lab(f'{audio_name}.lab')
    mean_std = lab_data[-2:]
    lab_data = lab_data[:-2]
    timestamp_label = list(filter(lambda x: (x[2] == 'v' or x[2] == 'uv'), lab_data))
    t_i = 0
    t_f = signal.shape[0] / sr
    t = np.linspace(t_i, t_f, num=signal.shape[0])
    t_without_sl = remove_sl(t, timestamp_label, t)
    signal = remove_sl(signal, timestamp_label, t)
    return signal, sr, t_without_sl, timestamp_label


def separate_frames(signal, sr, t, frame_length=0.02):
    frame_size = int(sr * frame_length)
    frame_count = len(signal) // frame_size
    temp = 0
    signal_frames = []
    time_frames = []
    for i in range(0, frame_count * frame_size, frame_size):
        signal_frames.append(signal[i:i + frame_size])
        time_frames.append(t[i:i + frame_size])
    return np.array(signal_frames), np.array(time_frames), frame_size, frame_count


def calc_STE(signal_frames):
    STE = []
    frame_size=len(signal_frames[0])
    frames_count=len(signal_frames)
    for i in range(frames_count):
        value = np.sum(np.square(signal_frames[i])) / np.ones(frame_size)
        STE.append(value)
    STE = np.array(STE)

    STE = STE.reshape(-1)
    STE = array_norm(STE)
    return STE


def calc_ZCR(signal_frames):
    ZCR = []
    frames_count=len(signal_frames)
    frame_size=len(signal_frames[0])
    for i in range(frames_count):
        value = np.sum(np.abs(np.diff(np.sign(signal_frames[i])))) / np.ones(frame_size)
        ZCR.append(value)
    ZCR = np.array(ZCR)
    ZCR = ZCR.reshape(-1)
    ZCR = array_norm(ZCR)
    return ZCR


def separate_vu(STE, timestamp_label, t):
    STE_v = np.array([])
    STE_uv = np.array([])
    for line in timestamp_label:
        if line[2] == 'v':
            try:
                idx1 = int(get_closest_idx(t, line[0]))
                idx2 = int(get_closest_idx(t, line[1]))
                STE_v = np.append(STE_v, STE[idx1:idx2])
            except:
                print(line)
        if line[2] == 'uv':
            try:
                idx1 = int(get_closest_idx(t, line[0]))
                idx2 = int(get_closest_idx(t, line[1]))
                STE_uv = np.append(STE_uv, STE[idx1:idx2])
            except:
                print(line)
    return np.array(STE_v), np.array(STE_uv)


def calc_T_binsearch(g, f):
    Nf = len(f)
    Ng = len(g)
    Tmax = max(np.max(f), np.max(g))
    Tmin = min(np.min(f), np.min(g))
    T = (Tmax + Tmin) / 2
    p = sum(g > T)
    i = sum(f < T)
    j = -1
    q = -1
    while j != j or p != q:
        if 1 / Nf * np.sum(f[f > T] - T) - 1 / Ng * np.sum(T - g[g < T]) > 0:
            Tmin = T
        else:
            Tmax = T
        T = (Tmax + Tmin) / 2
        j = i
        q = p
        p = sum(g > T)
        i = sum(f < T)
    return T

def jitter_remove(VU_jit, frame_size):
    frames_count=len(VU_jit)//frame_size
    STA_VU=np.array([])
    for i in range(0, frames_count*frame_size, frame_size):
        STA_VU=np.append(STA_VU, np.sum(VU_jit[i:i+frame_size])/np.ones(frame_size))
    STA_VU=1/frame_size*STA_VU
    VU=np.where(STA_VU>=0.5, 1, 0)
    return VU
# %%

def voiced_unvoiced(audio_name: str):
    audio_name_list=['studio_F1', 'studio_M1', 'phone_M1', 'phone_F1']
    signal_list=[0]*len(audio_name_list)
    sr_list=[0]*len(audio_name_list)
    t_list=[0]*len(audio_name_list)
    timestamp_label_list=[0]*len(audio_name_list)
    signal_frames_list=[0]*len(audio_name_list)
    time_frames_list=[0]*len(audio_name_list)
    frame_size_list=[0]*len(audio_name_list)
    frames_count_list=[0]*len(audio_name_list)
    STE_list=[0]*len(audio_name_list)
    ZCR_list=[0]*len(audio_name_list)
    STE_voiced_list=[0]*len(audio_name_list)
    STE_unvoiced_list=[0]*len(audio_name_list)
    ZCR_voiced_list=[0]*len(audio_name_list)
    ZCR_unvoiced_list=[0]*len(audio_name_list)
    for i in range(len(audio_name_list)):
        signal_list[i], sr_list[i], t_list[i], timestamp_label_list[i] = load_data(audio_name_list[i])
        signal_frames_list[i], time_frames_list[i], frame_size_list[i], frames_count_list[i] = separate_frames(signal_list[i], sr_list[i], t_list[i])
        signal_list[i]=signal_list[i][:frames_count_list[i]*frame_size_list[i]]
        t_list[i]=t_list[i][:frames_count_list[i]*frame_size_list[i]]
        STE_list[i] = calc_STE(signal_frames_list[i])
        ZCR_list[i] = calc_ZCR(signal_frames_list[i])
        STE_voiced_list[i], STE_unvoiced_list[i] = separate_vu(STE_list[i], timestamp_label_list[i], t_list[i])
        ZCR_voiced_list[i], ZCR_unvoiced_list[i] = separate_vu(ZCR_list[i], timestamp_label_list[i], t_list[i])
    signal, sr, t, timestamp_label=load_data(audio_name)
    signal_frames, time_frames, frame_size, frames_count = separate_frames(signal, sr, t)
    signal = signal[:frames_count * frame_size]
    t = t[:frames_count * frame_size]
    STE = calc_STE(signal_frames)
    ZCR = calc_ZCR(signal_frames)
    STE_voiced, STE_unvoiced = separate_vu(STE, timestamp_label, t)
    ZCR_voiced, ZCR_unvoiced = separate_vu(ZCR, timestamp_label, t)
    T_STE = calc_T_binsearch(STE_voiced, STE_unvoiced)
    T_ZCR = calc_T_binsearch(ZCR_voiced, ZCR_unvoiced)
    STE_norm_T = array_norm_by_T(STE, T_STE)
    ZCR_norm_T = array_norm_by_T(ZCR, T_ZCR)
    VU_jit = np.sign(STE_norm_T - ZCR_norm_T)
    VU = jitter_remove(VU_jit, frame_size)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal, name='signal'))
    # fig.add_trace(go.Scatter(x=t, y=STE_norm_T, name='STE', line=dict(color='firebrick', width=1, dash='dot')))
    # fig.add_trace(go.Scatter(x=t, y=ZCR_norm_T, name='ZCR'))
    fig.add_trace(go.Scatter(x=t, y=VU, name='VU'))
    # fig.add_trace(go.Scatter(x=t, y=VU_jit, name='VU_jit'))
    for i in range(len(timestamp_label)):
        color = 'green' if timestamp_label[i][2] == 'v' else 'blue'
        fig.add_vrect(x0=float(timestamp_label[i][0]), x1=float(timestamp_label[i][1]),
                    fillcolor=color,
                    opacity=0.25, line_width=0)
    fig.update_layout(
        title=audio_name
    )
    fig.show()
# audio_name = 'phone_F1'
# signal, sr, t, timestamp_label = load_data(audio_name)
# signal_frames, time_frames, frame_size, frames_count = separate_frames(signal, sr, t)
# signal = signal[:frames_count * frame_size]
# t = t[:frames_count * frame_size]

# # %%
# # Calculate STE
# STE = calc_STE(signal_frames)
# ZCR = calc_ZCR(signal_frames)
# # %%
# STE_voiced, STE_unvoiced = separate_vu(STE, timestamp_label, t)
# ZCR_voiced, ZCR_unvoiced = separate_vu(ZCR, timestamp_label, t)
# # %%
# T_STE = calc_T_binsearch(STE_voiced, STE_unvoiced) / np.ones(len(t))
# T_ZCR = calc_T_binsearch(ZCR_voiced, ZCR_unvoiced) / np.ones(len(t))
# #%%
# STE_norm_T=array_norm_by_T(STE, T_STE)
# ZCR_norm_T=array_norm_by_T(ZCR, T_ZCR)
# VU_jit=np.sign(STE_norm_T-ZCR_norm_T)
# VU=jitter_remove(VU_jit, int(sr*0.02))
# %%
voiced_unvoiced('phone_F1')

# %%
