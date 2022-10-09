# %%
import numpy as np
import plotly.graph_objects as go
import librosa

# speech-silence and voice-unvoiced
audio_name='studio_M1'
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
    idx=np.array(idx)
    idx[array_time[idx] - values > np.diff(array_time).mean() * 0.5] -= 1
    return array_time[idx]


def get_closest_idx(array_time, values):
    array_time = np.array(array_time)
    values = np.array(values, dtype=np.float64)
    idx = np.searchsorted(array_time, values)
    idx=np.array(idx)
    idx[array_time[idx] - values > np.diff(array_time).mean() * 0.5] -= 1
    return idx

def array_norm(arr: np.ndarray):
    arr=np.copy(arr)
    max=np.max(arr)
    min=np.min(arr)
    return (arr-min)/(max-min)


def remove_sl(array:np.ndarray, data):
    array=np.copy(array)
    new_array=np.array([])
    for line in data:
        if line[2]!='sil':
            idx1=int(get_closest_idx(t,line[0]))
            idx2=int(get_closest_idx(t,line[1]))
            new_array=np.append(new_array, array[idx1:idx2])
    return new_array


# %%
signal, sr= librosa.load(f'{audio_name}.wav')
data=read_lab(f'{audio_name}.lab')
mean_std=data[-2:]
data=data[:-2]
data_v_uv=list(filter(lambda x:(x[2]=='v' or x[2]=='uv'), data))
t_i = 0
t_f = signal.shape[0] / sr
t = np.linspace(t_i, t_f, num=signal.shape[0])
t_without_sl=remove_sl(t, data)
signal_without_sl=remove_sl(signal, data)
# fig=px.line(x=t, y=signal)
# fig.update_layout(title="signal with slience")
# fig.show()
# fig=px.line(x=t_without_sl, y=signal_without_sl)
# fig.update_layout(title="signal without slience")
# fig.show()
# %%
# Calculate STE
window_len=int(sr*0.01)
STE=np.array([np.sum(1/window_len*np.square(signal[i:i+window_len]))/np.ones(window_len) 
              for i in range(0, len(signal), window_len)])
STE=STE.reshape(-1)
STE_without_sl_for_plot=np.array([np.sum(1/window_len*np.square(signal_without_sl[i:i+window_len]))/np.ones(window_len) 
                        for i in range(0, len(signal_without_sl), window_len)])
STE_without_sl=np.array([np.sum(1/window_len*np.square(signal_without_sl[i:i+window_len]))/np.ones(window_len) 
                        for i in range(0, len(signal_without_sl), window_len)])=
STE_without_sl=STE_without_sl.reshape(-1)
STE_norm=array_norm(STE)
STE_without_sl_norm=array_norm(STE_without_sl)

# fig=go.Figure(data=[go.Histogram(x=STE_without_sl_norm)])
# fig.show()
# %%
# Calculate spectral centroid
# C=np.array([np.sum([(k+1)*signal[k] for k in range(i, i+window_len)])/np.sum(signal[])/np.ones(window_len) 
#             for i in range(0, len(signal), window_len)])
# C_without_sl=np.array([np.sum([(k+1)*signal[k] for k in range(i, i+window_len)])/np.sum([signal[k] for k in range(i, i+window_len)])/np.ones(window_len) 
#             for i in range(0, len(signal), window_len)])

# %%
# Calculate MA
MA=np.array([np.sum(np.abs(signal[i:i+window_len]))/np.ones(window_len) for i in range(0, len(signal), window_len)])
MA=MA.reshape(-1)
MA_norm=array_norm(MA)
# %%


# %%
# %%
def outputT(STE):
    w=1
    Hist_Ste, X_Ste = np.histogram(STE, len(STE))
    max1=Hist_Ste[1]
    max1Index=1
    max2=0
    max2Index=0
    for i in range(1, len(Hist_Ste)):
        if Hist_Ste[1]>max1:
            max2=max1
            max2Index=max1Index
            max1=Hist_Ste[i]
            max1Index=i
        if Hist_Ste[i]<max1 and Hist_Ste[i]>max2:
            max2=Hist_Ste[i]
            max2Index=i
    max1=X_Ste[max1Index]
    max2=X_Ste[max2Index]
    T=(w*max1+max2)/(w+1)
    return T, Hist_Ste, X_Ste
a,cnts,bins=outputT(STE_without_sl)

a=a/np.ones(len(t))
# %%
fig=go.Figure()
# fig.add_trace(go.Scatter(x=t, y=signal, name='signal'))
fig.add_trace(go.Scatter(x=t, y=STE, name='STE', line=dict(color='firebrick', width=1, dash='dot')))
# fig.add_trace(go.Scatter(x=t, y=MA, name='MA', line=dict(color='royalblue', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=t, y=a, name='T'))
for i in range(len(data_v_uv)):
    color='green' if data_v_uv[i][2]=='v' else 'blue'
    fig.add_vrect(x0=float(data_v_uv[i][0]), x1=float(data_v_uv[i][1]),
                  fillcolor=color,
                  opacity=0.25, line_width=0)
fig.update_layout(
    title=audio_name
)
fig.show()
# %%