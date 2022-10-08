# %%
import librosa.display
import librosa
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# speech-silence and voice-unvoiced
def read_lab(lab_file_name: str):
    data = []
    with open(lab_file_name) as f:
        for line in f.readlines():
            data.append(line.split())
    return data


def get_closest(array, values):
    array = np.array(array)
    idx = np.searchsorted(array, values)
    idx[array[idx] - values > np.diff(array).mean() * 0.5] -= 1
    return array[idx]


def array_norm(arr: np.ndarray):
    arr=np.copy(arr)
    max=np.max(arr)
    min=np.min(arr)
    return (arr-min)/(max-min)


# %%
signal, sr= librosa.load('./studio_F2.wav')
# librosa.display.waveshow(signal, sr=sr)
t_i = 0
t_f = signal.shape[0] / sr
t = np.linspace(t_i, t_f, num=signal.shape[0])
# fig=px.line(x=t, y=signal)
# fig.show()
# %%
# Calculate STE
window_len=490
STE=np.array([np.sum(np.square(signal[i:i+window_len]))/np.ones(window_len) for i in range(0, len(signal), window_len)])
STE=STE.reshape(-1)
STE_norm=array_norm(STE)
counts, bins=np.histogram(STE, bins=len(t)/window_len)
bins=.5*(bins[:-1]+bins[1:])
fig=px.bar(x=bins, y=counts)
fig.show(renderer='browser')

# %%
# Calculate MA
MA=np.array([np.sum(np.abs(signal[i:i+window_len]))/np.ones(window_len) for i in range(0, len(signal), window_len)])
MA=MA.reshape(-1)
MA_norm=array_norm(MA)
# %%


# %%
data=read_lab('./studio_F2.lab')
mean_std=data[-2:]
data=data[:-2]
data_v_uv=list(filter(lambda x:(x[2]=='v' or x[2]=='uv'), data))
print(data_v_uv)
# %%
def outputT(STE):
    w=10000000
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
    return T
a=outputT(STE)
a=a/np.ones(len(STE))
# %%
fig=go.Figure()
fig.add_trace(go.Scatter(x=t, y=signal, name='signal'))
fig.add_trace(go.Scatter(x=t, y=STE_norm, name='STE'))
fig.add_trace(go.Scatter(x=t, y=MA_norm, name='MA'))
fig.add_trace(go.Scatter(x=t, y=a, name='T'))
for i in range(len(data_v_uv)):
    color='green' if data_v_uv[i][2]=='v' else 'blue'
    fig.add_vrect(x0=float(data_v_uv[i][0]), x1=float(data_v_uv[i][1]),
                  fillcolor=color,
                  opacity=0.25, line_width=0)
fig.update_layout(
    title="Phone_F2"
)
fig.show(renderer="browser")
# %%
