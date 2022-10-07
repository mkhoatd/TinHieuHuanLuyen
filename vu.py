# %%
import librosa.display
import librosa
import numpy as np
import plotly.graph_objects as go

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
STE=array_norm(STE)


# %%
# Calculate MA
MA=np.array([np.sum(np.abs(signal[i:i+window_len]))/np.ones(window_len) for i in range(0, len(signal), window_len)])
MA=MA.reshape(-1)
MA=array_norm(MA)
# %%
fig=go.Figure()
fig.add_trace(go.Scatter(x=t, y=signal, name='signal'))
fig.add_trace(go.Scatter(x=t, y=STE, name='STE'))
fig.add_trace(go.Scatter(x=t, y=MA, name='MA'))
fig.update_layout(
    title="Phone_F2"
)
fig.show()

# %%
data=read_lab('./phone_F2.lab')
mean_std=data[-2:]
data=data[:-2]
data_v_uv=list(filter(lambda x:(x[2]=='v' or x[2]=='uv'), data))
print(data_v_uv)




# %%
x=data[1]
(x[2]=='v') or (x[2]=='uv')
# %%
