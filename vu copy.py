# %%
import librosa.display
import librosa
import numpy as np
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


# %%
print(read_lab('./studio_F2.lab'))

# %%
signal, sr= librosa.load('./phone_F2.wav')
# librosa.display.waveshow(signal, sr=sr)
t_i = 0
t_f = signal.shape[0] / sr
t = np.linspace(t_i, t_f, num=signal.shape[0])
fig=px.line(x=t, y=signal)
fig.show()
# %%



# %%
samples_per_milis=sr/100


# e = STE(phone_f2[0])
# fig, ax = plt.subplots(1, 1)
# ax.plot(t[10000:11000], phone_f2[0][10000:11000])
# ax.plot(t[50000:70000], e[50000:70000], ':r')
# ax.plot(t, e, ':b')

# plt.show()



# %%
