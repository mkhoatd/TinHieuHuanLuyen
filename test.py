import scipy as sp

# Create input of real sine wave
fs = 1.0
fc = 0.25
n = sp.arange(0, 300)
x = sp.cos(2*sp.pi*n*fc/fs)


# Rearrange x into 10 30 second windows
x = sp.reshape(x, (-1, 30))

# Calculate power over each window [J/s]
p = sp.sum(x*x, 1)/x.size

# Calculate energy [J = J/s * 30 second]
e = p*x.size