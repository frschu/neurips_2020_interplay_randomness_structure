import numpy as np

# Path for figures
figure_path = "./figures"

# Width of figures (full page)
fig_width = 8
gr = 1.618
fig_height = fig_width / gr
# Fontsize
fs = 12
# Line width, marker size
lw = 2
ms = 4

# Relative loss for convergence time
rel_loss_c = 0.05

# Alphabetic labels for plots
flbs = ["(%s)"%s for s in "abcdefghijkl"] 

# Color for legend
c_leg = '0.5'

# Colors
tango = np.array([
    [252, 233,  79], #Butter 1
    [237, 212,   0], #Butter 2
    [196, 160,   0], #Butter 3
    [252, 175,  62], #Orange 1
    [245, 121,   0], #Orange 2
    [206,  92,   0], #Orange 3
    [233, 185, 110], #Chocolate 1
    [193, 125,  17], #Chocolate 2
    [143,  89,   2], #Chocolate 3
    [138, 226,  52], #Chameleon 1
    [115, 210,  22], #Chameleon 2
    [ 78, 154,   6], #Chameleon 3
    [114, 159, 207], #Sky Blue 1
    [ 52, 101, 164], #Sky Blue 2
    [ 32,  74, 135], #Sky Blue 3
    [173, 127, 168], #Plum 1
    [117,  80, 123], #Plum 2
    [ 92,  53, 102], #Plum 3
    [239,  41,  41], #Scarlet Red 1
    [204,   0,   0], #Scarlet Red 2
    [164,   0,   0], #Scarlet Red 3
#     [238, 238, 236], #Aluminium 1
    [211, 215, 207], #Aluminium 2
#     [186, 189, 182], #Aluminium 3
    [136, 138, 133], #Aluminium 4
    [ 85,  87,  83], #Aluminium 5
#     [ 46,  52,  54], #Aluminium 6
]) / 255
# Sort:
# blue, orange, green, red, violet, butter, alum, choc
sorter = np.array([4, 1, 3, 6, 5, 0, 7, 2])
tango = tango.reshape((-1, 3, 3))[sorter].reshape((-1, 3))

tango_0 = tango[0::3]
tango_1 = tango[1::3]
tango_2 = tango[2::3]
colors = tango_2

# Colors for different g
cs = tango.copy().reshape((-1, 3, 3))[:, ::-1]
# Blues
# cs[0, 0] = np.array([ 32,  74, 135]) / 255 # Sky Blue 3
cs[0, 0] = np.array([ 17,  54, 110]) / 255 # Sky Blue 3 -> darker
# cs[0, 1] = np.array([ 52, 101, 164]) / 255 # Sky Blue 2
cs[0, 1] = np.array([ 55, 108, 178]) / 255 # Sky Blue 2 -> lighter
cs[0, 2] = np.array([114, 159, 207]) / 255 # Sky Blue 1
# Oranges
# cs[1, 0] = np.array([206,  92,   0]) / 255 # Orange 3
cs[1, 0] = np.array([156,  73,   0]) / 255 # Orange 3 -> darker
cs[1, 1] = np.array([245, 121,   0]) / 255 # Orange 2
cs[1, 2] = np.array([252, 175,  62]) / 255 # Orange 1
# Greens
# cs[2, 0] = np.array([ 78, 154,   6]) / 255 # Chameleon 3
# cs[2, 0] = np.array([ 40,  80,   6]) / 255 # Chameleon 3 -> make a bit darker
cs[2, 0] = np.array([ 30,  70,   0]) / 255 # Chameleon 3 -> make a bit darker
# cs[2, 1] = np.array([115, 210,  22]) / 255 # Chameleon 2
# cs[2, 1] = np.array([ 78, 154,   6]) / 255 # use Chameleon 1 instead
cs[2, 1] = np.array([ 64, 128,   8]) / 255 # use Chameleon 1 instead
cs[2, 2] = np.array([138, 226,  52]) / 255 # Chameleon 1 

# Adapt
colors[:3] = cs[:3, 1]
