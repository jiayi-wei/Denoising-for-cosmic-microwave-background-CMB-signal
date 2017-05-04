import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

model = 'new_model'
name = model + '_loss.txt'
with open(name, 'r') as line:
	line_info = line.readlines()

x = np.array(line_info, dtype='|S4')
y = x.astype(np.float)

l = np.arange(len(y))

plt.plot(l, y)
plt.axis([0, 6500, 0, 100000])
plt.show()
plt.savefig(model + '.png')
