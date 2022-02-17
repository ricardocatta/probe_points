import matplotlib.pylab as plt
import numpy as np
import pandas as pd

dataframe = pd.read_csv("computacional_std_u.csv")                        # read csv for the datas import of paraview

experimental = pd.read_csv("experimental_x_computational_std_u2.csv")       # read csv for the experimenta data

plt.style.use('ggplot')


x = dataframe.x_comp * (1.0 /0.15)                                # admensionalize the positions
y = dataframe.std_u_comp

x_exp = experimental.x_exp
y_exp = experimental.vel_u_exp

fig = plt.figure(figsize=(10.0, 4.0))
axes1 = fig.add_subplot(1, 1, 1)

axes1.set_ylabel('std_u')                             # gráfico para o valor médio
axes1.set_xlabel('x / L')
x1, x2, y1,y2 = plt.axis()
plt.axis((x1, x2, -0.02 ,0.15))
plt.plot(x, y,'-', label="Caso 1B")
plt.plot(x_exp, y_exp, '*', label="Experimental")

plt.legend(loc='upper right')

fig.tight_layout()

#plt.grid()

plt.show()
