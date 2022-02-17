import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import statistical_module as sm
import scipy.fft

################################################################
# Ricardo Tadeu Oliveira Catta Preta
# e-mail: ricardocatta@gmail.com
################################################################



def plot_mean_vel(x, vel, dt, rho, dataframe, exp_std_vel):
    """
     Plota o valor médio da velocidade em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - vel = vetor de velocidade com a quantidade tempo por número de probes;
    - dataframe = tabela dos dados coletados pelo paraview;
    - exp_std_vel = tabela com os dados experimentais.

    OUTPUT:

    Retorna o gráfico da velocidade média em função da posição.
    """
    x_paraview = dataframe.x_comp * (1.0 /0.15)
    y_paraview = dataframe.w_average
    
    rho_paraview = dataframe.density_average

    w_paraview = []
    for i in range(29):
        w_paraview.append(np.sum((y_paraview[i] * rho_paraview[i])) / np.sum(rho_paraview[i])) # média de ponderada pelo dt.

    x_exp = exp_std_vel.x1_exp
    y_exp = exp_std_vel.mean_w_exp

    plt.style.use('ggplot')
        
    fig = plt.figure(figsize=(10.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('average_w')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15
    
    dt = np.array(dt)
    rho = np.array(rho)
    y1 = np.array(vel)

    x2 = []
    y2 = []
    #rho = []
    
    for i in range(34):
        x2.append(np.mean(x1[i]))
        #y2.append(np.mean(y1[i]))    # antigo
        y2.append(np.sum((dt[i] * y1[i])) / np.sum(dt[i])) # média de ponderada pelo dt.
        #y2.append(np.sum((dt[i] * y1[i] * rho[i])) / np.sum(dt[i] * rho[i])) # média de favre.

    

    plt.plot(x2, y2 , '-', label="Smag Cs = 0.15 (meu código)")
    plt.plot(x_paraview, w_paraview, '--', label="Smag Cs = 0.15 (paraview)")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def plot_std_ke(x, u, v, w, dataframe, exp_ke_std):
    """
    Plota o desvio padrão da energia cinética em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - u = vetor de velocidade u com a dimensão da quantidade tempo por número de probes;
    - v = vetor de velocidade v com a dimensão da quantidade tempo por número de probes;
    - w = vetor de velocidade w com a dimensão da quantidade tempo por número de probes;
    - dataframe = tabela dos dados coletados pelo paraview;
    - exp_ke_std = tabela com os dados experimentais para energia cinética turbulenta.

    OUTPUT:

    Retorna o desvio padrão da energia cinética turbulenta em função da posição.
    """

    std_u = []
    std_v = []
    std_w = []
    ke = []

    for i in range(34):
        #std_u.append(sm.mean(sm.covariance(u[i],u[i], 3), 3))
        #std_v.append(sm.mean(sm.covariance(v[i],v[i], 3), 3))
        #std_w.append(sm.mean(sm.covariance(w[i],w[i], 3), 3))
        std_u.append(sm.std(u[i], 3))
        std_v.append(sm.std(v[i], 3))
        std_w.append(sm.std(w[i], 3))


    mean_var_u = np.array(std_u)
    mean_var_v = np.array(std_v)
    mean_var_w = np.array(std_w)


    #ke = ((0.5) * np.abs(mean_var_u + mean_var_v + mean_var_w)
    ke = (0.5) * np.abs(mean_var_u ** 2 + mean_var_v ** 2 + mean_var_w ** 2)
    
    x_exp = exp_ke_std.x1_exp
    y_exp = exp_ke_std.std_ke_exp

    x_parav = dataframe.x_comp * (1.0 /0.15)
    y_parav = dataframe.ke_stddev  

    plt.style.use('ggplot')
        
    fig = plt.figure(figsize=(10.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('std_ke')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15

    y1 = np.array(ke)

    x2 = []

    for i in range(34):
        x2.append(np.mean(x1[i]))

    
    plt.plot(x2, y1 , '-', label="Smag Cs = 0.15 (meu código)")
    plt.plot(x_parav, y_parav, '--', label="Smag Cs = 0.15 (paraview)")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    return y1


def plot_ln_ke(x, u, v, w, dataframe, exp_ke_std):
    """
    Plota o logarítimo natural da energia cinética em função do logarítimo natural do tempo.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - u = vetor de velocidade u com a dimensão da quantidade tempo por número de probes;
    - v = vetor de velocidade v com a dimensão da quantidade tempo por número de probes;
    - w = vetor de velocidade w com a dimensão da quantidade tempo por número de probes;
    - dataframe = tabela dos dados coletados pelo paraview;
    - exp_ke_std = tabela com os dados experimentais para energia cinética turbulenta.

    OUTPUT:

    Retorna o desvio padrão da energia cinética turbulenta em função da posição.
    """
    std_u = []
    std_v = []
    std_w = []
    ke = []

    for i in range(34):
        #std_u.append(sm.mean(sm.covariance(u[i],u[i], 3), 3))
        #std_v.append(sm.mean(sm.covariance(v[i],v[i], 3), 3))
        #std_w.append(sm.mean(sm.covariance(w[i],w[i], 3), 3))
        std_u.append(sm.std(u[i], 3))
        std_v.append(sm.std(v[i], 3))
        std_w.append(sm.std(w[i], 3))


    mean_var_u = np.array(std_u)
    mean_var_v = np.array(std_v)
    mean_var_w = np.array(std_w)


    #ke = ((0.5) * np.abs(mean_var_u + mean_var_v + mean_var_w)
    ke = (0.5) * np.abs(mean_var_u ** 2 + mean_var_v ** 2 + mean_var_w ** 2)

    x_exp = exp_ke_std.x1_exp
    y_exp = exp_ke_std.std_ke_exp
    
    plt.style.use('ggplot')
        
    fig = plt.figure(figsize=(10.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('$\ln{E}$')
    axes1.set_xlabel('$\ln{f}$')
                                      # adimensionalizando a coordenada x
    x1 = 1 / np.array(x)  
    x3_delta = np.pi / np.array(x)
    x3_delta_exp = 7 * np.pi / np.array(x_exp)
    y1 = np.array(ke)             # Retirei o primeiro e o segundo valor, pois eram iguais a zero
    y4_exp = np.array(y_exp)
    x2 = []
    x4_delta =[]
    x4_delta_exp =[]
    y2 = np.log(y1)
    y4_exp = np.log(y4_exp)
    print("y2 = ", y2)

    for i in range(34):
        x2.append(np.mean(x1[i]))
        x4_delta.append(np.mean(x3_delta[i]))

    x2 = np.log(x2)
    x4_delta = np.log(x4_delta)
    x4_delta_exp = np.log(x3_delta_exp)


    x_reta = x4_delta[1:4] #m = -0.534
    y_reta = y2[1:4]       #m = -0.534

    #x_reta = x4_delta[3:10] #m = -0.343
    #y_reta = y2[3:10]       #m = -0.343

    m1, b1 = sm.least_square(x_reta, y_reta, 3)

    print("x2 = ", x2)
    print("shape x2 = ", np.shape(x2))

    #plt.plot(x2, y2 , '-', label="$\ln{t} \times \ln{E}$")
    plt.plot(x4_delta, y2 , '-', label="MFSim $\pi/\Delta$")
    plt.plot(x4_delta_exp, y4_exp , 'o', label="Experimental $7\pi/\Delta$")
    y_reta = m1 * x_reta + b1
    plt.plot(x_reta, y_reta, '--', color="k", label="$m = -0.534")
    #plt.plot(x_reta, y_reta, '--', color="k", label="$m = -0.343")
    print("O coeficiente angular é = ", m1)
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def plot_std_vel(x, vel, dataframe, exp_std_vel):
    """
    Plota o valor desvio padrão da velocidade em função da posição.

    INPUT:

    - x = vetor de posição com a quantidade tempo por número de probes;
    - vel = vetor de velocidade com a quantidade tempo por número de probes;

    OUTPUT:

    Retorna um vetor com 34 pontos da velocidade média em função da posição.
    """

    std_vel = []


    for i in range(34):
        #std_vel.append(sm.mean(sm.covariance(vel[i],vel[i], 3), 3))
        std_vel.append(sm.std(vel[i], 3))
   
    mean_var_vel = np.abs(np.array(std_vel))
   
    
    x_exp = exp_std_vel.x1_exp
    y_exp = exp_std_vel.std_u_exp

    x_paraview = dataframe.x_comp * (1.0 /0.15)
    y_paraview = dataframe.u_stddev  

    plt.style.use('ggplot')
        
    fig = plt.figure(figsize=(10.0, 4.0))
    axes1 = fig.add_subplot(1, 1, 1)
    axes1.set_ylabel('std_u')
    axes1.set_xlabel('x / L')
        
    x0 = np.array(x)                                # adimensionalizando a coordenada x
    x1 = x0 * 1.0 / 0.15

    y1 = np.array(mean_var_vel)

    x2 = []

    for i in range(34):
        x2.append(np.mean(x1[i]))

    x2 = np.array(x2)
    print("x2 = ", x2)

    plt.plot(x2, y1 , '-', label="Smag Cs = 0.15 (meu código)")
    plt.plot(x_paraview, y_paraview, '--', label="Smag Cs = 0.15 (paraview)")
    plt.plot(x_exp, y_exp, '*', color="green", label="Experimental")
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()

def fft_ke(t, s):
    fft = np.fft.fft(s)
    T = t[1] - t[0] # 0.001 -> 1/T = 1000
    N = np.size(s)
    #f = np.linspace(0, 1 / T, N)
    # fornece os componentes de frequência correspondentes aos dados
    f = np.fft.fftfreq(len(s), T)
    frequencias = f[:N // 2]
    amplitudes = np.abs(fft)[:N // 2] * 1 / N
    plt.ylabel("Amplitude")
    plt.xlabel("Frequência (Hz)")
    plt.bar(frequencias, amplitudes, width=1.5)
    plt.savefig('fft_freq.png')
    plt.close()
