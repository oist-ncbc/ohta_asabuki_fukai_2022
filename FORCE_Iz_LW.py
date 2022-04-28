#%%
import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time

def Levywalk(nt):
    np.random.seed(0)
    alpha = 1.5
    beta = 0
    # R = np.random.pareto(alpha, nt)
    R = levy_stable.rvs(alpha=alpha, beta=beta, size=nt)
    # print(R)
    theta = np.random.rand(nt)*360
    x = np.cumsum([R[i]*math.cos(theta[i]) for i in range(nt)])
    y = np.cumsum([R[i]*math.sin(theta[i]) for i in range(nt)])
    
    # print(np.max(np.abs(x)),np.max(np.abs(y)))
    
    x = 4*(x-np.min(x)) /(np.max(x)-np.min(x))-2
    y = 4*(y-np.min(y)) /(np.max(y)-np.min(y))-2
    
    return [[x[i], y[i]] for i in range(nt)]

def Circle(nt):
    theta = [2*math.pi*(i/nt) for i in range(nt)]
    x = [math.cos(i) for i in theta]
    y = [math.sin(i) for i in theta]
    
    return [[x[i], y[i]] for i in range(nt)]

#%%
def plot(dt,span):
    f = open('FORCE_IzCH_output.data','r')
    output = f.readlines()
    output = [i.split() for i in output]
    ox = [float(i[0].rstrip('\x00')) for i in output]
    oy = [float(i[1].rstrip('\x00')) for i in output]
    # print(len(x),len(y))
    
    g = open('FORCE_IzCH_target.data','r')
    target = g.readlines()
    target = [i.split() for i in target]
    tx = [float(i[0].rstrip('\x00')) for i in target]
    ty = [float(i[1].rstrip('\x00')) for i in target]
    # print(len(x),len(y))
    start = 250000
    end = start + 2*span + 1
    # end = len(ox)
    pltop = [ox[start:end],oy[start:end]]
    plttg = [tx[start:end],ty[start:end]]
    # print(pltop)
    N = end - start
    
    oparr = np.array([ox[start:end],oy[start:end]])
    tgarr = np.array([tx[start:end],ty[start:end]])
    err = np.sum((oparr-tgarr)**2, axis=0)
    err = np.sum(np.sqrt(err))/N
    print(err)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(*pltop,label='output')
    ax.plot(*plttg,'--',label='target',alpha=0.7)
    #ax.set_xlim(-2.1,2.1)
    #ax.set_ylim(-2.1,2.1)
    ax.set_xlabel(r'$x$',fontsize=20)
    ax.set_ylabel(r'$y$',fontsize=20)
    #plt.legend(fontsize=20)
    # plt.show()
    plt.savefig('FORCE_IzCH_Levywalk2D.pdf',transparent=True,bbox_inches="tight", pad_inches=0.0)
    plt.savefig('FORCE_IzCH_Levywalk2D.png',transparent=True,bbox_inches="tight", pad_inches=0.0)
    
    tnt = [dt*i for i in range(len(ox))]
    ont = [dt*i for i in range(len(tx))]
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(211)
    ax1.plot(tnt,ox,label='output')
    ax1.plot(ont,tx,'--',label='target',alpha=0.7)
    # ax1.set_xlim(5000,6000)
    start_1 = 10000
    end_1 = start_1 + span/5
    ax1.set_xlim(start_1,end_1)
    #ax1.set_ylim(-2.1,2.1)
    # ax1.set_xlabel(r'time',fontsize=15)
    ax1.set_ylabel(r'$x$',fontsize=20)
    #plt.legend(fontsize=15)
    ax2 = fig.add_subplot(212)
    ax2.plot(tnt,oy,label='output')
    ax2.plot(ont,ty,'--',label='target',alpha=0.7)
    # ax2.set_xlim(5000,6000)
    ax2.set_xlim(start_1,end_1)
    #ax2.set_ylim(-2.1,2.1)
    ax2.set_xlabel(r'time (ms)',fontsize=15)
    ax2.set_ylabel(r'$y$',fontsize=20)
    #plt.legend(fontsize=15)
    # plt.show()
    plt.savefig('FORCE_IzCH_Levywalk1D.pdf',transparent=True, bbox_inches="tight", pad_inches=0.0)
    # plt.savefig('FORCE_CH_Levywalk1D.png',transparent=True, bbox_inches="tight", pad_inches=0.0)

# Levywalk(10)
#%%
T = 15000 #シミュレーション時間 [ms]
dt = 0.04 #刻み幅
nt = round(T/dt) #ステップ数
N = 1000 #reservoir内のニューロン数

span = 10000 #一周期のステップ数
ntl = round(nt/span) 
ft = Levywalk(span) * ntl #レヴィウォークの教師信号
# ft = Circle(span) * ntl
# print(nt,ntl)
# ft = Circle(1000) * ntl
# print(len(ft))
g = open('FORCE_IzCH_target.data', 'w')
for i in range(nt):
    print("%8.5e %8.5e"%(ft[i][0], ft[i][1]), file=g)
np.random.seed(round(time.time()))

#表１に記載されている、ニューロンモデルのパラメータ****************************
C = 250 #ニューロンの膜時定数
vr = -60 #静止膜電位
vt = -20 #活動電位のしきい値
a = 0.02 #0.01
b = 0.2#0
vpeak = 30 #膜電位の最大値
################################################################################
# vreset = -50 #burst #発火後、もしくはvpeakに達した際のリセット値
# d = 2 #burst
vreset = -65 #normal
d = 8 #normal
lam=1
################################################################################
I_bias = 10#1000 #バイアス入力
k = 0.04#2.5
tr = 2 #EPSPの上昇時定数
td = 20 #EPSPの下降時定数

#ネットワークのパラメータ*******************************************************
p = 0.1 #リカレント回路の結合確率
G = 100#00 #リカレント結合の強度
Q = 100#00 #フィードバック結合の強度
# G = 5000
# Q = 5000

#Initialization***************************************************
v0 = vr + (vpeak-vr)*np.random.rand(N) #vはニューロンの膜電位です
v = v0
v_ = v #v_は、直前のタイムステップの膜電位です。ニューロンモデルの時間発展を計算する際に使います
u0 = np.zeros(N) #Izhikevichニューロンの、adaptation current
u = u0
x_hat = 0 #出力ユニットの活動の初期値
y_hat = 0

I_net = np.zeros(N) #(11)式によれば、(13)で決まる結合行列とrを用いて各ニューロンへの入力が計算されるのですが、これをそのまま実装すると非常に計算時間が長くなります。そこで、(13)式から決まる入力を二つの項に分解して計算します。第二項とrの積は、x_hatを用いて計算できることに注意してください。ここでのI_netは、第一項から決まる入力としておきます。

#以下のh,r,hr,h_updataは、各ニューロンへの入力と、学習の際に使う重み付きの入力の計算に使います。これは少しややこしいので、よくわからなければいつでも聞いてください
h = np.zeros(N)
r = np.zeros(N)
hr = np.zeros(N)
h_update = np.zeros(N)

omega0 = (np.random.randn(N,N))*(np.random.rand(N,N)<p)/(np.sqrt(N)*p) #リカレント結合。この重みは固定です
PSP = G*omega0 #(13)式で決まる入力が、各スパイクに応じてどの程度上昇するかを決めます
phix = np.random.randn(N)/np.sqrt(N) #readout結合
phiy = np.random.randn(N)/np.sqrt(N)
etax =(2*np.random.rand(N)-1) #出力ユニットからreservoirへのフィードバック結合
etay =(2*np.random.rand(N)-1)


Pinv = np.eye(N)/lam #相関行列の逆行列(に、regularizationのパラメータをかけたもの)
step = 20 #Sussilloの論文に書かれていたはずですが、FORCEはある一定の時刻ごとに重みの更新を行います
imin = int(110/dt)#round(5000/dt) #学習の開始時刻
icrit = nt-int(110/dt)#学習の終了時刻
out_hist = np.zeros((nt,2)) #出力値を格納
v_hist = np.zeros((nt, 5)) #reservoir neuronの活動を格納
errx = 0
erry = 0
#Learning****************************
f = open('FORCE_IzCH_output.data', 'w')

for i in tqdm(range(nt)):

    I = I_net + etax*Q*x_hat + etay*Q*y_hat + I_bias #(13)式のomegaにrをかけることで得られる入力(にバイアス入力を加えた)。
    if i % span == 0:
        v = v0
        u = u0
    else:
        # v += dt*((k*(v - vr)*(v - vt) - u + I)) / C #(6)式
        v += dt*(k*v**2 + 5*v + 140 - u + I)
        # u += dt*(a*(b*(v_-vr)-u)) #(7)式
        u += dt*(a*(b*v_-u))

    index = (v>=vpeak) #スパイクが起きたニューロンのインデックスがTrueになる

    h_update = np.sum(PSP[:, index], axis=1) #I_netに対する更新の度合いを決める

    I_net = (1-dt/tr)*I_net + h*dt
    h = (1-dt/td)*h + h_update/(tr*td)
    
    r = (1-dt/tr)*r + hr*dt
    hr = (1-dt/td)*hr + index/(tr*td)
    
    x_hat = np.dot(phix, r) #readoutの出力
    y_hat = np.dot(phiy, r)

    errx = x_hat - ft[i][0] #error
    erry = y_hat - ft[i][1]
    
    ##FORCEでの学習
    if i % step == 0:
        if i > imin:
            if i < icrit:
                cd = np.dot(Pinv, r)
                c = 1. / (1. + np.dot(r.T, cd))
                phix = phix - (cd * errx)
                phiy = phiy - (cd * erry)
                # phix = phix - (cd * errx * c)
                # phiy = phiy - (cd * erry * c)
                Pinv = Pinv - np.outer(cd,cd) * c

    u = u + d*index
    v = v + (vreset-v)*index
    v_ = v
    v_hist[i,:] = v[0:5]
    out_hist[i] = [x_hat, y_hat]
    print("%8.5e %8.5e"%(x_hat, y_hat), file=f)
    

#Plotting
#学習前のreservoirニューロンの活動(5例)
# step_range = 125000
step_range = 125000
plt.figure(figsize=(6, 6))
for j in range(5):
    plt.plot(np.arange(step_range)*dt, v_hist[:step_range,j]/(50-vreset)+j)
plt.title('Pre-Learning')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.xlim(0,110)
plt.savefig("IzCH_pre.pdf", dpi=350,bbox_inches="tight", pad_inches=0.0)
# plt.savefig("IzCH_pre.png",dpi=350,bbox_inches="tight", pad_inches=0.0)
 
#学習後のreservoirニューロンの活動(5例)
plt.figure(figsize=(6, 6))
for j in range(5):
    plt.plot(np.arange(nt-step_range, nt)*dt, v_hist[nt-step_range:,j]/(50-vreset)+j,)
plt.title('Post Learning')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.xlim(T-110-1,T-1)
# plt.ylim(-1,5)
plt.savefig("IzCH_post.pdf", dpi=350,bbox_inches="tight", pad_inches=0.0)
# plt.savefig("IzCH_post.png", dpi=350, bbox_inches="tight", pad_inches=0.0)

# start = 250000
# end = start + 10
# Narr = start - end
# oparr = np.array([out_hist[start:end]]).T
# tgarr = np.array([ft[start:end]]).T
# # print(oparr)
# # print(tgarr)
# # print(oparr-tgarr)
# err = np.sum((oparr-tgarr)**2, axis=0)
# # print(err)
# err = np.sqrt(np.sum(err))/Narr
# print(err)

plot(dt=dt,span=span)
#%%
    
# plot(dt=0.04)
#%%
# plot(dt=0.04,span=20000)
# %%
