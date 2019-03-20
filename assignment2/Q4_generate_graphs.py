import numpy as np
import matplotlib.pyplot as plt


folder=r'.\runs\Q4_1\01_RNN_ADAM'
file='\learning_curves.npy'
lc_path=folder+file
title='01_RNN_ADAM'

x = np.load(lc_path)[()]

epoch=np.arange(1,41)
walltime=epoch*x.get('times')

plt.figure()
plt.plot(epoch, x.get('train_ppls'), label='train PPL')
plt.plot(epoch, x.get('val_ppls'),label='val PPL')
plt.ylabel('PPL')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.title('PPL over epochs for experiment:'+title)
plt.savefig(folder+'\PPLvsEpoch.png')

plt.figure()
plt.plot(walltime, x.get('train_ppls'), label='train PPL')
plt.plot(walltime, x.get('val_ppls'),label='val PPL')
plt.ylabel('PPL')
plt.xlabel('wall-clock-time (s)')
plt.legend()
plt.grid(True)
plt.title('PPL over wall-clock-time for experiment:'+title)
plt.savefig(folder+'\PPLvsWCT.png')