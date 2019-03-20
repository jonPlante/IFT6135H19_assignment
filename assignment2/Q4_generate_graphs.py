import numpy as np
import matplotlib.pyplot as plt
import os
plt.figure(1)#=plt.figure(1)
         #
plt.figure(2)#=plt.figure(2)
         #
plt.figure(3)#=plt.figure(3)
         #
plt.figure(4)#=plt.figure(4)
         #
plt.figure(5)#=plt.figure(5)
         #
plt.figure(6)#=plt.figure(6)
         #
plt.figure(7)#=plt.figure(7)
         #
plt.figure(8)#=plt.figure(8)
         #
plt.figure(9)#=plt.figure(9)

plt.figure(10)#=plt.figure(10)

plt.figure(11)#=plt.figure(11)

plt.figure(12)#=plt.figure(12)

folder=r'.\runs'
file='learning_curves.npy'

listDir=os.listdir(folder)
for dir in listDir:
	if os.path.isdir(os.path.join(folder,dir)):
		titles=os.listdir(os.path.join(folder,dir))
		for title in titles:
			if os.path.isdir(os.path.join(folder,dir,title)):
				print('Processing : '+ title + '\n')
				lc_path=os.path.join(folder,dir,title,file)
				x = np.load(lc_path)[()]
				
				epoch=np.arange(1,41)
				walltime=epoch*x.get('times')
				
				plt.figure(13)
				plt.plot(epoch, x.get('train_ppls'), label='train PPL')
				plt.plot(epoch, x.get('val_ppls'),label='val PPL')
				plt.ylabel('PPL')
				plt.xlabel('Epoch')
				plt.legend()
				plt.grid(True)
				plt.title('PPL over epochs for experiment : '+title)
				plt.xlim(0,40)
				plt.savefig(os.path.join(folder,dir,title,'PPLvsEpoch_foc.png'))
				plt.ylim(0,1000)
				plt.savefig(os.path.join(folder,dir,title,'PPLvsEpoch.png'))
				plt.clf()
				
				plt.figure(14)
				plt.plot(walltime, x.get('train_ppls'), label='train PPL')
				plt.plot(walltime, x.get('val_ppls'),label='val PPL')
				plt.ylabel('PPL')
				plt.xlabel('Wall-clock-time (s)')
				plt.legend()
				plt.grid(True)
				plt.title('PPL over wall-clock-time for experiment : '+title)
				plt.savefig(os.path.join(folder,dir,title,'PPLvsWCT.png'))
				plt.ylim(0,1000)
				plt.savefig(os.path.join(folder,dir,title,'PPLvsWCT_foc.png'))
				plt.clf()
				
				firstRNN=True
				firstGRU=True
				firstTRA=True
				
				if 'RNN' in title:
					plt.figure(1)
					plt.plot(epoch,x.get('val_ppls'),label=title.replace('_RNN',''))
					plt.figure(2)
					plt.plot(walltime,x.get('val_ppls'),label=title.replace('_RNN',''))
					if firstRNN:
						plt.figure(1)
						plt.ylabel('Validation PPL')
						plt.xlabel('Epoch')
						plt.grid(True)
						plt.title('Val. PPL over epochs for RNN')
						plt.xlim(0,40)
						plt.figure(2)
						plt.ylabel('Validation PPL')
						plt.xlabel('Wall-clock-time (s)')
						plt.grid(True)
						plt.title('Val. PPL over wall clock time for RNN')
						firstRNN=False
				
				elif 'GRU' in title:
					plt.figure(3)
					plt.plot(epoch,x.get('val_ppls'),label=title.replace('_GRU',''))
					plt.figure(4)
					plt.plot(walltime,x.get('val_ppls'),label=title.replace('_GRU',''))
					if firstGRU:
						plt.figure(3)
						plt.ylabel('Validation PPL')
						plt.xlabel('Epoch')
						plt.grid(True)
						plt.title('Val. PPL over epochs for GRU')
						plt.xlim(0,40)
						plt.figure(4)
						plt.ylabel('Validation PPL')
						plt.xlabel('Wall-clock-time (s)')
						plt.grid(True)
						plt.title('Val. PPL over wall clock time for GRU')
						firstGRU=False
				else:
					plt.figure(5)
					plt.plot(epoch,x.get('val_ppls'),label=title.replace('_TRANSFORMER',''))
					plt.figure(6)
					plt.plot(walltime,x.get('val_ppls'),label=title.replace('_TRANSFORMER',''))
					if firstTRA:
						plt.figure(5)
						plt.ylabel('Validation PPL')
						plt.xlabel('Epoch')
						plt.grid(True)
						plt.title('Val. PPL over epoch for Transformer')
						plt.xlim(0,40)
						plt.figure(6)
						plt.ylabel('Validation PPL')
						plt.xlabel('Wall-clock-time (s)')
						plt.grid(True)
						plt.title('Val. PPL over wall-clock-time for Transformer')
						firstTRA=False
						
				firstADAM=True
				firstSGDs=True
				firstSGD=True
				if 'ADAM' in title:
					plt.figure(7)
					plt.plot(epoch,x.get('val_ppls'),label=title.replace('_ADAM',''))
					plt.figure(8)
					plt.plot(walltime,x.get('val_ppls'),label=title.replace('_ADAM',''))
					if firstADAM:
						plt.figure(7)
						plt.ylabel('Validation PPL')
						plt.xlabel('Epoch')
						plt.grid(True)
						plt.title('Val. PPL over epoch for ADAM')
						plt.xlim(0,40)
						plt.figure(8)
						plt.ylabel('Validation PPL')
						plt.xlabel('Wall-clock-time (s)')
						plt.grid(True)
						plt.title('Val. PPL over wall-clock-time for ADAM')
						firstADAM=False
				elif 'SGD_LR_SCHEDULE' in title:
					plt.figure(11)
					plt.plot(epoch,x.get('val_ppls'),label=title.replace('_SGD_LR_SCHEDULE',''))
					plt.figure(12)
					plt.plot(walltime,x.get('val_ppls'),label=title.replace('_SGD_LR_SCHEDULE',''))
					if firstSGDs:
						plt.figure(11)
						plt.ylabel('Validation PPL')
						plt.xlabel('Epoch')
						plt.grid(True)
						plt.title('Val. PPL over epoch for SGD_LR_SCHEDULE')
						plt.xlim(0,40)
						plt.figure(12)
						plt.ylabel('Validation PPL')
						plt.xlabel('Wall-clock-time (s)')
						plt.grid(True)
						plt.title('Val. PPL over wall-clock-time for SGD_LR_SCHEDULE')
						firstSGDs=False
				else:
					plt.figure(9)
					plt.plot(epoch,x.get('val_ppls'),label=title.replace('_SGD',''))
					plt.figure(10)
					plt.plot(walltime,x.get('val_ppls'),label=title.replace('_SGD',''))
					if firstSGD:
						plt.figure(9)
						plt.ylabel('Validation PPL')
						plt.xlabel('Epoch')
						plt.grid(True)
						plt.title('Val. PPL over epoch for SGD')
						plt.xlim(0,40)
						plt.figure(10)
						plt.ylabel('Validation PPL')
						plt.xlabel('Wall-clock-time (s)')
						plt.grid(True)
						plt.title('Val. PPL over wall-clock-time for SGD')
						firstSGD=False


plt.figure(1)
plt.legend()
plt.savefig('RNN_epoch.png')
plt.ylim(0,1000)
plt.savefig('RNN_epoch_foc.png')
plt.figure(2)
plt.legend()
plt.savefig('RNN_WCT.png')
plt.ylim(0,1000)
plt.savefig('RNN_WCT_foc.png')
plt.figure(3)
plt.legend()
plt.savefig('GRU_epoch.png')
plt.ylim(0,1000)
plt.savefig('GRU_epoch_foc.png')
plt.figure(4)
plt.legend()
plt.savefig('GRU_WCT.png')
plt.ylim(0,1000)
plt.savefig('GRU_WCT_foc.png')
plt.figure(5)
plt.legend()
plt.savefig('TRA_epoch.png')
plt.ylim(0,1000)
plt.savefig('TRA_epoch_foc.png')
plt.figure(6)
plt.legend()
plt.savefig('TRA_WCT.png')
plt.ylim(0,1000)
plt.savefig('TRA_WCT_foc.png')

plt.figure(7)
plt.legend()
plt.savefig('ADAM_epoch.png')
plt.ylim(0,1000)
plt.savefig('ADAM_epoch_foc.png')
plt.figure(8)
plt.legend()
plt.savefig('ADAM_WCT.png')
plt.ylim(0,1000)
plt.savefig('ADAM_WCT_foc.png')
plt.figure(9)
plt.legend()
plt.savefig('SGD_epoch.png')
plt.ylim(0,1000)
plt.savefig('SGD_epoch_foc.png')
plt.figure(10)
plt.legend()
plt.savefig('SGD_WCT.png')
plt.ylim(0,1000)
plt.savefig('SGD_WCT_foc.png')
plt.figure(11)
plt.legend()
plt.savefig('SGD_LR_SCHEDULE_epoch.png')
plt.ylim(0,1000)
plt.savefig('SGD_LR_SCHEDULE_epoch_foc.png')
plt.figure(12)
plt.legend()
plt.savefig('SGD_LR_SCHEDULE_WCT.png')
plt.ylim(0,1000)
plt.savefig('SGD_LR_SCHEDULE_WCT_foc.png')
