import numpy as np

gain_read = 'gain_dump/'
gain_write = 'normalized_gains/'
scalar_read = 'scalars/'

EA=0
EW=0

least_gain=1000

#input activations
EAL = np.load(gain_read+'input.activation.npy')*16.0
least_gain = least_gain if least_gain<EAL else EAL
np.savetxt(gain_write+'input.activation.txt',np.asarray([EAL]))
EA+=EAL

#conv1
EAL = np.load(gain_read+'conv1.activation.npy')
least_gain = least_gain if least_gain<EAL else EAL
np.savetxt(gain_write+'conv1.activation.txt',np.asarray([EAL]))
EA+=EAL

EWL = np.load(gain_read+'conv1.weight.npy')
scalar = np.load(scalar_read+'conv1.weight.npy')
EWL*=np.square(scalar)
least_gain = least_gain if least_gain<EWL else EWL
np.savetxt(gain_write+'conv1.weight.txt',np.asarray([EWL]))
EW+=EWL

#loop
for l in ['1','2','3','4']:
    for s in ['0','1']:
        name = 'layer'+l+'.'+s
        #inside.activation
        EAL = np.load(gain_read+name+'.inside.activation.npy')
        least_gain = least_gain if least_gain<EAL else EAL
        np.savetxt(gain_write+name+'.inside.activation.txt',np.asarray([EAL]))
        EA+=EAL

        #outsideactivation
        EAL = np.load(gain_read+name+'.outside.activation.npy')
        least_gain = least_gain if least_gain<EAL else EAL
        np.savetxt(gain_write+name+'.outside.activation.txt',np.asarray([EAL]))
        EA+=EAL

        #conv1.weight
        EWL = np.load(gain_read+name+'.conv1.weight.npy')
        scalar = np.load(scalar_read+name+'.conv1.weight.npy')
        EWL*=np.square(scalar)
        least_gain = least_gain if least_gain<EWL else EWL
        np.savetxt(gain_write+name+'.conv1.weight.txt',np.asarray([EWL]))
        EW+=EWL

        #conv2.weight
        EWL = np.load(gain_read+name+'.conv2.weight.npy')
        scalar = np.load(scalar_read+name+'.conv2.weight.npy')
        EWL*=np.square(scalar)
        least_gain = least_gain if least_gain<EWL else EWL
        np.savetxt(gain_write+name+'.conv2.weight.txt',np.asarray([EWL]))
        EW+=EWL

        #shortcut.weight
        if((l!='1')and(s=='0')):
            EWL = np.load(gain_read+name+'.shortcut.weight.npy')
            scalar = np.load(scalar_read+name+'.shortcut.weight.npy')
            EWL*=np.square(scalar)
            least_gain = least_gain if least_gain<EWL else EWL
            np.savetxt(gain_write+name+'.shortcut.weight.txt',np.asarray([EWL]))
            EW+=EWL

#linear.weight
EWL = np.load(gain_read+'linear.weight.npy')
scalar = np.load(scalar_read+'linear.weight.npy')
EWL*=np.square(scalar)
least_gain = least_gain if least_gain<EWL else EWL
np.savetxt(gain_write+'linear.weight.txt',np.asarray([EWL]))
EW+=EWL

np.savetxt(gain_write+'coarse_gains.txt',np.asarray([least_gain,EA,EW,EA+EW]))


