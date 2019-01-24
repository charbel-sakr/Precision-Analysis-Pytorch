import numpy as np

gain_read =  'normalized_gains/'
EMIN = np.loadtxt(gain_read+'coarse_gains.txt')[0]

precisions_write = 'precision_offsets/'


#input activations
EAL = np.loadtxt(gain_read+'input.activation.txt')
BAL = np.round(np.log2(np.sqrt(EAL/EMIN)))
np.savetxt(precisions_write+'input.activation.txt',np.asarray([BAL]))

#conv1
EAL = np.loadtxt(gain_read+'conv1.activation.txt')
BAL = np.round(np.log2(np.sqrt(EAL/EMIN)))
np.savetxt(precisions_write+'conv1.activation.txt',np.asarray([BAL]))

EWL = np.loadtxt(gain_read+'conv1.weight.txt')
BWL = np.round(np.log2(np.sqrt(EWL/EMIN)))
np.savetxt(precisions_write+'conv1.weight.txt',np.asarray([BWL]))

#loop
for l in ['1','2','3','4']:
    for s in ['0','1']:
        name = 'layer'+l+'.'+s
        #inside.activation
        EAL = np.loadtxt(gain_read+name+'.inside.activation.txt')
        BAL = np.round(np.log2(np.sqrt(EAL/EMIN)))
        np.savetxt(precisions_write+name+'.inside.activation.txt',np.asarray([BAL]))

        #outsideactivation
        EAL = np.loadtxt(gain_read+name+'.outside.activation.txt')
        BAL = np.round(np.log2(np.sqrt(EAL/EMIN)))
        np.savetxt(precisions_write+name+'.outside.activation.txt',np.asarray([BAL]))

        #conv1.weight
        EWL = np.loadtxt(gain_read+name+'.conv1.weight.txt')
        BWL = np.round(np.log2(np.sqrt(EWL/EMIN)))
        np.savetxt(precisions_write+name+'.conv1.weight.txt',np.asarray([BWL]))

        #conv2.weight
        EWL = np.loadtxt(gain_read+name+'.conv2.weight.txt')
        BWL = np.round(np.log2(np.sqrt(EWL/EMIN)))
        np.savetxt(precisions_write+name+'.conv2.weight.txt',np.asarray([BWL]))

        #shortcut.weight
        if((l!='1')and(s=='0')):
            EWL = np.loadtxt(gain_read+name+'.shortcut.weight.txt')
            BWL = np.round(np.log2(np.sqrt(EWL/EMIN)))
            np.savetxt(precisions_write+name+'.shortcut.weight.txt',np.asarray([BWL]))

#linear.weight
EWL = np.loadtxt(gain_read+'linear.weight.txt')
BWL = np.round(np.log2(np.sqrt(EWL/EMIN)))
np.savetxt(precisions_write+'linear.weight.txt',np.asarray([BWL]))

