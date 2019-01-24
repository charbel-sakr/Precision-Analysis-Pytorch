import numpy as np

gain_read = 'normalized_gains/'
BO_read = 'precision_offsets/'


for b in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    pm=0
    #input activations
    EAL = np.loadtxt(gain_read+'input.activation.txt')
    BAL = b + np.loadtxt(BO_read+'input.activation.txt')
    pm += EAL*np.power(4.0,1.0-BAL)

    #conv1
    EAL = np.loadtxt(gain_read+'conv1.activation.txt')
    BAL = b + np.loadtxt(BO_read+'conv1.activation.txt')
    pm += EAL*np.power(4.0,1.0-BAL)

    EWL = np.loadtxt(gain_read+'conv1.weight.txt')
    BWL = b + np.loadtxt(BO_read+'conv1.weight.txt')
    pm += EWL*np.power(4.0,1.0-BWL)

    #loop
    for l in ['1','2','3','4']:
        for s in ['0','1']:
            name = 'layer'+l+'.'+s
            #inside.activation
            EAL = np.loadtxt(gain_read+name+'.inside.activation.txt')
            BAL = b + np.loadtxt(BO_read+name+'.inside.activation.txt')
            pm += EAL*np.power(4.0,1.0-BAL)

            #outside.activation
            EAL = np.loadtxt(gain_read+name+'.outside.activation.txt')
            BAL = b + np.loadtxt(BO_read+name+'.outside.activation.txt')
            pm += EAL*np.power(4.0,1.0-BAL)

            #conv1.weight
            EWL = np.loadtxt(gain_read+name+'.conv1.weight.txt')
            BWL = b + np.loadtxt(BO_read+name+'.conv1.weight.txt')
            pm += EWL*np.power(4.0,1.0-BWL)

            #conv2.weight
            EWL = np.loadtxt(gain_read+name+'.conv2.weight.txt')
            BWL = b + np.loadtxt(BO_read+name+'.conv2.weight.txt')
            pm += EWL*np.power(4.0,1.0-BWL)

            #shorcut
            if((l!='1')and(s=='0')):
                EWL = np.loadtxt(gain_read+name+'.shortcut.weight.txt')
                BWL = b + np.loadtxt(BO_read+name+'.shortcut.weight.txt')
                pm += EWL*np.power(4.0,1.0-BWL)

    print(repr(b) + ' ' + repr(pm))
