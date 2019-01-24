'''Train CIFAR10 with PyTorch.'''
import numpy as np

read_folder_name = 'extracted_params/'
write_folder_name = 'scalars/'
name = 'conv1.weight.npy'
np.save(write_folder_name+name,np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(np.load(read_folder_name+name)))))))
print('done with '+name)
print(np.load(write_folder_name+name))

for l in ['1','2','3','4']:
    for s in ['0','1']:
        for c in ['1','2']:
            name = 'layer'+l+'.'+s+'.conv'+c+'.weight.npy'
            np.save(write_folder_name+name,np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(np.load(read_folder_name+name)))))))
            print('done with '+name)
            print(np.load(write_folder_name+name))
        if (l!='1') and (s=='0'):
            name = 'layer'+l+'.'+s+'.shortcut.weight.npy'
            np.save(write_folder_name+name,np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(np.load(read_folder_name+name)))))))
            print('done with '+name)
            print(np.load(write_folder_name+name))
name = 'linear.weight.npy'
np.save(write_folder_name+name,np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(np.load(read_folder_name+name)))))))
print('done with '+name)
print(np.load(write_folder_name+name))

