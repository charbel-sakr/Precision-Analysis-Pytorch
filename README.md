# Precision-Analysis-Pytorch
Pytorch version of precision analysis from my ICML 2017 and ICASSP 2018 papers.

In this repository, I am adding a CIFAR-10 ResNet 18 example. Please note that the code assumes (or guides to obtain) a trained network using the code by kuangliu (pytorch tutorial from https://github.com/kuangliu/pytorch-cifar).

You will find code to do the following: obtain clean pre-trained networks (only dot products and activations), quantization noise gains calculation (the E values), bound evaluation, and inference in fixed-point (fixed-point simulation). 

Please note two things: (1) This code assumes a network that was trained using hardtanh, (2) for cost evaluations, you can use the exact same numpy code from my other repository https://github.com/charbel-sakr/Precision-Analysis-of-Neural-Networks.

Please get in touch if you have any question or comment.

Sakr, Charbel, Yongjune Kim, and Naresh Shanbhag. "Analytical Guarantees on Numerical Precision of Deep Neural Networks." International Conference on Machine Learning. 2017.

@inproceedings{sakr2017analytical,

title={Analytical Guarantees on Numerical Precision of Deep Neural Networks},

author={Sakr, Charbel and Kim, Yongjune and Shanbhag, Naresh},

booktitle={International Conference on Machine Learning},

pages={3007--3016},

year={2017}

}

Charbel
