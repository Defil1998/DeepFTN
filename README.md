# DeepFTN
A MATLAB simulator for Faster Than Nyquist (FTN) with Deep Learning. 
The code has been developed in the framework of [1] and makes use of CVX, a package for specifying and solving convex programs [2], and of the 5G NR LDPC MATLAB implementation available at [3].

- FTN_train_CNN: This script trains the Convolutional Neural Network (CNN) with skip connections model presented in [1], which aims at demodulating FTN-transmitted QPSK symbols.
- FTN_train_DNNbenchmark: This script trains the Deep Neural Network (DNN) proposed in [4], which equalizes FTN-transmitted QPSK symbols.
- FTN_test: This script performs a Monte Carlo simulation to evaluate the Bit Error Rate (BER), Block Error Rate (BLER), and throughput achieved by the CNN model and several benchmarks, including the DNN model presented in [4], Frequency-Domain Equalization [5], and Set-Theory-based Semi-Definite Relaxation Sequence Detection [6].

When using this simulator, please cite as: B. De Filippo, C. Amatetti, A. Vanelli-Coralli, "Faster-Than-Nyquist Equalization with Convolutional Neural Networks", to be presented at 2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), Istanbul, Turkey, 2025. Available at: https://arxiv.org/abs/2501.11594

# References
[1] B. De Filippo, C. Amatetti, A. Vanelli-Coralli, "Faster-Than-Nyquist Equalization with Convolutional Neural Networks", to be presented at 2025 IEEE 36th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), Istanbul, Turkey, 2025. Available at: https://arxiv.org/abs/2501.11594

[2] CVX Research, Inc., "CVX: Matlab software for disciplined convex programming", version 2.0, April 2011. Available at: https://cvxr.com/cvx

[3] J. Mao, "5G-NR-LDPC", October 2021. Available at: https://github.com/just1nGH/5G-NR-LDPC/tree/main

[4] P. Song, F. Gong, Q. Li, G. Li and H. Ding, "Receiver Design for Faster-Than-Nyquist Signaling: Deep-Learning-Based Architectures," in IEEE Access, vol. 8, pp. 68866-68873, 2020, doi: 10.1109/ACCESS.2020.2986679.

[5] S. Sugiura, "Frequency-Domain Equalization of Faster-than-Nyquist Signaling," in IEEE Wireless Communications Letters, vol. 2, no. 5, pp. 555-558, October 2013, doi: 10.1109/WCL.2013.072313.130408.

[6] E. Bedeer, M. H. Ahmed and H. Yanikomeroglu, "Low-Complexity Detection of High-Order QAM Faster-Than-Nyquist Signaling," in IEEE Access, vol. 5, pp. 14579-14588, 2017, doi: 10.1109/ACCESS.2017.2719628.
