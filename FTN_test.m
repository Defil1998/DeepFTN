%% --------------------------------------------------------%%
%                                                           %
% Date:             November, 2024                          %
% Author:           Bruno De Filippo, Ph.D. student         %
% Affiliation:      DEI Department, University of Bologna   %
% Email:            bruno.defilippo@unibo.it                %
% Personal email:   brunodefilippo@gmail.com                %
%                                                           %
%-----------------------------------------------------------%

%% References
% [1] S. Sugiura, "Frequency-Domain Equalization of Faster-than-Nyquist Signaling," in IEEE Wireless Communications Letters, vol. 2, no. 5, pp. 555-558, October 2013, doi: 10.1109/WCL.2013.072313.130408
% [2] E. Bedeer, M. H. Ahmed and H. Yanikomeroglu, "Low-Complexity Detection of High-Order QAM Faster-Than-Nyquist Signaling," in IEEE Access, vol. 5, pp. 14579-14588, 2017, doi: 10.1109/ACCESS.2017.2719628
% [3] P. Song, F. Gong, Q. Li, G. Li and H. Ding, "Receiver Design for Faster-Than-Nyquist Signaling: Deep-Learning-Based Architectures," in IEEE Access, vol. 8, pp. 68866-68873, 2020, doi: 10.1109/ACCESS.2020.2986679

clear 
close all
clc

seed = 14;
rng(seed);

addpath('Functions')
addpath('Functions/LDPC/')
addpath('Functions/Tables/')
addpath('Results')

M = 4;                  % Modulation order
nSymbols = 50;          % Number of symbols per transmission
nRun = 1e5;             % Number of Monte Carlo iterations
EbN0dB = -10:20;        % Range of Eb/N0 values to test [dB]
sps_nyquist = 50;       % Samples per symbol at Nyquist rate
Tn = 1e-6;              % Nyquist symbol time

ftnParam = 0.6;         % FTN compression parameter
rollOff = 0.5;          % SRRC roll-off
NISI = 10;              % One-sided Inter-Symbol Interference (ISI) span (at Nyquist rate)
NPad = 12;              % Number of padding symbols for the Convolutional Neural Network (CNN) model
nu_FDE = 10;            % Half-length of cyclic prefix in Frequency-Domain Equalization (FDE) benchmark [1]
L_SDR = 1000;           % Number of iterations for Gaussian approximation in STSDRSE receiver [2]
nSymbolsIn_DNN = 32;    % Number of input symbols in DNN benchmark [3], extrapolated from DNN complexity using Excel solver
nSymbolsOut_DNN = 8;    % Number of output symbols in DNN benchmark [3], extrapolated from DNN complexity using Excel solver

Rc = 1/2;               % LDPC code rate
maxIter_LDPC = 10;      % Maximum number of iterations in LDPC decoder

toggle_FDE = true;      % Set true to add FDE to the list of receivers
toggle_SDR = true;      % Set true to add STSDRSE to the list of receivers
toggle_DNN = true;      % Set true to add the AI benchmark to the list of receivers

m = log2(M);                        % Number of bits per symbol
sps_ftn = ftnParam * sps_nyquist;   % Samples per symbol at FTN rate
nCodedBits = nSymbols * m;          % Length of encoded bits sequence
EbN0 = 10.^(EbN0dB ./ 10);          % Eb/N0 ratio
EsN0 = EbN0 * m;                    % Es/N0 ratio

if toggle_DNN           % Load DNN benchmark model
    load(strcat("Models/Bench-", string(ftnParam), "-", string(rollOff), ".mat"), "bestNet");
    bestNet_DNN = bestNet;
    nSymbolsExtraHalf = (nSymbolsIn_DNN - nSymbolsOut_DNN) / 2; % Total number of additional symbols to consider for DNN benchmark          
else
    nSymbolsExtraHalf = 0;      % Initialize number of additional symbols to account for ISI boundary conditions
end
if nSymbolsExtraHalf < NPad     % Update nSymbolsExtraHalf to ensure adequate padding is added for CNN
    nSymbolsExtraHalf = NPad;
end

load(strcat("Models/CNN-", string(ftnParam), "-", string(rollOff), ".mat"), "bestNet"); % Load CNN

%% ISI channel
g = rcosdesign(rollOff, 2*NISI, sps_nyquist);                               % Generate SRRC FIR filter
g_xcorr = xcorr(g);                                                         % Compute autocorrelation of SRRC filter
g_xcorr_samp = g_xcorr((length(g_xcorr) + 1) / 2:sps_ftn:end);              % Sample second half of xcorr at FTN rate (g_xcorr is symmetric)
nSymbolsExtraHalf = nSymbolsExtraHalf + length(g_xcorr_samp);               % Update the number of extra symbols to be added to data symbols to cover the entire ISI span
G = toeplitz([g_xcorr_samp, zeros(1, nSymbols + 2*nSymbolsExtraHalf - length(g_xcorr_samp))]);     % ISI channel matrix considering the entire ISI span

if toggle_FDE
    G_FDE = toeplitz([g_xcorr_samp, zeros(1, nSymbols + length(g_xcorr_samp) + 2*nu_FDE)]);        % For transmission (with CP)
    G_FDE_approx = toeplitz([g_xcorr_samp(1 : nu_FDE+1), zeros(1, nSymbols - (nu_FDE+1))]);        % For receiver (approximated channel matrix reported in )
    G_FDE_approx = G_FDE_approx(nu_FDE+1:end-nu_FDE, :);
    g_xcorr_FDE = [flip(g_xcorr_samp(2:nu_FDE + 1)), g_xcorr_samp(1:nu_FDE + 1)];
    G_FDE_approx = [G_FDE_approx; zeros(2*nu_FDE, nSymbols)];
    for k = 1:2*nu_FDE
        G_FDE_approx(nSymbols - 2*nu_FDE + k, :) = [g_xcorr_FDE(end - k + 1: end), ...   % Add circular rows to the end
                                                    zeros(1, nSymbols - 2*nu_FDE - 1), ...
                                                    g_xcorr_FDE(1:2*nu_FDE - k + 1)];
    end
    Q_FDE = fft(eye(nSymbols))./sqrt(nSymbols);     % Eigenvectors of circulant matrix (DFT matrix)
    Lambda_FDE = fft(G_FDE_approx(1, :));           % Eigenvalues of circulant matrix
end

if toggle_SDR
    Gn_temp = toeplitz([g_xcorr_samp, zeros(1, nSymbols - length(g_xcorr_samp))]);
    Gn_SDR = [real(Gn_temp), -imag(Gn_temp); ...
              imag(Gn_temp), real(Gn_temp)];
end

%% LDPC settings
nBitsPerPacket = ceil(nCodedBits*Rc); % Number of information bits
ldpc_info = LDPC_INFO(nCodedBits, nBitsPerPacket); 

%% Monte Carlo simulation
BER_th = 0.5 * erfc(sqrt(3/(M-1) * EbN0));  % Theoretical Bit Error Probability
LLR_th = log((1-BER_th)./BER_th);           % Theoretical LLR

%Initialize error counters
nErrors=zeros(length(EsN0), 1);
nErrors_CNN = zeros(length(EsN0), 1);
if toggle_FDE
    nErrors_FDE = zeros(length(EsN0), 1);
end
if toggle_SDR
    nErrors_SDR = zeros(length(EsN0), 1);
end
if toggle_DNN
    nErrors_DNN = zeros(length(EsN0), 1);
end

nErrBlock = zeros(length(EsN0), 1);
nErrBlock_CNN = zeros(length(EsN0), 1);
if toggle_FDE
    nErrBlock_FDE = zeros(length(EsN0), 1);
end
if toggle_SDR
    nErrBlock_SDR = zeros(length(EsN0), 1);
end
if toggle_DNN   
    nErrBlock_DNN = zeros(length(EsN0), 1);
end
nErrBlock_th = zeros(length(EsN0), 1);

statusToDisp = 0;
disp("Simulation progress: 0%")

for i=1:nRun

    status = floor((i-1)/nRun*100);     % Completion of the simulation (percentage)

    if status > statusToDisp            % Update status only when the value changes
        clc
        statusToDisp = status;
        disp(strcat("Simulation progress: ", string(statusToDisp), "%"))
    end 

    %% Bits source
    bits = round(rand(nBitsPerPacket, 1));
    
    %% LDPC Encoder
    bits_padded = [bits; zeros(ldpc_info.n_F, 1)];
    cword = nrldpc_encoder(ldpc_info, bits_padded);

    %% Rate matching and interleaving
    codedBits = nrldpc_rate_match(ldpc_info, cword, nCodedBits);    
    interBits = reshape(reshape(codedBits, m, []).', [], 1);      % Bit interleaving
    
    %% Mapper
    symbols = qammod([round(rand(m*nSymbolsExtraHalf, 1)); ...      % Dummy symbols to cover full ISI span, DNN benchmark padding, and CNN padding
                      interBits; ...                                % Data symbols
                      round(rand(m*nSymbolsExtraHalf, 1))], ...     % Dummy symbols to cover full ISI span, DNN benchmark padding, and CNN padding
                     M, 'gray', 'input', 'bit', 'UnitAveragePower', true);

    if toggle_FDE
        symbols_FDE = qammod([round(rand(m*length(g_xcorr_samp), 1)); ...   % Dummy symbols to cover full ISI span
                              interBits; ...                                % Data symbols
                              interBits(1:m*2*nu_FDE); ...                  % Cyclic prefix
                              round(rand(m*length(g_xcorr_samp), 1))], ...  % Dummy symbols to cover full ISI span
                              M, 'gray', 'input', 'bit', 'UnitAveragePower', true);
    end

    %% ISI channel
    % Apply ISI channel to generated symbols 
    txSymbols = G * symbols;

    if toggle_FDE
        txSymbols_FDE = G_FDE * symbols_FDE;
    end

    %% Iteration over Es/N0
    netInput = zeros(2*NPad + nSymbols, 2, length(EsN0));       % Initialize CNN input to be filled with symbols at different Es/N0

    for h=1:length(EsN0)
    
        %% AWGN
        Es = mean(abs(txSymbols(length(g_xcorr_samp) + 1:end - length(g_xcorr_samp))).^2);  % Average energy per symbol
        N0 = Es./EsN0;                                                                      % Noise power spectral density

        r_hat = txSymbols + (mvnrnd(zeros(1, size(G, 1)), G * N0(h)/2) + ...                % Add colored noise
             1i .* mvnrnd(zeros(1, size(G, 1)), G * N0(h)/2)).';

        if toggle_DNN
            r_hat_DNN = r_hat(length(g_xcorr_samp) + 1:end - length(g_xcorr_samp));     % Maintain only data symbols + padding required by DNN benchmark
        end

        if toggle_FDE
            Es_FDE = mean(abs(txSymbols_FDE(length(g_xcorr_samp) + 1:end - length(g_xcorr_samp))).^2);     % Average energy per symbol
            N0_FDE = Es_FDE./EsN0;                                                                         % Noise power spectral density

            r_hat_FDE = txSymbols_FDE + (mvnrnd(zeros(1, size(G_FDE, 1)), G_FDE * N0_FDE(h)/2) + ...       % Add colored noise
                1i .* mvnrnd(zeros(1, size(G_FDE, 1)), G_FDE * N0_FDE(h)/2)).';
            r_hat_FDE = r_hat_FDE(length(g_xcorr_samp) + nu_FDE + 1:end - length(g_xcorr_samp) - nu_FDE);  % Remove half CP from beginning and end of sequence
        end

        %% Preprocessing - CNN
        pre_symbols = r_hat(nSymbolsExtraHalf - NPad + 1:nSymbolsExtraHalf);                % Extract prevoius padding symbols
        rxSymbols = r_hat(nSymbolsExtraHalf + 1:end - nSymbolsExtraHalf);                   % Extract the data symbols
        post_symbols = r_hat(end - nSymbolsExtraHalf + 1:end - nSymbolsExtraHalf + NPad);   % Extract successive padding symbols

        netInput(:, :, h) = [real([pre_symbols; rxSymbols; post_symbols]), ...
                             imag([pre_symbols; rxSymbols; post_symbols])];

        %% Demodulation - Minimum Euclidean Distace benchmark
        llr = qamdemod(rxSymbols, M, 'gray', 'OutputType', 'llr');              % Estimate LLRs
        rxBits = double(llr < 0);                                               % Hard decision for uncoded BER       

        %% Demodulation - FDE benchmark
        % Processing according to [1]
        if toggle_FDE
            y_FDE = conj(Q_FDE) * r_hat_FDE;
            W_FDE = diag(conj(Lambda_FDE) ./ (abs(Lambda_FDE).^2 + N0_FDE(h)));
            rxSymbols_FDE = Q_FDE.' * W_FDE * y_FDE;

            llr_FDE = qamdemod(rxSymbols_FDE, M, 'gray', 'OutputType', 'llr');  % Estimate LLRs
            rxBits_FDE = double(llr_FDE < 0);                                   % Hard decision for uncoded BER
        end

        %% Demodulation - SDR benchmark
        % QPSK only - processing according to [2]
        if toggle_SDR
            y_SDR = [real(rxSymbols); imag(rxSymbols)];
            z_SDR = Gn_SDR\y_SDR;
            Theta_SDR = [Gn_SDR, -y_SDR; -y_SDR.', z_SDR.'*Gn_SDR*z_SDR];

            cvx_begin quiet
                variable Psi_SDR(2*nSymbols+1, 2*nSymbols+1) semidefinite
                expression Psi_11(2*nSymbols, 2*nSymbols)
                expression Psi_12(2*nSymbols, 1)
                expression Psi_21(1, 2*nSymbols)
                expression Psi_22(1)
                Psi_11 = Psi_SDR(1:end-1, 1:end-1);
                Psi_12 = Psi_SDR(1:end-1, end);
                Psi_21 = Psi_SDR(end, 1:end-1);
                Psi_22 = Psi_SDR(end, end);

                minimize( trace(Psi_SDR * Theta_SDR) )

                subject to
                    diag(Psi_11) >= 0.5 .* ones(2*nSymbols, 1)
                    Psi_12 + (1/sqrt(2)).*ones(2*nSymbols, 1) >= 0
                    Psi_12 - (1/sqrt(2)).*ones(2*nSymbols, 1) <= 0
                    Psi_12 == Psi_21.'
                    Psi_22 == 1

            cvx_end
            
            % Gaussian randomization
            samples_SDR = mvnrnd(zeros(1, size(Psi_SDR, 1)), Psi_SDR, L_SDR).';
            quant_SDR = 1/sqrt(2) .* sign(samples_SDR);
            quant_SDR(end, :) = ones(1, L_SDR);
            min_tr_SDR = inf;
            for k_SDR = 1:L_SDR
                tr_SDR = trace(Theta_SDR * quant_SDR(:, k_SDR) * quant_SDR(:, k_SDR).');
                if tr_SDR < min_tr_SDR
                    min_tr_SDR = tr_SDR;
                    rxSymbols_SDR = quant_SDR(1:nSymbols, k_SDR) + 1j.*quant_SDR(nSymbols+1:end-1, k_SDR);
                end
            end
            llr_SDR = qamdemod(rxSymbols_SDR, M, 'gray', 'OutputType', 'llr');  % Estimate LLRs
            rxBits_SDR = double(llr_SDR < 0);                                   % Hard decision for uncoded BER
        end

        %% Demodulation - DNN benchmark
        % Trained for QPSK only - processing according to [3]
        if toggle_DNN
            % Sliding window
            netInput_DNN = zeros(nSymbolsIn_DNN, 2*ceil(nSymbols / nSymbolsOut_DNN));   % Initialize DNN input
            for k = 1:floor(nSymbols / nSymbolsOut_DNN)
                netInput_DNN(:, 2*(k-1) + 1) = real(r_hat_DNN((k-1)*nSymbolsOut_DNN + 1 : (k-1)*nSymbolsOut_DNN + nSymbolsIn_DNN));
                netInput_DNN(:, 2*k) = imag(r_hat_DNN((k-1)*nSymbolsOut_DNN + 1 : (k-1)*nSymbolsOut_DNN + nSymbolsIn_DNN));
            end
            netInput_DNN(:, end-1) = real(r_hat_DNN(end - nSymbolsIn_DNN + 1 : end));   % Adjustment for last window position to coincide with end of codeword
            netInput_DNN(:, end) = imag(r_hat_DNN(end - nSymbolsIn_DNN + 1 : end));

            % Equalization
            netOutput_DNN = predict(bestNet_DNN, netInput_DNN);                            % Equalize received symbols
            netOutput_DNN = netOutput_DNN(:, 1:2:end-1) + 1j * netOutput_DNN(:, 2:2:end);   % Reconstruct received symbols from real and imaginary parts
            rxSymbols_DNN = [reshape(netOutput_DNN(:, 1:end-1), [], 1); ...         % Remove redundant symbols due to mod(nSymbols, nSymbolsOut_DNN) != 0
                             netOutput_DNN(end - mod(nSymbols, nSymbolsOut_DNN) + 1:end, end)];
            
            llr_DNN = qamdemod(rxSymbols_DNN, M, 'gray', 'OutputType', 'llr');  % Estimate LLRs
            rxBits_DNN = double(llr_DNN < 0);                                   % Hard decision for uncoded BER
        end
        
        %% Bit Error accumulation for BER
        nErrors(h) = nErrors(h) + sum(interBits~=rxBits);
        if toggle_FDE
            nErrors_FDE(h) = nErrors_FDE(h) + sum(interBits~=rxBits_FDE);
        end
        if toggle_SDR
            nErrors_SDR(h) = nErrors_SDR(h) + sum(interBits~=rxBits_SDR);
        end
        if toggle_DNN
            nErrors_DNN(h) = nErrors_DNN(h) + sum(interBits~=rxBits_DNN);
        end

        %% De-interleaving
        deinterBits = reshape(reshape(llr, [], m).', [], 1);
        if toggle_FDE
            deinterBits_FDE = reshape(reshape(llr_FDE, [], m).', [], 1);
        end
        if toggle_SDR
            deinterBits_SDR = reshape(reshape(llr_SDR, [], m).', [], 1);
        end
        if toggle_DNN
            deinterBits_DNN = reshape(reshape(llr_DNN, [], m).', [], 1);
        end

        % Add random errors according to theoretical Bit Error Rate and convert in LLRs
        idxErrors = randsample([0, 1], m*nSymbols, true, [1-BER_th(h), BER_th(h)]);
        deinterBits_th = codedBits;
        deinterBits_th(idxErrors > 0) = not(deinterBits_th(idxErrors > 0));
        deinterBits_th = -(2.*deinterBits_th - 1).*LLR_th(h);

        %% Rate recovery
        recovBits = nrldpc_rate_recover(ldpc_info, deinterBits);
        if toggle_FDE
            recovBits_FDE = nrldpc_rate_recover(ldpc_info, deinterBits_FDE);
        end
        if toggle_SDR
            recovBits_SDR = nrldpc_rate_recover(ldpc_info, deinterBits_SDR);
        end
        if toggle_DNN
            recovBits_DNN = nrldpc_rate_recover(ldpc_info, deinterBits_DNN);
        end
        recovBits_th = nrldpc_rate_recover(ldpc_info, deinterBits_th);

        %% Decoding
        decBits = nrldpc_decoder(ldpc_info, recovBits, maxIter_LDPC);
        if toggle_FDE
            decBits_FDE = nrldpc_decoder(ldpc_info, recovBits_FDE, maxIter_LDPC);
        end
        if toggle_SDR
            decBits_SDR = nrldpc_decoder(ldpc_info, recovBits_SDR, maxIter_LDPC);
        end
        if toggle_DNN
            decBits_DNN = nrldpc_decoder(ldpc_info, recovBits_DNN, maxIter_LDPC);
        end
        decBits_th = nrldpc_decoder(ldpc_info, recovBits_th, maxIter_LDPC);

        %% Block Error accumulation for BLER
        nErrBlock(h) = nErrBlock(h) + (sum(bits~=decBits) > 0);
        if toggle_FDE
            nErrBlock_FDE(h) = nErrBlock_FDE(h) + (sum(bits~=decBits_FDE) > 0);
        end
        if toggle_SDR
            nErrBlock_SDR(h) = nErrBlock_SDR(h) + (sum(bits~=decBits_SDR) > 0);
        end
        if toggle_DNN
            nErrBlock_DNN(h) = nErrBlock_DNN(h) + (sum(bits~=decBits_DNN) > 0);
        end
        nErrBlock_th(h) = nErrBlock_th(h) + (sum(bits~=decBits_th) > 0);
    end

    %% Demodulation - CNN
    netOutput = predict(bestNet, netInput);     % Estimate LLRs
        
    for h = 1:length(EsN0)
        llr_CNN = -reshape(netOutput(:, :, h).', [], 1);     % Reshape LLRs and adjust sign
        rxBits_CNN = double(llr_CNN < 0);           % Hard decision for uncoded BER

        nErrors_CNN(h) = nErrors_CNN(h) + sum(interBits~=rxBits_CNN);           % Bit errors count
        deinterBits_CNN = reshape(reshape(llr_CNN, [], m).', [], 1);            % De-interleaving
        recovBits_CNN = nrldpc_rate_recover(ldpc_info, deinterBits_CNN);        % Rate recovery
        decBits_CNN = nrldpc_decoder(ldpc_info, recovBits_CNN, maxIter_LDPC);   % LDPC decoding
        nErrBlock_CNN(h) = nErrBlock_CNN(h) + (sum(bits~=decBits_CNN) > 0);     % Block errors count
    end
end

%% Bit Error Rate
BER = nErrors ./ (i * nCodedBits);
BER_CNN = nErrors_CNN ./ (i * nCodedBits);
if toggle_FDE
    BER_FDE = nErrors_FDE ./ (i * nCodedBits);
end
if toggle_SDR
    BER_SDR = nErrors_SDR ./ (i * nCodedBits);
end
if toggle_DNN
    BER_DNN = nErrors_DNN ./ (i * nCodedBits);
end

%% Block Error Rate
BLER = nErrBlock ./ i;
BLER_CNN = nErrBlock_CNN ./ i;
if toggle_FDE
    BLER_FDE = nErrBlock_FDE ./ i;
end
if toggle_SDR
    BLER_SDR = nErrBlock_SDR ./ i;
end
if toggle_DNN
    BLER_DNN = nErrBlock_DNN ./ i;
end
BLER_th = nErrBlock_th ./ i;

%% Throughput
TP = (1-BLER) * nBitsPerPacket/(Tn * nSymbols * ftnParam);
TP_CNN       = (1-BLER_CNN) * nBitsPerPacket/(Tn * nSymbols * ftnParam);
if toggle_FDE
    TP_FDE = (1-BLER_FDE) * nBitsPerPacket/(Tn * (nSymbols + 2*nu_FDE) * ftnParam);
end
if toggle_SDR
    TP_SDR = (1-BLER_SDR) * nBitsPerPacket/(Tn * nSymbols * ftnParam);
end
if toggle_DNN
    TP_DNN = (1-BLER_DNN) * nBitsPerPacket/(Tn * nSymbols * ftnParam);
end
TP_th = (1-BLER_th) * nBitsPerPacket/(Tn * nSymbols);

clc
disp("Simulation completed!")

%% Save results
toSave = {BER, BER_CNN, BER_th.', BLER, BLER_CNN, BLER_th, TP, TP_CNN, TP_th, EbN0dB.', toggle_FDE, toggle_SDR, toggle_DNN};
if toggle_FDE
    toSave{end+1} = BER_FDE;
    toSave{end+1} = BLER_FDE;
    toSave{end+1} = TP_FDE;
end
if toggle_SDR
    toSave{end+1} = BER_SDR;
    toSave{end+1} = BLER_SDR;
    toSave{end+1} = TP_SDR;
end                 
if toggle_DNN
    toSave{end+1} = BER_DNN;
    toSave{end+1} = BLER_DNN;
    toSave{end+1} = TP_DNN;
end

filename = strcat("Results/Results_", string(ftnParam), "_", string(rollOff), "_", string(Rc), ".mat");
save(filename, "toSave");
