%% --------------------------------------------------------%%
%                                                           %
% Date:             November, 2024                          %
% Author:           Bruno De Filippo, Ph.D. student         %
% Affiliation:      DEI Department, University of Bologna   %
% Email:            bruno.defilippo@unibo.it                %
% Personal email:   brunodefilippo@gmail.com                %
%                                                           %
%-----------------------------------------------------------%

%% 
% This code implements the Deep Neural Network (DNN) model proposed in [1].
% The authors did not include an exhaustive list of parameters; hence,
% assumptions were made where necessary.
%
% 1. The input and output length was not specified in the paper. To
%    overcome this, we exploited the reported computational complexity to
%    extrapolate a fair assumption for such parameters. In particular, we
%    used the Microsoft Excel solver function to solve the following
%    equation for nSymbolsIn and nSymbolsOut:
%
%      C = ((nSymbolsIn*N_1) + (N_1*N_2) + (N_2*N_3) + (N_3*N_4) + (N_4*nSymbolsOut)) / nSymbolsOut,
%    
%    where C=8196 represents the number of multiplications per equalized
%    symbol, and N_i represents the number of neurons in the i-th hidden
%    layer (320, 160, 80, and 40, for hidden layer 1 to 4, respectively).
%    Although not univocal, the solution (nSymbolsIn=32, nSymbolsOut=8)
%    seemed to provide a good balance between amount of padding symbols and
%    the overall sequence length. Furthermore, it satisfies the reasonable
%    constraint nSymbolsOut < N_4.
%
% 2. The paper reported a set of learning rates {1e-3, 2e-4, 4e-5} without
%    further specifying which one to use, or in which way to use the
%    entire set. Thus, we used the first value as starting learning rate
%    and implemented a reduction of the learning rate on loss function
%    plateaus with the remaining ones, i.e., the learning rate is reduced
%    by a factor 5 during training if the loss function does not improve
%    for plateauPatience (fixed to 50 as in our work) epochs, at most
%    twice. After two reductions, the learning rate assumes the lowest
%    specified value and does not decrease anymore.
%
% 3. The authors did not mention early stopping or similar techniques;
%    however, it is fair to assume that this was implemented,  as it is
%    one of the most widespread training techniques. We fixed the
%    corresponding patience value to 200 epochs.
%
% 4. The system model presented in [1] does not make comments on
%    implementation aspects, such as the truncation of the autocorrelation
%    function of the ISI channel or the related boundary effect on the 
%    first and last symbols of the transmitted sequence. Hence, to allow
%    for a fair comparison, we here maintained the same assumptions made
%    for our work, i.e., ISI spans NISI=10 Nyquist symbol times on the 
%    original pulse on each side, or, equivalently, 20 Nyquist symbol times
%    on the autocorrelation function of the ISI channel. The effective span
%    of ISI at FTN rate is floor(2*NISI*ftnParam), e.g., 33 symbols for 
%    ftnParam=0.6 and 28 symbols for ftnParam=0.7. We also pad the
%    transmission with enough dummy symbols to ensure that all symbols that
%    are fed to the DNN are affected by the full ISI as in a continuous
%    transmission, removing the boundary conditions on the ISI channel
%    (i.e., the last symbol in a transmission would not be affected by the
%    ISI due to any upcoming symbol, and similarly for the first one). The
%    BER performance we obtain with the trained model does not match that
%    reported in Figure 7 in [1], and we hypothesize that this last 
%    assumption in particular may be the main reason for this mismatch.

%% References
% [1] P. Song, F. Gong, Q. Li, G. Li and H. Ding, "Receiver Design for Faster-Than-Nyquist Signaling: Deep-Learning-Based Architectures," in IEEE Access, vol. 8, pp. 68866-68873, 2020, doi: 10.1109/ACCESS.2020.2986679

clear 
close all
clc

seed = 14;
rng(seed);

% Add folders to path
addpath('Functions')
addpath('Models')

%% Parameters
M = 4;                  % Modulation order
nSymbolsIn = 32;        % Number of input symbols (found through Excel solver)
nSymbolsOut = 8;        % Number of output symbols (found through Excel solver)
sps_nyquist = 50;       % Samples per symbol

ftnParam = 0.6;         % FTN parameter
rollOff = 0.5;          % SRRC rolloff
NISI = 10;              % One-sided Inter-Symbol Interference (ISI) span

EbN0dB = 7.9;           % Eb/N0 [dB] working point for training (set according to [1])
numEpochs = 2000;       % Number of training epochs
batchSize = 2048;       % Number of examples per minibatch
learnRate = 0.001;      % Initial learning rate
ESpatience = 200;       % Number of epochs without loss improvement for early stopping
plateauPatience = 50;   % Number of epochs without loss improvement for ReduceLROnPlateau
lrDecay = 5;            % Learning rate reduction factor
nPlateausMax = 2;       % Max number of learning rate reductions

m = log2(M);                                    % Number of bits per symbol
EsN0 = (10.^(EbN0dB./10)) * m;                  % Convert the Eb/N0 [dB] in linear Es/N0;
sps_ftn = ftnParam * sps_nyquist;               % Samples per symbol in FTN
halfExtraSymb = (nSymbolsIn - nSymbolsOut)/2;   % Half of excess number of symbols in DNN input

%% ISI channel
g = rcosdesign(rollOff, 2*NISI, sps_nyquist);                               % Generate SRRC FIR filter
g_xcorr = xcorr(g);                                                         % Compute cross-correlation of SRRC filter
g_xcorr_samp = g_xcorr((length(g_xcorr) + 1) / 2:sps_ftn:end);              % Sample second half of xcorr at FTN rate
Gn = toeplitz([g_xcorr_samp, zeros(1, nSymbolsIn+length(g_xcorr_samp))]);   % ISI channel matrix considering entire ISI span

%% DNN model
% Set up DNN
dataIn = zeros(nSymbolsIn, batchSize);      % Initialize DNN input
dataOut = zeros(nSymbolsOut, batchSize);    % Initialize DNN output

layers = [  
    inputLayer(size(dataIn), 'CB')
    fullyConnectedLayer(320)
    reluLayer
    fullyConnectedLayer(160)
    reluLayer
    fullyConnectedLayer(80)
    reluLayer
    fullyConnectedLayer(40)
    reluLayer
    fullyConnectedLayer(nSymbolsOut)
];

net = dlnetwork(layers);    % Generate dlnetwork object

%% Training
velocity = [];          % Initialize Adam velocity
averageGrad = [];       % Initialize Adam averageGrad
averageSqGrad = [];     % Initialize Adam averageSqGrad

bestLoss = inf;         % Initialize best loss
EScounter = 0;          % Initialize counter for early stopping
plateauCounter = 0;     % Initialize counter for loss plateaus detection
nPlateausCounter = 0;   % Initialize counter for number of loss plateaus reached

% Start monitor for DL training
monitor = trainingProgressMonitor( ...
    Metrics = "Loss", ...
    Info = ["Epoch" "LearnRate" "BestLoss" "EpochsSinceBest"], ...
    XLabel = "Iteration");

epochCounter = 0;      % Initialize epochs counter
while epochCounter < numEpochs && ~monitor.Stop  % Loop over epochs

        epochCounter = epochCounter + 1;  % Update epoch number

        for i = 1:batchSize/2  % Fill half of minibatch with real part and other half with imaginary part
        
            %% Bit source
            % Generate i.i.d. bits
            bits = round(rand(nSymbolsIn*m, 1));
            
            %% Mapper
            % Map with random padding symbols before and after data symbols to ensure full ISI is considered
            symbols = qammod([round(rand(m*length(g_xcorr_samp), 1)); ...
                              bits; ...
                              round(rand(m*length(g_xcorr_samp), 1))], ...
                            M, 'gray', 'input', 'bit', 'UnitAveragePower', true);
            
            % Extract data symbols to be used as labels for training
            symbolsOut = symbols(length(g_xcorr_samp) + halfExtraSymb + 1:length(g_xcorr_samp) + halfExtraSymb + nSymbolsOut);

            %% ISI channel
            % Apply ISI channel to generated symbols 
            txSignal = Gn * symbols; 
            
            %% AWGN
            Es = mean(abs(txSignal(length(g_xcorr_samp) + 1:end - length(g_xcorr_samp))).^2);   % Average energy per symbol
            N0 = Es/EsN0;                                                                       % Noise power spectral density

            % Add colored noise
            rxSignal = txSignal + (mvnrnd(zeros(1, size(Gn, 1)), Gn * N0/2) + ...
                1i .* mvnrnd(zeros(1, size(Gn, 1)), Gn * N0/2)).';

            %% Data preparation
            rxSymbols = rxSignal(length(g_xcorr_samp)+1:end-length(g_xcorr_samp));     % Extract received data symbols

            % Fill input (/output) data matrices with real and imaginary parts of the received symbols (/equalized data symbols)
            dataIn(:, i) = real(rxSymbols);
            dataIn(:, batchSize/2 + i) = imag(rxSymbols);
            dataOut(:, i) = real(symbolsOut);
            dataOut(:, batchSize/2 + i) = imag(symbolsOut);
        end

        % Push minibatches to GPU
        dataIn = dlarray(dataIn, 'CB');
        dataOut = dlarray(dataOut, 'CB');

        % Evaluate the model gradients, state, and loss using dlfeval and the modelLoss function and update the network state
        [loss, gradients, state] = dlfeval(@lossDNNbenchmark, net, dataIn, dataOut);
        net.State = state;

        % Update the network parameters using the Adam optimizer
        [net,averageGrad,averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, epochCounter, learnRate);

        % Check loss improvement
        if loss < bestLoss
            % Update best loss and model
            bestLoss = loss;
            bestNet = net;

            % Reset counters
            EScounter = 0; 
            plateauCounter = 0;
        else
            % Increase counters
            EScounter = EScounter + 1;
            plateauCounter = plateauCounter + 1;
        end
    
        % Early stopping
        if EScounter >= ESpatience
            disp("Early stopping: Loss hasn't improved for " + ESpatience + " epochs.");
            break;
        end

        % Reduce learning rate if loss plateau has been detected
        if (plateauCounter >= plateauPatience) && (nPlateausCounter < nPlateausMax)
            disp("ReduceLROnPlateau: Loss hasn't improved for " + plateauPatience + " epochs.");
            learnRate = learnRate/lrDecay;              % Reduce learning rate
            plateauCounter = 0;                         % Reset counter for plateau detection
            nPlateausCounter = nPlateausCounter + 1;    % Increase counter for total number of plateaus reached
        end

        % Update the training progress monitor
        recordMetrics(monitor, epochCounter, Loss = loss);
        updateInfo(monitor, ...
                   Epoch=epochCounter, ...
                   LearnRate=learnRate, ...
                   BestLoss=bestLoss, ...
                   EpochsSinceBest=EScounter);
        monitor.Progress = 100 * epochCounter/numEpochs;
end

%% Save results
% Extract the best loss achived
bestLoss = gather(bestLoss);

% Create file name
filename = strcat("Models/DNN-", string(ftnParam), "-", string(rollOff), ".mat");

% Save best network and corresponding loss
save(filename, "bestNet", "bestLoss");
