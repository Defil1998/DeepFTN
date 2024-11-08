%% --------------------------------------------------------%%
%                                                           %
% Date:             November, 2024                          %
% Author:           Bruno De Filippo, Ph.D. student         %
% Affiliation:      DEI Department, University of Bologna   %
% Email:            bruno.defilippo@unibo.it                %
% Personal email:   brunodefilippo@gmail.com                %
%                                                           %
%-----------------------------------------------------------%

clear 
close all
clc

seed=14;
rng(seed);
addpath('Functions')
addpath('Models')

%% Parameters
M = 4;                  % Modulation order
nSymbols = 50;          % Number of symbols per transmission
sps_nyquist= 50;        % Oversampling factor

ftnParam = 0.6;         % FTN parameter
rollOff = 0.5;          % SRRC rolloff
NISI = 10;              % One-sided Inter-Symbol Interference (ISI) span
LISI_AI = 12;            % Number of additional pre/post symbols to consider at Rx

EbN0dBRange = -10:10;                           % Range of Eb/N0 training points
weight_AWGN = (1:length(EbN0dBRange)).^2;       % Weights of the Eb/N0 distribution in the training dataset
weight_AWGN = weight_AWGN ./ sum(weight_AWGN);  % Normalize weights

numEpochs = 2000;       % Number of training epochs
batchSize = 4096;       % Number of examples per minibatch
learnRate = 0.01;       % Initial learning rate
l2reg = 1e-4;           % L2 regularization
ESpatience = 150;       % Number of epochs without loss improvement for early stopping
plateauPatience = 50;   % Number of epochs without loss improvement for ReduceLROnPlateau
lrDecay = 10;           % Learning rate reduction factor

m = log2(M);            % Number of bits per symbol
sps_ftn = ftnParam * sps_nyquist;               % Samples per symbol in FTN
EsN0Range = 10.^(EbN0dBRange./10) * m;          % Convert Eb/N0 values in linear Es/N0

%% ISI channel
g = rcosdesign(rollOff, 2*NISI, sps_nyquist);                               % Generate SRRC FIR filter
g_xcorr = xcorr(g);                                                         % Compute cross-correlation of SRRC filter
g_xcorr_samp = g_xcorr((length(g_xcorr) + 1) / 2:sps_ftn:end);              % Sample second half of xcorr at FTN rate
Gn = toeplitz([g_xcorr_samp, zeros(1, nSymbols+(2*LISI_AI)+length(g_xcorr_samp))]);   % ISI channel matrix considering entire ISI span

%% Training
% Set up DNN
dataIn = zeros(nSymbols + 2*LISI_AI, 2, batchSize);    % Add LISI_AI previous symbols to the input window
dataOut = zeros(nSymbols, m, batchSize);

net = dlnetwork(inputLayer(size(dataIn), 'SCB', Name="input"));

layers = [
    convolution1dLayer(4*LISI_AI + 1, 32, "Padding", LISI_AI, Name="Conv1D_1")   % Add padding to get a sequence of length nSymbols
    batchNormalizationLayer
    leakyReluLayer()
    convolution1dLayer(11, 32, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer(Name="lrelu_1")
    
    convolution1dLayer(9, 32, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer()
    convolution1dLayer(9, 32, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer()
    
    additionLayer(2, Name="add_res1")
    
    convolution1dLayer(7, 16, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer(Name="lrelu_2")

    convolution1dLayer(5, 16, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer()
    convolution1dLayer(5, 16, "Padding", "same")
    batchNormalizationLayer
    leakyReluLayer()

    additionLayer(2, Name="add_res2")

    convolution1dLayer(3, m, "Padding","same")
];

net = addLayers(net, layers);
net = connectLayers(net, "input", "Conv1D_1");
net = connectLayers(net, "lrelu_1", "add_res1/in2");
net = connectLayers(net, "lrelu_2", "add_res2/in2");
net = initialize(net);

% Test to ensure dimension compatibility
YPredicted = predict(net, dlarray(dataIn, "SCB"));
loss = crossentropy(YPredicted, dlarray(dataOut, "SCB"));

%% Training
velocity = [];          % Initialize Adam velocity
averageGrad = [];       % Initialize Adam averageGrad
averageSqGrad = [];     % Initialize Adam averageSqGrad

bestLoss = inf;         % Initialize best loss
EScounter = 0;          % Initialize counter for early stopping
plateauCounter = 0;     % Initialize counter for loss plateaus detection

monitor = trainingProgressMonitor( ...
    Metrics="Loss", ...
    Info=["Epoch" "LearnRate" "BestLoss" "EpochsSinceBest"], ...
    XLabel="Iteration");

epochCounter = 0;       % Initialize epochs counter
while epochCounter < numEpochs && ~monitor.Stop     % Loop over epochs

        epochCounter = epochCounter + 1;          % Increase epochs counter
        EsN0 = randsample(EsN0Range, batchSize, true, weight_AWGN);     % Generate new EsN0 values

        for i=1:batchSize
        
            %% Bit source
            bits = round(rand(nSymbols * m, 1));
            
            %% Mapper
            symbols = qammod([round(rand(m*length(g_xcorr_samp), 1)); ...            % Dummy symbols to cover full ISI span
                              round(rand(m*LISI_AI, 1)); ...                         % Dummy symbols to cover CNN padding
                              bits; ...                                            
                              round(rand(m*LISI_AI, 1)); ...
                              round(rand(m*length(g_xcorr_samp), 1))], ...
                              M, 'gray', 'input', 'bit', 'UnitAveragePower', true);

            %% ISI channel
            txSignal = Gn * symbols;    % Apply ISI channel to generated symbols
            
            %% AWGN
            N0 = mean(abs(txSignal(length(g_xcorr_samp) + 1:end - length(g_xcorr_samp))).^2)./EsN0(i);   % Noise power spectral density

            % Add colored noise
            rxSignal = txSignal + (mvnrnd(zeros(1, size(Gn, 1)), Gn * N0/2) + ...
                1i .* mvnrnd(zeros(1, size(Gn, 1)), Gn * N0/2)).';

            %% Data preparation
            pre_symbols = rxSignal(length(g_xcorr_samp)+1:length(g_xcorr_samp)+LISI_AI);
            rxSymbols = rxSignal(length(g_xcorr_samp)+LISI_AI+1:end-length(g_xcorr_samp)-LISI_AI);  % Extract data symbols
            post_symbols = rxSignal(end-length(g_xcorr_samp)-LISI_AI+1:end-length(g_xcorr_samp));

            % Fill input (/output) data matrices with real and imaginary parts of the received symbols (/corresponding bits)
            dataIn(:, 1, i) = real([pre_symbols; rxSymbols; post_symbols]);
            dataIn(:, 2, i) = imag([pre_symbols; rxSymbols; post_symbols]);

            dataOut(:, :, i) = reshape(bits, m, []).';
        end
        
        % Push minibatches to GPU
        dataIn = dlarray(dataIn, 'SCB');
        dataOut = dlarray(dataOut, 'SCB');

        % Evaluate the model gradients, state, and loss using dlfeval and the modelLoss function and update the network state
        [loss, gradients, state] = dlfeval(@lossCNN, net, dataIn, dataOut);
        net.State = state;

        % L2-Regularization
        idx = net.Learnables.Parameter == "Weights";
        gradients(idx, :) = dlupdate(@(g, w) g + l2reg*w, gradients(idx, :), net.Learnables(idx, :));

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
        if plateauCounter >= plateauPatience
            disp("ReduceLRonPlateau: Loss hasn't improved for " + plateauPatience + " epochs.");
            learnRate = learnRate/lrDecay;      % Reduce learning rate
            plateauCounter = 0;                 % Reset counter for plateau detection
        end

        % Update the training progress monitor
        recordMetrics(monitor,epochCounter,Loss=loss);
        updateInfo(monitor, ...
                   Epoch=epochCounter, ...
                   LearnRate=learnRate, ...
                   BestLoss=bestLoss, ...
                   EpochsSinceBest=EScounter);
        monitor.Progress = 100 * epochCounter/numEpochs;
end

bestLoss = gather(bestLoss);
filename = strcat("Models/CNN-", string(ftnParam), "-", string(rollOff), ".mat");

save(filename, "bestNet", "bestLoss");
