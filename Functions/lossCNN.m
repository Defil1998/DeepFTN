%% --------------------------------------------------------%%
%                                                           %
% Date:             November, 2024                          %
% Author:           Bruno De Filippo, Ph.D. student         %
% Affiliation:      DEI Department, University of Bologna   %
% Email:            bruno.defilippo@unibo.it                %
% Personal email:   brunodefilippo@gmail.com                %
%                                                           %
%-----------------------------------------------------------%

function [loss, gradients, state] = lossCNN(net, X, T)

    % Forward pass
    [Y, state] = forward(net, X);
    
    Y = sigmoid(Y);     % Apply sigmoid to obtain bit probability
    loss = crossentropy(Y, T, "ClassificationMode", "multilabel") / (numel(T)/length(T));   % Compute Binary Crossentropy
    
    % Calculate gradients of loss with respect to learnable parameters
    gradients = dlgradient(loss, net.Learnables);
end