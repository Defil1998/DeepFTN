%% --------------------------------------------------------%%
%                                                           %
% Date:             November, 2024                          %
% Author:           Bruno De Filippo, Ph.D. student         %
% Affiliation:      DEI Department, University of Bologna   %
% Email:            bruno.defilippo@unibo.it                %
% Personal email:   brunodefilippo@gmail.com                %
%                                                           %
%-----------------------------------------------------------%

function [loss, gradients, state] = lossDNNbenchmark(net, X, T)

    % Forward data through network.
    [Y, state] = forward(net, X);
    loss = mse(Y, T);
    
    % Calculate gradients of loss with respect to learnable parameters.
    gradients = dlgradient(loss, net.Learnables);

end