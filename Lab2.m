clc;
clear all;
close all;

%% Generate Training Data 
x = 0.1:1/22:1;                                     % 20 input samples
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;  % Target function
%figure; 
%plot(x,d,'b-*','LineWidth',1.5,'MarkerSize',8); 
%title('Target Function'); xlabel('x'); ylabel('d(x)'); grid on

%% Initialize Parameters 
% Hidden layer weights (5 neurons, layer 1)
w11_1 = rand(1);   % input -> hidden neuron1
w12_1 = rand(1);   
w13_1 = rand(1);   
w14_1 = rand(1);   
w15_1 = rand(1);   

% Hidden layer biases
b1_1  = rand(1);   
b2_1  = rand(1);  
b3_1  = rand(1);   
b4_1  = rand(1);   
b5_1  = rand(1);   

% Output layer weights (from 5 hidden neurons to output, layer 2)
w11_2 = rand(1);   % hidden1 -> output
w12_2 = rand(1);   
w13_2 = rand(1);   
w14_2 = rand(1);   
w15_2 = rand(1);  

% Output layer bias
b1_2  = rand(1);   % output neuron bias

eta = 0.1;         % learning rate


%% Training Loop 
for ep = 1:1000000
    for i = 1:length(x)

        %% Feedforward 
        % Hidden layer net inputs
        v1_1 = x(i)*w11_1 + b1_1;
        v2_1 = x(i)*w12_1 + b2_1;
        v3_1 = x(i)*w13_1 + b3_1;
        v4_1 = x(i)*w14_1 + b4_1;
        v5_1 = x(i)*w15_1 + b5_1;

        % Hidden layer activations (sigmoid)
        y1_1 = 1/(1 + exp(-v1_1));
        y2_1 = 1/(1 + exp(-v2_1));
        y3_1 = 1/(1 + exp(-v3_1));
        y4_1 = 1/(1 + exp(-v4_1));
        y5_1 = 1/(1 + exp(-v5_1));

        % Output layer
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + y5_1*w15_2 + b1_2;
        y1_2 = v1_2;      % linear activation

        %% Error 
        e = d(i) - y1_2;

        %% Backpropagation
        % Output layer delta
        delta1_2 = e;

        % Hidden layer deltas (sigmoid derivative: y*(1-y))
        delta1_1 = y1_1*(1 - y1_1) * delta1_2 * w11_2;
        delta2_1 = y2_1*(1 - y2_1) * delta1_2 * w12_2;
        delta3_1 = y3_1*(1 - y3_1) * delta1_2 * w13_2;
        delta4_1 = y4_1*(1 - y4_1) * delta1_2 * w14_2;
        delta5_1 = y5_1*(1 - y5_1) * delta1_2 * w15_2;

        %% Weight Updates 
        % Output layer
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w12_2 = w12_2 + eta*delta1_2*y2_1;
        w13_2 = w13_2 + eta*delta1_2*y3_1;
        w14_2 = w14_2 + eta*delta1_2*y4_1;
        w15_2 = w15_2 + eta*delta1_2*y5_1;
        b1_2  = b1_2  + eta*delta1_2;

        % Hidden layer
        w11_1 = w11_1 + eta*delta1_1*x(i);
        w12_1 = w12_1 + eta*delta2_1*x(i);
        w13_1 = w13_1 + eta*delta3_1*x(i);
        w14_1 = w14_1 + eta*delta4_1*x(i);
        w15_1 = w15_1 + eta*delta5_1*x(i);

        b1_1  = b1_1 + eta*delta1_1;
        b2_1  = b2_1 + eta*delta2_1;
        b3_1  = b3_1 + eta*delta3_1;
        b4_1  = b4_1 + eta*delta4_1;
        b5_1  = b5_1 + eta*delta5_1;
    end
end


%% Testing 

x_test = 0.1:1/220:1;    % denser test grid
y_test = (1 + 0.6*sin(2*pi*x_test/0.7) + 0.3*sin(2*pi*x_test))/2;
Y_pred = zeros(1,length(x_test));

for i = 1:length(x_test)
    v1_1 = x_test(i)*w11_1 + b1_1;
    v2_1 = x_test(i)*w12_1 + b2_1;
    v3_1 = x_test(i)*w13_1 + b3_1;
    v4_1 = x_test(i)*w14_1 + b4_1;
    v5_1 = x_test(i)*w15_1 + b5_1;

    y1_1 = 1/(1 + exp(-v1_1));
    y2_1 = 1/(1 + exp(-v2_1));
    y3_1 = 1/(1 + exp(-v3_1));
    y4_1 = 1/(1 + exp(-v4_1));
    y5_1 = 1/(1 + exp(-v5_1));

    Y_pred(i) = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + y5_1*w15_2 + b1_2;
end


%% Visualization 
figure(2);

plot(x,d,'bo','MarkerSize',8,'LineWidth',1.5); hold on
plot(x_test,y_test,'b--','LineWidth',2);
plot(x_test,Y_pred,'r','LineWidth',2);
legend('Training Data','Target Function','MLP Approximation','Location','best');
title('Function Approximation Using MLP (5 Sigmoid Neurons)'); 
xlabel('x'); ylabel('y'); grid on
