% Classification using perceptron (apples vs pears)
clc;
clear;
close all;

%% Reading apple images
A1 = imread('apple_04.jpg');
A2 = imread('apple_05.jpg');
A3 = imread('apple_06.jpg');
A4 = imread('apple_07.jpg');
A5 = imread('apple_11.jpg');
A6 = imread('apple_12.jpg');
A7 = imread('apple_13.jpg');
A8 = imread('apple_17.jpg');
A9 = imread('apple_19.jpg');

%% Reading pear images
P1 = imread('pear_01.jpg');
P2 = imread('pear_02.jpg');
P3 = imread('pear_03.jpg');
P4 = imread('pear_09.jpg');

%% Calculating for each image: colour and roundness
% For Apples
hsv_value_A1 = spalva_color(A1);          % color
metric_A1    = apvalumas_roundness(A1);   % roundness

hsv_value_A2 = spalva_color(A2);
metric_A2    = apvalumas_roundness(A2);

hsv_value_A3 = spalva_color(A3);
metric_A3    = apvalumas_roundness(A3);

hsv_value_A4 = spalva_color(A4);
metric_A4    = apvalumas_roundness(A4);

hsv_value_A5 = spalva_color(A5);
metric_A5    = apvalumas_roundness(A5);

hsv_value_A6 = spalva_color(A6);
metric_A6    = apvalumas_roundness(A6);

hsv_value_A7 = spalva_color(A7);
metric_A7    = apvalumas_roundness(A7);

hsv_value_A8 = spalva_color(A8);
metric_A8    = apvalumas_roundness(A8);

hsv_value_A9 = spalva_color(A9);
metric_A9    = apvalumas_roundness(A9);

% For Pears
hsv_value_P1 = spalva_color(P1);
metric_P1    = apvalumas_roundness(P1);

hsv_value_P2 = spalva_color(P2);
metric_P2    = apvalumas_roundness(P2);

hsv_value_P3 = spalva_color(P3);
metric_P3    = apvalumas_roundness(P3);

hsv_value_P4 = spalva_color(P4);
metric_P4    = apvalumas_roundness(P4);

%% Selecting features (colour, roundness) for ALL data (13 samples)

x1 = [ ...
    hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 ...
    hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 ...
    hsv_value_P1 hsv_value_P2 hsv_value_P3 hsv_value_P4];   % color

x2 = [ ...
    metric_A1 metric_A2 metric_A3 metric_A4 ...
    metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 ...
    metric_P1 metric_P2 metric_P3 metric_P4];               % roundness

% Desired output vector T (1 = apple, -1 = pear)
T = [1 1 1 1 1 1 1 1 1 -1 -1 -1 -1]';

disp('All data (x1=color, x2=roundness, T=class):');
training_table = table(x1', x2', T, 'VariableNames', ...
                       {'x1_color', 'x2_roundness', 'T_class'});
disp(training_table);

%% Train/test split
% Indexing in x1/x2/T: apples = 1..9, pears = 10..13
train_idx = [1 2 3 4 10 11 12];    % 4 apples + 3 pears
Ntrain = numel(train_idx);

% For clarity, test on ALL samples (1..13)
test_idx = 1:length(T);

%% Initialize perceptron parameters

w1 = randn(1);
w2 = randn(1);
b  = randn(1);

eta = 0.1;   % learning rate, 0 < eta < 1

%% Training algorithm

maxIter = 10000;
iter = 0;

while (iter < maxIter)
    iter = iter + 1;

    % One epoch over training set
    for k = 1:Ntrain
        n = train_idx(k);      % actual index in full dataset

        % current inputs and desired output
        x1n = x1(n);
        x2n = x2(n);
        dn  = T(n);            % desired output

        % calculate output for current example
        v = x1n*w1 + x2n*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % error for current example
        en = dn - y;

        % update parameters
        w1 = w1 + eta*en*x1n;
        w2 = w2 + eta*en*x2n;
        b  = b  + eta*en;
    end

    % Check total error on training set (for possible early stopping)
    e = 0;
    for k = 1:Ntrain
        n = train_idx(k);
        v = x1(n)*w1 + x2(n)*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end
        e = e + abs(T(n) - y);
    end

    if e == 0
        % all training samples classified correctly
        break;
    end
end

disp('Training finished. Final parameters:')
fprintf('w1 = %.4f\n', w1);
fprintf('w2 = %.4f\n', w2);
fprintf('b  = %.4f\n', b);
fprintf('Iterations = %d\n', iter);

%% TESTING: ALL 13 samples

y_test = zeros(size(T));
for p = test_idx
    v = x1(p)*w1 + x2(p)*w2 + b;
    if v > 0
        y_test(p) = 1;
    else
        y_test(p) = -1;
    end
end

num_errors = sum(y_test ~= T);
fprintf('Test errors on all 13 samples = %d out of %d\n', ...
        num_errors, length(T));

results_table = table((1:length(T))', x1', x2', T, y_test, ...
    'VariableNames', {'Index','x1_color','x2_roundness','T_true','Y_pred'});
disp('Test results (ALL samples):');
disp(results_table);

%% Plot data points and decision boundary

figure;
hold on;
grid on;
title('Perceptron classification (apples / pears)');
xlabel('Color feature (x1)');
ylabel('Roundness feature (x2)');

% True classes
idx_pos = (T == 1);
idx_neg = (T == -1);

% Plot points
plot(x1(idx_pos), x2(idx_pos), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5); 
plot(x1(idx_neg), x2(idx_neg), 'bx', 'MarkerSize', 8, 'LineWidth', 1.5); 

% Decision boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
x1_min = min(x1) - 0.1;
x1_max = max(x1) + 0.1;
x1_line = linspace(x1_min, x1_max, 100);
x2_line = -(w1*x1_line + b)/w2;

plot(x1_line, x2_line, 'k-', 'LineWidth', 2);  % decision line

legend('Apples (+1)', 'Pears (-1)', 'Decision boundary', 'Location', 'best');
hold off;
