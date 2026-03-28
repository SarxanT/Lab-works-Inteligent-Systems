% Lab 1: Apples vs Pears classification using Naive Bayes
clc; 
clear;
close all;

%% 1. Read images

% Apples
A1 = imread('apple_04.jpg');
A2 = imread('apple_05.jpg');
A3 = imread('apple_06.jpg');
A4 = imread('apple_07.jpg');
A5 = imread('apple_11.jpg');
A6 = imread('apple_12.jpg');
A7 = imread('apple_13.jpg');
A8 = imread('apple_17.jpg');
A9 = imread('apple_19.jpg');

% Pears
P1 = imread('pear_01.jpg');
P2 = imread('pear_02.jpg');
P3 = imread('pear_03.jpg');
P4 = imread('pear_09.jpg');

%% 2. Extract features: color (HSV) and roundness

% Apples
hsv_A1 = spalva_color(A1);   met_A1 = apvalumas_roundness(A1);
hsv_A2 = spalva_color(A2);   met_A2 = apvalumas_roundness(A2);
hsv_A3 = spalva_color(A3);   met_A3 = apvalumas_roundness(A3);
hsv_A4 = spalva_color(A4);   met_A4 = apvalumas_roundness(A4);
hsv_A5 = spalva_color(A5);   met_A5 = apvalumas_roundness(A5);
hsv_A6 = spalva_color(A6);   met_A6 = apvalumas_roundness(A6);
hsv_A7 = spalva_color(A7);   met_A7 = apvalumas_roundness(A7);
hsv_A8 = spalva_color(A8);   met_A8 = apvalumas_roundness(A8);
hsv_A9 = spalva_color(A9);   met_A9 = apvalumas_roundness(A9);

% Pears
hsv_P1 = spalva_color(P1);   met_P1 = apvalumas_roundness(P1);
hsv_P2 = spalva_color(P2);   met_P2 = apvalumas_roundness(P2);
hsv_P3 = spalva_color(P3);   met_P3 = apvalumas_roundness(P3);
hsv_P4 = spalva_color(P4);   met_P4 = apvalumas_roundness(P4);

%% 3. Build feature matrix X and label vector T

% x1: color, x2: roundness
x1 = [hsv_A1 hsv_A2 hsv_A3 hsv_A4 hsv_A5 hsv_A6 hsv_A7 hsv_A8 hsv_A9 ...
      hsv_P1 hsv_P2 hsv_P3 hsv_P4];

x2 = [met_A1 met_A2 met_A3 met_A4 met_A5 met_A6 met_A7 met_A8 met_A9 ...
      met_P1 met_P2 met_P3 met_P4];

% Each row of X is [color, roundness] for one fruit
X = [x1' x2'];        % 13 x 2

% Labels: apples = +1, pears = -1
T = [ ones(9,1);     % 9 apples
     -ones(4,1) ];   % 4 pears

N = length(T);

% Show data
disp('Training data (x1=color, x2=roundness, T=class):');
training_table = table(X(:,1), X(:,2), T, ...
    'VariableNames', {'x1_color', 'x2_roundness', 'T_class'});
disp(training_table);

%% 4. Naive Bayes parameter estimation (Gaussian NB)

% Class indices
idxApple = (T == 1);
idxPear  = (T == -1);

X_apple = X(idxApple, :);   % all apple samples
X_pear  = X(idxPear,  :);   % all pear samples

% Priors P(C)
P_apple = sum(idxApple) / N;
P_pear  = sum(idxPear)  / N;

% Means per feature per class: mu_{class, feature}
mu_apple = mean(X_apple, 1);    % [mean_color_apple, mean_round_apple]
mu_pear  = mean(X_pear,  1);    % [mean_color_pear,  mean_round_pear]

% Variances per feature per class: sigma^2_{class, feature}
var_apple = var(X_apple, 0, 1); % [var_color_apple, var_round_apple]
var_pear  = var(X_pear,  0, 1); % [var_color_pear,  var_round_pear]

% Avoid zero variance
eps_val   = 1e-6;
var_apple = var_apple + eps_val;
var_pear  = var_pear  + eps_val;

% 1D Gaussian pdf
gauss = @(x,mu,s2) exp(-(x-mu).^2 ./ (2*s2)) ./ sqrt(2*pi*s2);

%% 5. Classification of all samples with Naive Bayes

y_pred = zeros(N,1);

for i = 1:N
    x = X(i,:);   % [x1, x2] for sample i
    
    % Likelihoods assuming independence (Naive Bayes)
    p_x1_apple = gauss(x(1), mu_apple(1), var_apple(1));
    p_x2_apple = gauss(x(2), mu_apple(2), var_apple(2));
    
    p_x1_pear  = gauss(x(1), mu_pear(1),  var_pear(1));
    p_x2_pear  = gauss(x(2), mu_pear(2),  var_pear(2));
    
    % Scores proportional to posterior probabilities
    score_apple = P_apple * p_x1_apple * p_x2_apple;
    score_pear  = P_pear  * p_x1_pear  * p_x2_pear;
    
    % Choose class with higher posterior score
    if score_apple >= score_pear
        y_pred(i) = 1;
    else
        y_pred(i) = -1;
    end
end

%% 6. Evaluation

num_correct = sum(y_pred == T);
accuracy = num_correct / N * 100;

disp('Sample   True   Predicted');
disp([(1:N)' T y_pred]);

fprintf('Naive Bayes accuracy: %d / %d (%.2f%%)\n', num_correct, N, accuracy);

%% 7. Plot data (apples vs pears)

figure; hold on; grid on;
title('Naive Bayes classification: apples (+1) vs pears (-1)');
xlabel('Color feature (x1)');
ylabel('Roundness feature (x2)');

plot(X(idxApple,1), X(idxApple,2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(X(idxPear,1),  X(idxPear,2),  'bx', 'MarkerSize', 8, 'LineWidth', 1.5);

legend('Apples (+1)', 'Pears (-1)', 'Location', 'best');
hold off;

