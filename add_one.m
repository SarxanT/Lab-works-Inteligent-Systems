clc;
clear;
close all;

% features (x1 – color, x2 – roundness)
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
x1 = [hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 hsv_value_A5 ...
      hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 ...
      hsv_value_P1 hsv_value_P2 hsv_value_P3 hsv_value_P4];

x2 = [metric_A1 metric_A2 metric_A3 metric_A4 metric_A5 ...
      metric_A6 metric_A7 metric_A8 metric_A9 ...
      metric_P1 metric_P2 metric_P3 metric_P4];

X = [x1' x2'];                  % 13x2
T = [ones(9,1); -ones(4,1)];    % apples = 1, pears = -1

% split by class
idxA = (T == 1);
idxP = (T == -1);

XA = X(idxA,:);
XP = X(idxP,:);

% priors
P_A = sum(idxA)/length(T);
P_P = sum(idxP)/length(T);

% mean and variance for each feature
muA  = mean(XA,1);              % [muA1 muA2]
muP  = mean(XP,1);              % [muP1 muP2]
varA = var(XA,0,1);             % [varA1 varA2]
varP = var(XP,0,1);             % [varP1 varP2]

% simple 1D Gaussian
gauss = @(x, m, v) exp(-(x-m).^2./(2*v)) ./ sqrt(2*pi*v);

% classification
y = zeros(size(T));

for i = 1:length(T)
    xi = X(i,:);
    
    % apple class probability
    pA = gauss(xi(1), muA(1), varA(1)) * ...
         gauss(xi(2), muA(2), varA(2)) * P_A;
    
    % pear class probability
    pP = gauss(xi(1), muP(1), varP(1)) * ...
         gauss(xi(2), muP(2), varP(2)) * P_P;
    
    if pA >= pP
        y(i) = 1;
    else
        y(i) = -1;
    end
end

% results
disp([T y])
acc = sum(T == y)/length(T)*100;
fprintf('Accuracy = %.2f%%\n', acc);

% plot
figure; hold on; grid on;
plot(XA(:,1), XA(:,2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(XP(:,1), XP(:,2), 'bx', 'MarkerSize', 8, 'LineWidth', 1.5);

mis = (T ~= y);
if any(mis)
    plot(X(mis,1), X(mis,2), 'ko', 'MarkerSize', 10, 'LineWidth', 1.5);
    legend('Apples','Pears','Misclassified','Location','best');
else
    legend('Apples','Pears','Location','best');
end

xlabel('x1 (color)');
ylabel('x2 (roundness)');
title('Gaussian Naive Bayes: apples vs pears');
hold off;
