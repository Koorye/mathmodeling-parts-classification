clc;
clear;

disp('Enter judging matrix: ');
A = input('Matrix:');
% A = [  1   1   4 1/3   3;
%        1   1   4 1/3   3
%      1/4 1/4   1 1/3 1/2;
%        3   3   3   1   3;
%      1/3 1/3   2 1/3   1];

n = size(A, 1);

% Arithmetical Average
SumA = sum(A);
SumA = repmat(SumA, n, 1);
Standard = A ./ SumA;
Result = sum(Standard, 2) / n;
disp('The result is: ');
disp(Result);

% Geometric Average
Product = prod(A, 2);
Standard = Product .^ (1 / n);
Result = Standard ./ sum(Standard);
disp('The result is: ');
disp(Result);

% Characteristic Value
[V, D] = eig(A);
max_eig = max(D(:));
[~, col] = find(D == max_eig,1);
Result = V(:, col) ./ sum(V(:, col));
disp('The result is: ');
disp(Result);

% Consistency Ratio
CI = (max_eig - n) / (n - 1);
RI = [0 0 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49];
CR = CI / RI(n);
disp('CI: ');
disp(CI);
disp('CR: ');
disp(CR);
