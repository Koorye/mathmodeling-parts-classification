clear;
clc;
disp('Enter judging matrix.');
A = input('Matrix: ');

n = size(A, 1);
% Characteristic Value
[V, D] = eig(A);
max_eig = max(D(:));
[~, col] = find(D == max_eig,1);
Weight = V(:, col) ./ sum(V(:, col));
disp('Weight: ');
disp(Weight);

% Consistency Ratio
CI = (max_eig - n) / (n - 1);
RI = [0 0 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49];
CR = CI / RI(n);
disp('CI: ');
disp(CI);
disp('CR: ');
disp(CR);

clear A row col;
disp('Enter topsis matrix.');
A = input('Matrix: ');

[row,col] = size(A);
disp(['Number of objects: ' num2str(row)]);
disp(['Number of tags: ' num2str(col)]);
 
% Forward
disp('--*-- Positive Trend --*--');
disp('Enter col of tag if need positive trend. Enter nothing to exit');
positive_col = input('Col of tag: ');
while ~isempty(positive_col)
    disp('Enter type of positive trend.');
    disp('1. minimal, 2. central, 3.interval.');
    trend_type = input('Type of tag: ');
    while isempty(trend_type)
        disp('ERROR! Need a type of positive trend! Enter again!');
        trend_type = input('Type of tag: ');   
    end
    A(:, positive_col) = positive_trend(A(:, positive_col), trend_type);
    positive_col = input('Col of tag: ');
end

% Standardization
Standard = standard_trend(A);
disp(Standard)
% Weighting
if(length(Weight) ~= col)
    disp('ERROR! The number of weight is not mixed with cols!');
    return;
end
for i=1:length(Weight)
    Standard(:, i) = Standard(:, i) .* Weight(i);
end

disp(Standard)

% Distance
DistanceToMax = sum((Standard - repmat(max(Standard), row, 1)).^2, 2) .^ 0.5;
DistanceToMin = sum((Standard - repmat(min(Standard), row, 1)).^2, 2) .^ 0.5;

% Score
Score = DistanceToMin ./ (DistanceToMax + DistanceToMin);
StandScore = Score / sum(Score);
disp('--*-- Result --*--');
disp('Score: ');
disp(StandScore);
