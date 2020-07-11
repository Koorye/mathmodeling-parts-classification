clc;
clear;

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

% Calculate Distance
DistanceToMax = sum((Standard - repmat(max(Standard), row, 1)).^2, 2) .^ 0.5;
DistanceToMin = sum((Standard - repmat(min(Standard), row, 1)).^2, 2) .^ 0.5;

% Calculate Score
Score = DistanceToMin ./ (DistanceToMax + DistanceToMin);
StandScore = Score / sum(Score);
disp('--*-- Result --*--');
disp('Score: ');
disp(StandScore);
