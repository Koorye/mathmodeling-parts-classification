disp('Enter the matrix which need positive trend.')
A = input('Matrix: ');
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

disp('Positive Matrix: ');
disp(A);

Standard = standard_trend(A);

disp('Standard Matrix: ');
disp(Standard);