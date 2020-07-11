function Standard = standard_trend(A)
[row, ~] = size(A)
SumA = sum(A);
SumA = repmat(A, row, 1);
Standard = A ./ SumA;
end