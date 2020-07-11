function [positive_col] = positive_trend(col, type)
    if type == 1 % Minimal
        positive_col = max(col) - col;
    elseif type == 2 % Central
        best = input('The best number: ');
        M = max(abs(col-best));
        positive_col = 1 - abs(col-best) / M;
    elseif type == 3 % Interval
        a = input('The lower: ');
        b = input('The upper: ');
        rows = size(col, 1);
        M = max([a-min(col),max(col)-b]);
        positive_col = ones(rows, 1);
        for i = 1:rows
            if col(i) < a
                positive_col(i) = 1 - (a-col(i)) / M;
            elseif col(i) > b
                positive_col(i) = 1 - (col(i)-b) / M;
            end
        end
    else
        disp('ERROR! Unknown type of positive trend!');
        return;
    end
end