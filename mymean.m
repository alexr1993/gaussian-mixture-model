function mean = mymean(input)

total = 0;

for i = 1:size(input, 2)
    total = total + input(:,i);
end

mean = total / i;