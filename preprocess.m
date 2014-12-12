function [new_data] = preprocess(data)

new_data = [];
for i = 1:length(data)
d = data(i,:);
S = mean(d);
dp = d-S;
T = (dp*dp'/256)^0.5;
d = (d-S)/T;
new_data(end+1,:) = d;
end


end