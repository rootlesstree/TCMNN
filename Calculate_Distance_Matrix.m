function [Distance_Matrix] = Calculate_Distance_Matrix(data)


Distance_Matrix = zeros(length(data),length(data));

for d = 1:length(data)
    
    for d2 = d+1:length(data)
        
        Distance_Matrix(d,d) = eps;
        Distance_Matrix(d,d2) = distance(data(d,:),data(d2,:));       
        Distance_Matrix(d2,d) = Distance_Matrix(d,d2);
        
    end


end


function t=distance(x,y)
 t = 0;

 for i = 1:length(x)
     t = t + (x(i)-y(i))^2;
 end

end

end