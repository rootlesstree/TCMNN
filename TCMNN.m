% Load in data

if( ~exist('data','var'))
USPS = load('USPS.mat');
data = USPS.fea;
labels = USPS.gnd;
data = preprocess(data);
end

copy_data = data;

avg_pure_accuracy = 0;
avg_precision = 0;
avg_large_than_one_class = 0;
avg_empty = 0;

for iteration = 1:100

iteration

data = copy_data;
% Random Train Test Split

split_ratio = 0.25;
test_index = randsample(length(data),length(data)*split_ratio);
train_index = [];
for i = 1:length(data)
    if ~ismember(i,test_index)
        train_index(end+1) = i;
    end
end
train_index = train_index';


% Test LMNN

%Get the data out for LMNN learner

X_train = [];
y_train = [];

for i= 1:length(train_index)
c = train_index(i);
X_train(end+1,:) = data(c,:);
y_train(end+1) = labels(c);
end    
X_train = X_train';

L0=pca(X_train)';
[L,~] = lmnn2(X_train, y_train,3,L0,'maxiter',1000,'quiet',1,'outdim',256,'mu',0.5,'validation',0.2,'earlystopping',25,'subsample',0.3);



data = L*data';
data = data';

%if ( ~exist('Dist_Matrix','var'))
Dist_Matrix = Calculate_Distance_Matrix(data);
%end
    






% Grouping training data

class_data = cell(10,1);
index_pointer = cell(10,1);

for i=1:length(train_index)
   d = data(train_index(i),:);
   label = labels(train_index(i));
   class_data{label,1}(end+1,:) = d;
   index_pointer{label,1}(end+1) = train_index(i);
end


% Initialize the container of iinformation

thr = 0.01;
k = 1;
Same_Class_Values = zeros(length(data),1);
Different_Class_Values = zeros(length(data),1);


Final_Same_Class_Values = zeros(length(data),1);
Final_Different_Class_Values = zeros(length(data),1);

alpha_values = zeros(length(train_index),1);

Region_Predictor = cell(length(test_index),1);
Largest_Pvalue_Prediction = cell(length(test_index),1);
% Calculate the Same/Different distance of Training Data


for i = 1:10

    same_class_data = class_data{i,1};
    
    
    for d = 1:length(same_class_data)
        
        
        % Find nearest k values
        same_class_values = [];
        
        for j = 1:length(same_class_data)
            same_class_values(end+1) = Dist_Matrix(index_pointer{i,1}(d),index_pointer{i,1}(j));
        end
        
        same_class_values = sort(same_class_values);
        same_class_values = same_class_values(1:k+1);
        
        
        
        % Find nearest k values from different classes
        
        distance_between_one_different_classes = [];
        
        for dif = 1:10
            % Eliminate same class index
            if (dif == i)
                continue;
            end
            
           different_class_data = class_data{dif,1};
           
           
           different_values = [];
           for j=1:length(different_class_data)
                different_values(end+1) = Dist_Matrix(index_pointer{i,1}(d),index_pointer{dif,1}(j));
           end
           different_values = sort(different_values);
           
           for j = 1:k
                distance_between_one_different_classes(end+1,:) = different_values(k);
           end
           
        end
        
        Same_Class_Values(index_pointer{i,1}(d)) = max(same_class_values);
        Different_Class_Values(index_pointer{i,1}(d)) = min(distance_between_one_different_classes);

    end
    

end



% Caluclate the Dame/Different distance of Testing Data


for i=1:length(test_index)
    
    label = labels(test_index(i));
    p_values = [];
    
    % Assign a class to the test data
    
    for c = 1:10
       
        
        % Calculate the Same Class Distance
        
        same_class_data = class_data{c,1};     
        same_class_values = [];
        
        for j = 1:length(same_class_data)
            same_class_values(end+1) = Dist_Matrix(test_index(i),index_pointer{c,1}(j));
        end
        same_class_values = sort(same_class_values);
        same_class_values = same_class_values(1:k);
        
        % Calculate Different Class Distance
        
        distance_between_one_different_classes = [];
        
        for dif = 1:10
            % Eliminate same class index
            if (dif == c)
                continue;
            end
            
           different_class_data = class_data{dif,1};           
           
           different_values = [];
           for j=1:length(different_class_data)
               different_values(end+1) = Dist_Matrix(test_index(i),index_pointer{dif,1}(j));
           end
           different_values = sort(different_values);
           
           for j = 1:k
               distance_between_one_different_classes(end+1,:) = different_values(k);           
           end
        end

        % Calculate the alpha value for test
        
        alpha_test = same_class_values/min(distance_between_one_different_classes);
        
        
        
        % Adjust the training statistics if neccessary 
        
        for s_d = 1:length(same_class_data)
            ptr = index_pointer{c,1}(s_d);
           Final_Different_Class_Values(ptr) = Different_Class_Values(ptr);
           
           if( Final_Same_Class_Values(ptr) > Dist_Matrix(ptr,test_index(i)) )
               Final_Same_Class_Values(ptr) = Dist_Matrix(ptr,test_index(i));
           else
               Final_Same_Class_Values(ptr) = Same_Class_Values(ptr); 
           end
        end
        
        
        for dif = 1:10
             % Eliminate same class index
            if (dif == c)
                continue;
            end
       
           different_class_data = class_data{dif,1};
            
           for d_d = 1:length(different_class_data)
               ptr = index_pointer{dif,1}(d_d);
              Final_Same_Class_Values(ptr) = Same_Class_Values(ptr);
               
              if (Different_Class_Values(ptr) > Dist_Matrix(ptr,test_index(i)) )
                  Final_Different_Class_Values(ptr) = Dist_Matrix(ptr,test_index(i));
              else
                  Final_Different_Class_Values(ptr) = Different_Class_Values(ptr);
              end
           end
            
        
        
        
        
        end
    
    
       % Calculate the p value of test sample
       
       for j = 1:length(train_index)
          alpha_values(j) = Final_Same_Class_Values(train_index(j))/Final_Different_Class_Values(train_index(j));
       end
    
          p_values(end+1) = length(alpha_values(alpha_values>=alpha_test))/length(alpha_values);
       
    end
    Region_Predictor{i,1} = find(p_values >= thr);
    Largest_Pvalue_Prediction{i,1} = find(p_values==max(p_values));
    
end



disp('---------------DISPLAY RESULT----------------');


accurate_count = 0;
Region_Prediction_Accuracy_count = 0;
multiple_result_count = 0;
error_count = 0;
empty_count = 0;

for i =1:length(test_index)
   
    true_label = labels(test_index(i));
    
    prediction_result = Region_Predictor{i,1};
    
    if( ismember(true_label,prediction_result) | length(prediction_result)==0 )
       Region_Prediction_Accuracy_count = Region_Prediction_Accuracy_count + 1; 
    end
    
    
    if( length(prediction_result) > 1 )
       multiple_result_count = multiple_result_count + 1;
    end
        
    if (length(prediction_result)==0)
       empty_count = empty_count + 1;
    else
    
        if ( Largest_Pvalue_Prediction{i,1} == true_label)
            accurate_count = accurate_count + 1;
        else
            error_count = error_count + 1;
        end
    end
end


fprintf('Accuracy Rate  = %f\n',accurate_count/(length(test_index)-empty_count));

fprintf('Region Prediction Accuracy Rate = %f\n',Region_Prediction_Accuracy_count/length(test_index));

fprintf('Error Rate = %f\n',error_count/(length(test_index)-empty_count));

fprintf('Empty Rate = %f\n',empty_count/length(test_index));

fprintf('> 1 class Rate = %f\n',multiple_result_count/length(test_index));



avg_pure_accuracy = avg_pure_accuracy + accurate_count/(length(test_index)-empty_count);
avg_precision = avg_precision + Region_Prediction_Accuracy_count/length(test_index);
avg_large_than_one_class = avg_large_than_one_class + multiple_result_count/length(test_index);
avg_empty = avg_empty + empty_count/length(test_index);

end
