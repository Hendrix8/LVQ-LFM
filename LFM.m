clear;
clc;

close all; 
format compact;

dbstop if error; 
dbstop if warning; 

file_name = 'WINE.TXT';
file = fopen(file_name);

data = extract_data(file, file_name, 3);

% shuffling the data matrix in order to simulate random choice of the data
data_shuffled = data(randperm(size(data,1)),:);

% displaying shuffled data 
disp(data_shuffled)

if isequal(file_name, 'WINE.TXT')
    X = data_shuffled(:, 1:13);
    y = data_shuffled(:, 14:16);

elseif isequal(file_name, 'IRIS.TXT')
    X = data_shuffled(:, 1:4); 
    y = data_shuffled(:, 5:7);
end

ppc_list = [1, 2, 5]; % all possible values for ppc
epoch_list = [5, 10, 15]; % all possible values for epochs
Ka_list = [0.01, 0.1]; % all possible values for Ka

num_folds = 10; % number of folds
a0 = 0.5; % initial learning rate value 

% initializing the lists that will contain the results of the experiments
ppc_values = [];
epoch_values = [];
Ka_values = [];
acc_validation_values = [];
acc_test_values = [];
acc_train_values = [];
time = [];

test_acc_graph = {};
train_acc_graph = {};
val_acc_graph = {};

% making all the experiments 
for i = 1:size(epoch_list, 2)     
    for j = 1:size(ppc_list, 2)
        for k = 1:size(Ka_list, 2)
            tic
            % initializing parameters 
            epochs = epoch_list(i);
            ppc = ppc_list(j);
            Ka = Ka_list(k);

            % initializing lists that will contain the accuracies for the
            % specific hyperparameters 
            train_accuracies = zeros(1,10);
            test_accuracies = zeros(1,10);
            validation_accuracies = zeros(1,10);
            
            test_epoch_acc = []; % will contain the test accuracies in order to make a graph 
            train_epoch_acc = [];
            val_epoch_acc = [];

            for fold = 1: num_folds
                [X_train, y_train, X_val, y_val, X_test, y_test] = data_split(X, y, fold); % splitting data with respect to the current fold
                
                [W, test_accs, train_accs, val_accs] = LeFM(X_train, y_train, X_test, y_test, X_val, y_val, epochs, Ka, a0, ppc); % training using LFM
                
                acc_train = accuracy(W,X_train, y_train, ppc); % calculating accuracy on training set
                acc_test = accuracy(W, X_test,y_test, ppc); % calculating accuracy on test set
                acc_val = accuracy(W, X_val, y_val, ppc); % calculating accuracy on validation set
                
                % calculating accuracies for the specific fold 
                train_accuracies(fold) = acc_train; 
                test_accuracies(fold) = acc_test;
                validation_accuracies(fold) = acc_val;

                test_epoch_acc = [test_epoch_acc; test_accs];
                train_epoch_acc = [train_epoch_acc; train_accs];
                val_epoch_acc = [val_epoch_acc; val_accs];

            end

            % adding the mean of the accuracies of the 10 folds
            test_acc_graph = cat(2, test_acc_graph, mean(test_epoch_acc, 1)); 
            train_acc_graph = cat(2, train_acc_graph, mean(train_epoch_acc, 1)); 
            val_acc_graph = cat(2, val_acc_graph, mean(val_epoch_acc, 1)); 

            % calculating accuracies as the mean of all the fold accuracies
            total_train_acc = mean(train_accuracies);
            total_test_acc = mean(test_accuracies);
            total_val_acc = mean(validation_accuracies);

            % adding the result values to the value lists 
            ppc_values(end + 1) = ppc;
            epoch_values(end + 1) = epochs;
            Ka_values(end + 1) = Ka;
            acc_validation_values(end + 1) = total_val_acc;
            acc_test_values(end + 1) = total_test_acc;
            acc_train_values(end + 1) = total_train_acc;
            
            elapsed_time = toc; % calculating time
            time(end + 1) = elapsed_time;
        end
    end
end

% finding the best performance index based on the validation set values
best_idx = find(acc_validation_values == max(acc_validation_values), 1);

fprintf('---------------------------------------------------')
fprintf('\nBEST VALIDATION SET PERFORMANCE STATS : \n')
fprintf('index = %d \n', best_idx);
fprintf('ppc = %d \n', ppc_values(best_idx));
fprintf('epochs = %d \n', epoch_values(best_idx));
fprintf('Ka = %.2f \n', Ka_values(best_idx));
fprintf('validation accuracy = %.3f \n', acc_validation_values(best_idx));
fprintf('test accuracy = %.3f \n', acc_test_values(best_idx));
fprintf('training accuracy = %.3f \n', acc_train_values(best_idx));
fprintf('test accuracies per epoch =');
disp(test_acc_graph{best_idx});
fprintf('elapsed time = %.3f sec \n', time(best_idx));
fprintf('---------------------------------------------------')

% Create a figure to display the test accuracies plot per epochs 
p = figure('Name', 'Test Accuracies Per Epoch');

test_ac = test_acc_graph{best_idx}; % accuracies for the best performance
train_ac = train_acc_graph{best_idx}; % accuracies for the best performance
val_ac = val_acc_graph{best_idx}; % accuracies for the best performance

epochs = 1:size(test_ac, 2); % epochs

plot(epochs, test_ac, 'o-') % connecting the dots in a plot 
hold on;
plot(epochs, train_ac, 'o-') % connecting the dots in a plot 
hold on;
plot(epochs, val_ac, 'o-') % connecting the dots in a plot 

legend('Test', 'Training', 'Validation', 'Location', 'northwest' )

% labeling the axes
xlabel('Epoch') 
ylabel('Accuracy')

% plot title 
title(sprintf('Experiment with best performance ( index = %d )', best_idx));


saveas(gcf, strcat('LFM_plot_',file_name(1:4), '.pdf'), 'pdf'); 




% Create a figure to display the uitable
f = figure('Name', 'LFM');
set(f, 'Position', [100 100 650 600])

column_names = {'epochs', 'ppc', 'Ka', 'Val acc', 'Test acc', 'Train acc', 'Time'}; % create cell array of column names
table_data = [epoch_values ; ppc_values ; Ka_values ; acc_validation_values ; acc_test_values ; acc_train_values ; time]; % creating a matrix with all the data as columns

t = uitable('Data', table_data', 'ColumnName', column_names); % create a table and specify the data and column names
t.Position = [30 30 575 370]; % set the position and size of the table

% adding a text box to the figure with the desired name
annotation('textbox', [0.37 0.9 0.1 0.1], 'String', strcat('LFM-', file_name), 'FontSize', 10, 'FontWeight', 'bold');


saveas(gcf, strcat('LFM_', file_name(1:4), '.pdf'), 'pdf'); % gcf refers to the current figure


% ---------------------------------------------------------------------------
%                                FUNCTIONS
% ---------------------------------------------------------------------------
function accuracy = accuracy(W, X, y, ppc)

    y_pred = []; % it will contain the predictions for X_test
    X_dim = size(X);
    W_dim = size(W);
    
    for i = 1:X_dim(1)
    
        p_dists = []; % list that contains all distances of the current X(i) from the prototypes
        for j = 1:W_dim(2)
            dist = eucl(X(i, :), transpose(W(:, j))); % finding all the distances between the prototypes and the current X(i)
            p_dists(end + 1) = dist; % adding the distance to the list of distances
            
        end
        winner_dist = min(p_dists); % winner distance with the X(i)
    
        % winner idx :  if index is <= ppc then class = [1 0 0],
        %               if index is  ppc < idx <= 2 * ppc then class = [0 1 0]
        %               if index is  2 * ppc <= idx then class = [0 0 1]
        winner_idx = find(p_dists == winner_dist, 1); 
    
        if winner_idx <= ppc
            y_pred = [y_pred ; 1, 0, 0];
    
        elseif winner_idx >=  ppc && winner_idx <= 2 .* ppc
            y_pred = [y_pred ; 0, 1, 0];
    
        else
            y_pred = [y_pred ; 0, 0, 1];
        end
    
    end
    
    y_pred_dim = size(y_pred);
    y_pred_size = y_pred_dim(1);
    y_pred_correct = 0;
    for i = 1:y_pred_size
        if y_pred(i, :) == y(i, :)
            y_pred_correct = y_pred_correct + 1;
        end
    end
    
    accuracy = y_pred_correct ./ y_pred_size;

end

% LFM training function
function [W, test_accs, train_accs, val_accs] = LeFM(X, y, X_test, y_test, X_val, y_val, epochs, Ka, a0, ppc )

    % initializing W
    W = init_W(X, y, ppc);

    test_accs = []; % gathering the test accuracies per epoch for the graph in the end 
    train_accs = [];
    val_accs = [];

    X_dim = size(X); % X dimensions
    W_dim = size(W); % W dimensions
    n_pat = X_dim(1); % number of rows 

    for epoch = 1:epochs
        
        % looping through all rows of X
        for i = 1:n_pat

            p_dists = []; % list that contains all distances of the current X(i) from the prototypes
            for j = 1:W_dim(2)
                dist = eucl(X(i, :), transpose(W(:, j))); % finding all the distances between the prototypes and the current X(i)
                p_dists(end + 1) = dist; % adding the distance to the list of distances
                
            end
    
            d1 = min(p_dists); % finding the distance of X(i) and the closest prototype
            c1_idx = find(p_dists == d1, 1);  % finding the index of the closest prototype to the current X(i)
           
            closest_class = find_class(c1_idx, ppc); % finding the class of the closest prototype 
           
            Xi_class = y(i, :); % getting the class of the current X(i)
           
            if all(Xi_class == closest_class) % if it belongs to the same class
                continue; % do nothing 
            else % if they are on different classes 
                  % UPDATE STEP
                  W(:, c1_idx) = W(:, c1_idx) - a(a0, Ka, i) .* (transpose(X(i, :)) - W(:, c1_idx) ); % move away         
            end

            % creating a list that contains the prototypes that have the
            % same class with the training point 
            same_class = [];
            idxs = []; % contains the indexes of the prototypes in respect to the W vector
            for  idx = 1:size(p_dists, 2)
                c = find_class(idx, ppc); % finding the class of the current prototype
                if all(Xi_class == c) % if it is on the same class as X(i)
                    idxs(end + 1) = idx; % adding the index
                    same_class  = [same_class, W(:, idx)];  % add prototype to the list
                end
    
            end

            p_dists_same = []; % this will contain the distances of prototypes of the same class
            for j = 1:size(same_class, 2)
                dist = eucl(X(i, :), same_class(:, j)' ); % finding all the distances between the prototypes and the current X(i)
                p_dists_same(end + 1) = dist; % adding the distance to the list of distances
                
            end

            d2 = min(p_dists_same); % finding the closest prototype of the same class
            c2_idxx = find(p_dists_same == d2, 1); % finding the index of the closest same class point
            c2_idx = idxs(find(idxs == c2_idxx, 1)); % finding the index of the closest same class point in respect to the W vector
            
            W(:, c2_idx) = W(:, c2_idx) + a(a0, Ka, i) .* (transpose(X(i, :)) - W(:, c2_idx) ); % bring closer         
    
        end
        test_accs(end + 1) = accuracy(W, X_test, y_test, ppc);
        train_accs(end + 1) = accuracy(W, X, y, ppc);
        val_accs(end + 1) = accuracy(W, X_val, y_val, ppc);
    end

end

function c = find_class(c_idx, ppc) % finds a class of a prototype 
    % determining closest point class
    if c_idx <= ppc
        c = [1, 0, 0];
    
    elseif c_idx >=  ppc && c_idx <= 2 .* ppc
        c = [0, 1, 0];
    
    else
        c = [0, 0, 1];
    end
end

% creating the learning rate function 
function a_t = a(a0, Ka, t)
    a_t = a0 / (1 + Ka .* t);
end

function W = init_W(X, y, ppc)

    % dimensions of X and y 
    X_dim = size(X);
    y_dim = size(y);
    
    % input and outpout sizes 
    nin = X_dim(2);
    nout = y_dim(2);

    % initializing prototype vectors 
    W = zeros(nin, nout .* ppc); % creating an empty matrix which will have the prototype vectors as columns
    
    
    w_idx = 1; % this index serves as an index for W to put the new columns
    for i = 1:nout
        [r, c] = find(y(:, i) == 1); % rows and columns of class i
        
        for j = 1:ppc
            idx = randperm(length(r), 1); % choosing a random index of r array 
            w = X(r(idx), :); % using the random index to get a random row of the chosen class
            W(:,w_idx) = w; % adding the chosen row as a column in the W matrix
            r = r(r~=r(idx)); % removing the row so that it does not get chosen again   
            w_idx = w_idx + 1; % adding to the index to replace the next column of W
        end
    end
    
    
    % Checking if W columns are correct
%     for i =1:nout .* ppc
%         [result, result_loc] = ismember(transpose(W(:,i)),X,'rows');
%         y(result_loc, :)
%     end

end

% measures the euclidean distance of two vectors
function dist = eucl(x, y)
    dist = sqrt(sum((x - y).^2));
end

% performs train - test - split division for the dataset, choice of the
% parameter "division" will return a different choice of quantization
% which is going to be helpful for the cross-validation part
function [X_train, y_train, X_val, y_val, X_test, y_test] = data_split(X, y, division)

    X_dim = size(X); % dimensions of the data
    rows_len = X_dim(1); % length of the rows of the data

    train_len = floor(rows_len .* 0.7); % train length 70%
    val_len = ceil(rows_len .* 0.2); % validation length 20%
    test_len = ceil(rows_len .* 0.1); % test set 10%
    
    switch division 
        case 1

            % ------------------ 1st Division ------------------
        
            % defining the training sets
            X_train = X(1: train_len,:);
            y_train = y(1: train_len, :);
            
            % defining the validation sets
            X_val = X(train_len + 1: train_len + val_len, :);
            y_val = y(train_len + 1: train_len + val_len, :);
            
            % defining the test sets 
            X_test = X(train_len + val_len + 1: train_len + val_len + test_len, :);
            y_test = y(train_len + val_len + 1: train_len + val_len + test_len, :);

            % --------------------------------------------------
         
        case 2

            % ------------------ 2nd Division ------------------
        
            % defining the training sets
            X_train = X(test_len + 1: train_len + test_len,:);
            y_train = y(test_len + 1: train_len + test_len, :);
            
            % defining the validation sets
            X_val = X(train_len + test_len + 1: train_len + val_len + test_len, :);
            y_val = y(train_len + test_len + 1: train_len + val_len + test_len, :);
            
            % defining the test sets 
            X_test = X(1: test_len, :);
            y_test = y(1: test_len, :);
                       
            % --------------------------------------------------

        case 3

            % ------------------ 3rd Division ------------------
        
            % defining the training sets
            X_train = X(floor(val_len / 2) + test_len + 1: floor(val_len / 2) + test_len + train_len, :);
            y_train = y(floor(val_len / 2) + test_len + 1: floor(val_len / 2) + test_len + train_len, :);
            
            % defining the validation sets
            X_val =[ X(floor(val_len / 2) + test_len + train_len + 1: train_len + val_len + test_len, :); 
                     X(1:floor(val_len / 2 ), :) ];
            y_val = [y(floor(val_len / 2) + test_len + train_len + 1: train_len + val_len + test_len, :);
                     y(1:floor(val_len / 2 ), :)];
            
            % defining the test sets 
            X_test = X(floor(val_len / 2) + 1: floor(val_len / 2) + test_len, :);
            y_test = y(floor(val_len / 2) + 1: floor(val_len / 2) + test_len, :);
                       
            % --------------------------------------------------

        case 4 

            % ------------------ 4th Division ------------------
        
            % defining the training sets
            X_train = X(val_len + test_len + 1: val_len + test_len + train_len, :);
            y_train = y(val_len + test_len + 1: val_len + test_len + train_len, :);
            
            % defining the validation sets
            X_val = X(1: val_len, :); 
            y_val = y(1: val_len, :);
                     
            
            % defining the test sets 
            X_test = X(val_len + 1: val_len + test_len, :);
            y_test = y(val_len + 1: val_len + test_len, :);
                       
            % --------------------------------------------------

        case 5 

            % ------------------ 5th Division ------------------
        
            % defining the training sets
            X_train = [X(1: floor(train_len / 7), :); 
                       X(floor(train_len / 7) + test_len + val_len + 1: ...
                       test_len + val_len + train_len, :)];

            y_train = [y(1: floor(train_len / 7), :); 
                       y(floor(train_len / 7) + test_len + val_len + 1: ...
                       test_len + val_len + train_len, :)];
            
            % defining the validation sets
            X_val = X(floor(train_len / 7) + 1: floor(train_len / 7) + val_len, :); 
            y_val = y(floor(train_len / 7) + 1: floor(train_len / 7) + val_len, :);
                     
            
            % defining the test sets 
            X_test = X(floor(train_len / 7) + val_len + 1: floor(train_len / 7) + val_len + test_len, :);
            y_test = y(floor(train_len / 7) + val_len + 1: floor(train_len / 7) + val_len + test_len, :);
                       
            % --------------------------------------------------

        case 6

            % ------------------ 6th Division ------------------
        
            % defining the training sets
            X_train = [X( floor((2/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       X(1:floor((2/7) .* train_len), :)];

            y_train = [y( floor((2/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       y(1:floor((2/7) .* train_len), :)];
            % defining the validation sets
            X_val = X(floor((2/7) .* train_len) + 1: floor((2/7) .* train_len) + val_len, :); 
            y_val = y(floor((2/7) .* train_len) + 1: floor((2/7) .* train_len) + val_len, :);
                     
            
            % defining the test sets 
            X_test = X(floor((2/7) .* train_len) + val_len + 1: floor((2/7) .* train_len) + val_len + test_len, :);
            y_test = y(floor((2/7) .* train_len) + val_len + 1: floor((2/7) .* train_len) + val_len + test_len, :);
                       
            % --------------------------------------------------

        case 7

            % ------------------ 7th Division ------------------
        
            % defining the training sets
            X_train = [X( floor((3/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       X(1:floor((3/7) .* train_len), :)];

            y_train = [y( floor((3/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       y(1:floor((3/7) .* train_len), :)];
            % defining the validation sets
            X_val = X(floor((3/7) .* train_len) + 1: floor((3/7) .* train_len) + val_len, :); 
            y_val = y(floor((3/7) .* train_len) + 1: floor((3/7) .* train_len) + val_len, :);
                     
            
            % defining the test sets 
            X_test = X(floor((3/7) .* train_len) + val_len + 1: floor((3/7) .* train_len) + val_len + test_len, :);
            y_test = y(floor((3/7) .* train_len) + val_len + 1: floor((3/7) .* train_len) + val_len + test_len, :);
                       
            % --------------------------------------------------

        case 8 

            % ------------------ 8th Division ------------------
        
            % defining the training sets
            X_train = [X( floor((4/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       X(1:floor((4/7) .* train_len), :)];

            y_train = [y( floor((4/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       y(1:floor((4/7) .* train_len), :)];
            % defining the validation sets
            X_val = X(floor((4/7) .* train_len) + 1: floor((4/7) .* train_len) + val_len, :); 
            y_val = y(floor((4/7) .* train_len) + 1: floor((4/7) .* train_len) + val_len, :);
                     
            
            % defining the test sets 
            X_test = X(floor((4/7) .* train_len) + val_len + 1: floor((4/7) .* train_len) + val_len + test_len, :);
            y_test = y(floor((4/7) .* train_len) + val_len + 1: floor((4/7) .* train_len) + val_len + test_len, :);
                       
            % --------------------------------------------------

        case 9

            % ------------------ 9th Division ------------------
        
            % defining the training sets
            X_train = [X( floor((5/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       X(1:floor((5/7) .* train_len), :)];

            y_train = [y( floor((5/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       y(1:floor((5/7) .* train_len), :)];
            % defining the validation sets
            X_val = X(floor((5/7) .* train_len) + 1: floor((5/7) .* train_len) + val_len, :); 
            y_val = y(floor((5/7) .* train_len) + 1: floor((5/7) .* train_len) + val_len, :);
                     
            
            % defining the test sets 
            X_test = X(floor((5/7) .* train_len) + val_len + 1: floor((5/7) .* train_len) + val_len + test_len, :);
            y_test = y(floor((5/7) .* train_len) + val_len + 1: floor((5/7) .* train_len) + val_len + test_len, :);
                       
            % --------------------------------------------------
            
        otherwise 

            % ------------------ 8th Division ------------------
        
            % defining the training sets
            X_train = [X( floor((6/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       X(1:floor((6/7) .* train_len), :)];

            y_train = [y( floor((6/7) .* train_len) + val_len + test_len + 1: val_len + test_len + train_len, :);
                       y(1:floor((6/7) .* train_len), :)];
            % defining the validation sets
            X_val = X(floor((6/7) .* train_len) + 1: floor((6/7) .* train_len) + val_len, :); 
            y_val = y(floor((6/7) .* train_len) + 1: floor((6/7) .* train_len) + val_len, :);
                     
            
            % defining the test sets 
            X_test = X(floor((6/7) .* train_len) + val_len + 1: floor((6/7) .* train_len) + val_len + test_len, :);
            y_test = y(floor((6/7) .* train_len) + val_len + 1: floor((6/7) .* train_len) + val_len + test_len, :);
                       
            % --------------------------------------------------
            
    end


end


function data = extract_data(file, file_name, spare_lines)
    
    % throwing away spare lines
    for i = 1:spare_lines
        fgetl(file);
    end
    content = readlines(file_name);
   
    rows_len = length(content) - spare_lines;
    cols_len = length(regexp(content(spare_lines + 1),'\d+.\d+','Match'));

    
    data = []; % creating an empty matrix which will hold the features(inputs) and the labels(outputs)
    
    for i = 1:rows_len
        line = fgetl(file); % getting the current line 
    
        row = []; % contains the numbers of the row
        for j = 1:cols_len
            % getting number
            cell = regexp(line,'\d+.\d+','Match');
            str = cell{j};
            num = str2num(str); 
    
            % constructing the row
            row = [row num]; % adding the number to the row 
            
        end
        
        % adding the constructed rows to the data matrix
        data = [data; row];
    
    end

end