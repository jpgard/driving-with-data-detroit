% ensure working directory is set to location containing
% install_tensor_toolbox and tensor.mat
clear all;close all;clc
%preamble used to "install" the tensor toolbox
run install_tensor_toolbox
% end preable

% code to read MDST data
FILENAME = 'tensor-data/vehicle_year/pre_tensor_vehicle_year_log.dat';
cd ..;cd ..;
data = dlmread(FILENAME, '\t');
X = sptensor(data(:,[1 2 3]), data(:,4));
F = 25;
factors = cp_nmu(X,F);
A=sparse(factors.U{1});B=sparse(factors.U{2});C=sparse(factors.U{3});
lambda = factors.lambda;

% write factor matrices
dlmwrite("tensor-data/vehicle_year/A_vehicle_year_log.txt", full(A));
dlmwrite("tensor-data/vehicle_year/B_vehicle_year_log.txt", full(B));
dlmwrite("tensor-data/vehicle_year/C_vehicle_year_log.txt", full(C));
dlmwrite("tensor-data/vehicle_year/lambda_vehicle_year_log.txt", full(lambda))
% 
% % create single F x time faceted plot showing each factor of C over time
% figure
% for p = 1:F
%     % for line plots instead uncomment below
%     % plot(subplot(F,1,p), C(:,p))
%     bar(subplot(F,1,p), C(:,p))
% end
% saveas(gcf,sprintf('img/factors_time',i),'png');
% 
% % create 3way plots for each F component
% for i = 1:F
%     if nnz(C(:,i))== 1 
%         continue
%     end
%     figure
%    subplot(3,1,1);plot(A(:,i));
%    subplot(3,1,2);plot(B(:,i));
%    subplot(3,1,3);plot(C(:,i));
%     saveas(gcf,sprintf('img/component%d',i),'png');
% end

