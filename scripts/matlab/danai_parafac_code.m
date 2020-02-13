clear all;close all;clc

%preamble used to "install" the tensor toolbox
mypath = pwd;
cd ..
run install_tensor_toolbox
cd(mypath);
%end of "splat"-able preamble

I = 100; J = 200; K = 300;
F = 10;
nonzeros = 1000;
X = sptenrand([I J K], nonzeros);%create a random tensor
factors = cp_als(X,2);%run PARAFAC without non-negativity
% factors = cp_nmu(X,F);%run PARAFAC with non-negativity
A=sparse(factors.U{1});B=sparse(factors.U{2});C=sparse(factors.U{3});
lambda = factors.lambda;




