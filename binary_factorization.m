clear all; close all;
% load facebook_xiid
addpath('BinaryTensorFactorization');
addpath('Dataset');

% load kinship_idxi
% load nation_idxi
load kinship
% load nation
% load umls

R=20;
numiters=1000;
fraction=1;% ratio between number of zeros and ones in testing data
for k=1:3 
    N(k) = max(id{k}); 
end

trainfraction=0.8;%90 percent as training data
isbatch=0;% 1: batch gibbs; 0: online gibbs
batchsize=floor(length(id{1})*trainfraction/10);%9/1 training/test split, batch size is 10 percent of training data

[U lambda pr eva time_trace] = BTF_OnlineGibbs(N,xi,id,R,batchsize,numiters,isbatch,fraction,trainfraction);