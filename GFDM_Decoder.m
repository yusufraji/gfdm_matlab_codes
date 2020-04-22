%TU Dresden GFDM Demodulator Code Adaptation...
function [EVM, BER, const_real, const_imag] = GFDM_Decoder(GFDM_Rx_real,...
    GFDM_Rx_imag, K, M, Kon, BitRateDefault, SampleRateDefault, ...
    TimeWindow, mu, numBlocks, Pstruct, ZerosAdded, BC, ReceiveQAM, ...
    inputFilename, ChannelLabel, ReceivedBits, DownsampleRate)

% clc;
% clear all;

[filepath,name,ext] = fileparts(inputFilename);
disp(filepath)
[filepath,name,ext] = fileparts(filepath);
disp(filepath)



fullFileName = fullfile(filepath, 'Outputs', ChannelLabel,'Pstruct.mat');
load (fullFileName);

fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'ZerosAdded.mat');
load (fullFileName);

fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'BC.mat');
load (fullFileName);

N = p.K*p.M;
L = p.Kon*p.M;
mu = p.mu;
% numBlocks = 1;
% BitRateDefault = 2.5*(10^9);
% SampleRateDefault = 4*BitRateDefault;
% TimeWindow = 8*mu*512/BitRateDefault;

% N = K*M;
% L = Kon*M;

% load the results of of the modulator as well as the input data to
% compute the BER
fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'GFDMInputData.mat');
load(fullFileName)
fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'GFDMTxreal.mat');
load(fullFileName)
fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'GFDMTximag.mat');
load(fullFileName)
fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'BC.mat');
load(fullFileName)

% Downsampling
GFDM_Rx_real = downsample(GFDM_Rx_real, DownsampleRate);
GFDM_Rx_imag = downsample(GFDM_Rx_imag, DownsampleRate);

% remove zeros
GFDM_Rx_real = GFDM_Rx_real(Zeros_Added/2+1:length(GFDM_Rx_real)-Zeros_Added/2);
GFDM_Rx_imag = GFDM_Rx_imag(Zeros_Added/2+1:length(GFDM_Rx_imag)-Zeros_Added/2);

GFDM_real = reshape(GFDM_Rx_real, N, numBlocks);
GFDM_imag = reshape(GFDM_Rx_imag, N, numBlocks);
GFDM_Complex_Rx = complex(zeros(N, numBlocks));
GFDM_Mod = zeros(N,numBlocks);
GFDM_Mod_real = zeros(N,numBlocks);
GFDM_Mod_imag = zeros(N,numBlocks);
GFDM_Rx = complex(zeros(N,numBlocks));
for block = 1:numBlocks
    GFDM_Rx(:,(block)) = GFDM_real(:,(block))+1i*GFDM_imag(:,(block));
    % Normalisation
    GFDM_Mod(:,block) = sqrt(real(GFDM_Rx(:,block)).^2 + imag(GFDM_Rx(:,block)).^2);
    Max_GFDM_Mod = max(GFDM_Mod(:,block));
    GFDM_Mod_real(:,block) = real(GFDM_Rx(:,block))./Max_GFDM_Mod;
    GFDM_Mod_imag(:,block) = imag(GFDM_Rx(:,block))./Max_GFDM_Mod;
    GFDM_Complex_Rx(:,block) = GFDM_Mod_real(:,block) + 1i*GFDM_Mod_imag(:,block);
end

b = zeros(L, numBlocks);
shm = zeros(L, numBlocks);
Cx_inpData = zeros(L, numBlocks);

for block = 1:numBlocks
    %Receiver zero forcing
    a = do_demodulate(p, GFDM_Complex_Rx(:,(block)), 'ZF');
    b(:,(block)) = do_unmap(p, a);
%     shm(:,(block)) = do_qamdemodulate(b, p.mu);
%     %     remodulate inpData for EVM evaluation
%     Cx_inpData(:,(block)) = do_qammodulate(inpData(:,block), p.mu);
end

A = zeros(L,numBlocks);
A_real = zeros(L,numBlocks);
A_imag = zeros(L,numBlocks);
New_b = zeros(L,numBlocks);


for block = 1:numBlocks
    % Normalisation for constellation plot
    A(:,block) = sqrt(real(b(:,block)).^2 + imag(b(:,block)).^2);
    Max_A = max(A(:,block));
    A_real(:,block) = real(b(:,block))./Max_A;
    A_imag(:,block) = imag(b(:,block))./Max_A;
    New_b(:,block) = A_real(:,block) + 1i*A_imag(:,block);
end

New_b = New_b*abs(real(B_C(1)))/abs(real(New_b(1)));

for block = 1:numBlocks
    %Receiver zero forcing
%     a = do_demodulate(p, GFDM_Complex_Rx(:,(block)), 'ZF');
%     b(:,(block)) = do_unmap(p, a);
    shm(:,(block)) = do_qamdemodulate(New_b, p.mu);
    %     remodulate inpData for EVM evaluation
    Cx_inpData(:,(block)) = do_qammodulate(inpData(:,block), p.mu);
end
% Plot the constellation
constellation = reshape(New_b, 1, L*numBlocks);
% scatterplot(constellation, p.mu);
const = [constellation(:)',zeros(1,(TimeWindow*BitRateDefault/p.mu)...
    -length(constellation))];
const_real = real(const);
const_imag = imag(const);


hEVM = comm.EVM;
% EVM = step(hEVM,  (Cx_inpData(:)),New_b)
EVM = step(hEVM,  B_C,New_b)
disp(['EVM (%) = ' num2str(EVM) '%']);
EVMdB = 20*log10(EVM/100)

fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'ReceiveQAM.mat');
save (fullFileName, 'New_b')
fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'ReceivedBits.mat');
save (fullFileName, 'shm')
% this writes the constellation and transmitted bits to a csv file,
% to be used by the machine learning algorithm for classification 
% using the constellation and transmitted bits

I=real(constellation)';
Q=imag(constellation)';
Bits=inpData;
T = table(I, Q, Bits);
T.Properties.VariableNames = {'I','Q','Bits'};
fullFileName = fullfile(filepath, 'Outputs', ChannelLabel, 'ReceivedConstellation.csv');
writetable (T,fullFileName, 'Delimiter', ',');


% Measure BER with appropriate delay
BitER = comm.ErrorRate;
ber = step(BitER, inpData(:), shm(:));
BER = ber(1);
% Display Bit error
disp(['GFDM Reception BER = ' num2str(ber(1)) ])
disp(['GFDM Reception total error count = ' num2str(ber(2)) ])
disp(['GFDM Reception total bits received = ' num2str(ber(3)) ])

% save(nameMatrixBERatt,'matrixBER')
end