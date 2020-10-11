clc
warning off


%%
% 1. 随机产生训练集/测试集
% a = randperm(208);
P_train = Trail(1:113,1:8);
T_train = Trail(1:113,10);

[p_train, ps_input]= mapminmax(P_train',0,1);
[t_train, ps_output]= mapminmax(T_train',0,1);  
%%
% 3. 测试数据
P_test = Trail(114:end -1,1:8);
T_test = Trail(114:end -1,10);
p_test = mapminmax('apply', P_test',ps_input);
%% III. 创建随机森林和bp回归器
model = regRF_train(P_train,T_train,2000,7);
net = newff(p_train, t_train,[8,8]);

net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;
net.trainParam.epochs = 1000; % 设置神经网络参数
net.trainParam.goal = 1e-3;
net.trainParam.lr = 0.01;

net = train(net, p_train, t_train); % 训练网络

%% IV. 仿真测试
rfPreTr = regRF_predict(P_train,model);
rfPreT = regRF_predict(P_test,model);

bpPretr = sim(net,p_train);  
bpPret = sim(net,p_test );

%反归一化
bpPreTr = mapminmax('reverse', bpPretr, ps_output);
bpPreT = mapminmax('reverse', bpPret, ps_output);


rfPre = [rfPreTr;rfPreT];
bpPre = [bpPreTr,bpPreT]';
%% 计算RMSE和R2
%----计算RMSE
rfRmseTr = sqrt(mse(T_train, rfPreTr));
rfRmseT = sqrt(mse(T_test, rfPreT));
rfRmse = sqrt(mse(Trail(1:end-1,10), rfPre));

%----计算拟合优度R2
rfR2Tr = 1 - rfRmseTr/var(T_train);
rfR2T = 1 - rfRmseT/var(T_test);
rfR2 = 1 - rfRmse/var(Trail(1:end -1,10));

%----计算bp RMSE
bpRmseTr = sqrt(mse(T_train, bpPreTr'));
bpRmseT = sqrt(mse(T_test, bpPreT'));
bpRmse = sqrt(mse(Trail(1:end -1,10), bpPre));

%----计算bp 拟合优度R2
bpR2Tr = 1 - bpRmseTr/var(T_train);
bpR2T = 1 - bpRmseT/var(T_test);
bpR2 = 1 - bpRmse/var(Trail(1:end -1,10));

%% 绘制图像
figure()
scatter(1:length(T_test), T_test,'o', 'r');
hold on 
plot(1:length(T_test), T_test, 'r');
hold on 
scatter(1:length(T_test), rfPreT,'o', 'b');
hold on 
scatter(1:length(T_test), bpPreT,'o', 'black');

%% 计算累计位移R2和RMSE
T_train = accumu(1:113); T_test = accumu(114:end);
rfPreTr = accuRfPre(1:113); rfPreT = accuRfPre(114:end);
bpPreTr = accuBpPre(1:113); bpPreT = accuBpPre(114:end);

rfRmseTr = sqrt(mse(T_train, rfPreTr));
rfRmseT = sqrt(mse(T_test, rfPreT));
rfRmse = sqrt(mse(accumu, accuRfPre));

%----计算拟合优度R2
rfR2Tr = 1 - rfRmseTr/var(T_train);
rfR2T = 1 - rfRmseT/var(T_test);
rfR2 = 1 - rfRmse/var(accumu);

%----计算bp RMSE
bpRmseTr = sqrt(mse(T_train, bpPreTr));
bpRmseT = sqrt(mse(T_test, bpPreT));
bpRmse = sqrt(mse(accumu, accuBpPre));

%----计算bp 拟合优度R2
bpR2Tr = 1 - bpRmseTr/var(T_train);
bpR2T = 1 - bpRmseT/var(T_test);
bpR2 = 1 - bpRmse/var(accumu);