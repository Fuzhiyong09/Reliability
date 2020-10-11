clc
warning off
%-----------随机样本组合下预测模型的可靠度问题----------------------%
clear Pf_m
%% 输入抽取样本个数
M = 125; 
nsample = 100;
j = 1;

for i = 1:nsample
	%% 建立随机训练集脚标
    K = 15 ; %测试集26< = K< =40
    a = (randperm(125))';%原始数据序号是从41开始的所以加40，生成41:120的乱序序列
    b = a(1:(M-K),:);% 获取训练集脚标
    c = a(M-K+1:end,:); % 获取测试集脚标
	
	%% 获取训练集
    P_train = Trail(b,1:8); %确定训练集的诱发因素集合 
    T_train = Trail(b,10);%获取位移训练集
    
    [p_train, ps_input]= mapminmax(P_train',0,1);  %输入数据归一化
    [t_train, ps_output]= mapminmax(T_train',0,1); 
    %% 获取测试集
    P_test = Trail(c,1:8); %确定测试集诱发因素集合
    T_test = Trail(c,10);  %获取归一化后的位移测试集    
    
    p_test = mapminmax('apply', P_test',ps_input);  %测试数据反归一化
    %% 创建随机森林回归器
    model = regRF_train(P_train,T_train,1000,7); 
    net = newff(p_train, t_train,[8,8]);
   
    % 构建神经网络
    net.trainParam.showWindow = false;  % 关闭弹窗
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = 1000; % 设置神经网络参数
    net.trainParam.goal = 1e-3;
    net.trainParam.lr = 0.01;

    net = train(net, p_train, t_train); % 训练网络
    
    %% 位移预测
    rfPreTr = regRF_predict(P_train,model);  
    rfPreT = regRF_predict(P_test,model);
    
    bpPretr = sim(net,p_train);  
    bpPret = sim(net,p_test );
    
    %输出数据反归一化
    bpPreTr = mapminmax('reverse', bpPretr, ps_output);
    bpPreT = mapminmax('reverse', bpPret, ps_output);
    
	%% 蒙特卡洛模拟计算
	Gx = zeros(K,1);Tx = zeros(K,1);Fx = zeros(K,1); %初始化参数
	
	%求解功能函数
	for n = 1:K; %不同情况下极限状态函数
		if rfPreT(n) * T_test(n)<0;
		  Gx(n) = 0;
		  Tx(n) = 1;
		elseif abs(rfPreT(n))> abs(T_test(n)); %测试集位移绝对值大于原始位移绝对值
		  Gx(n) = abs(T_test(n))/abs(rfPreT(n));
		  Tx(n) = 1 - Gx(n);
		else  %测试集位移绝对值大于原始位移绝对值
		  Gx(n) = abs(rfPreT(n))/abs(T_test(n));
		  Tx(n) = 1 - Gx(n);
		end
	
	    Fx(n) = 0.25* Gx(n)./Tx(n) - 1;
	    if Fx(n) < 0
			Rorf(j) = 1;
	    else 
			Rorf(j) = 0;
        end 
        
		if bpPreT(n) * T_test(n)<0;
		  Gxb(n) = 0;
		  Txb(n) = 1;
		elseif abs(bpPreT(n))> abs(T_test(n)); %测试集位移绝对值大于原始位移绝对值
		  Gxb(n) = abs(T_test(n))/abs(bpPreT(n));
		  Txb(n) = 1 - Gxb(n);
		else  %测试集位移绝对值大于原始位移绝对值
		  Gxb(n) = abs(bpPreT(n))/abs(T_test(n));
		  Txb(n) = 1 - Gxb(n);
		end
	
	    Fxb(n) = 0.25*Gxb(n)./Txb(n) - 1;
	    if Fxb(n) < 0
			Robp(j) = 1;
	    else 
			Robp(j) = 0;
        end 
        
          
       %% 存储绘图参数 
        picturePara(j,1) = c(n); %脚标矩阵
        picturePara(j,2) = rfPreT(n); %rf预测位移矩阵
        picturePara(j,5) = bpPreT(n); %bp预测位移矩阵
        picturePara(j,3) = T_test(n); %原始位移矩阵
        picturePara(j,4) = i; %抽样次数矩阵 
       
        j = j+1;

	end

end

%% 计算模型破坏概率
% 计算随机森林变异系数
Prf_m = mean (Rorf) % 计算失效概率
Prf_stand = sqrt(Prf_m*(1 - Prf_m)./size(Rorf,2)); %无偏标准差
Yiburf = Prf_stand/ Prf_m %计算模型变异系数

% 计算bp神经网络变异系数
Pbp_m = mean (Robp) % 计算失效概率
Pbp_stand = sqrt(Pbp_m*(1 - Pbp_m)./size(Robp,2)); %无偏标准差
Yibubp = Pbp_stand/ Pbp_m %计算模型变异系数

%% -------------绘图----------------------%
% 构建矩阵存储脚标、预测位移、原始位移和抽样次数 

Index = picturePara(:,1); %脚标矩阵
rfPredictDis = picturePara(:,2); %rf预测位移
bpPredictDis = picturePara(:,5); %bp预测位移
OriginDis = picturePara(:,3); %原始位移
SampleSeries = picturePara(:,4); %抽样次数
DrawOriginDis = Trail(:,10); %原始序列

rfError = rfPredictDis - OriginDis;


%% ---构造三维绘图矩阵---

x1 = Index; y1 = SampleSeries; z1 = rfError;
% 抽稀
x = x1(1:3000); y = y1(1:3000); z =z1(1:3000);

[X, Y] = meshgrid(min(x):0.5:max(x), min(y):0.5:max(y));
Z = griddata(x,y,z,X,Y,'v4');
figure(1);
surf(X,Y,Z)
shading interp;
colormap(jet);

x3d = X(:); y3d = Y(:); z3d = Z(:); 

%局部放大图
SelectIndex = 114:125;
LogitPartIndex = ismember(Index,SelectIndex);
PartIndex = Index(LogitPartIndex);
PartRfPre = rfPredictDis(LogitPartIndex);
PartRfPre121 = rfPredictDis(Index ==121);

%% 局部指标计算
PartRfPre14 = rfPredictDis(1:14);
PartOrPre14 = OriginDis(1:14);

Rmse14 = sqrt(mse(PartRfPre14, PartOrPre14));
R2_14 = 1 - Rmse14/var(PartOrPre14); 

%全局指标
RmseBP = sqrt(mse(bpPredictDis,OriginDis));
RmseRF = sqrt(mse(rfPredictDis, OriginDis));
RFR2 = 1 - RmseRF/var(OriginDis);
BPR2 = 1 - RmseBP/var(OriginDis);

MaxPreRF = zeros(125,1);
MinPreRF = zeros(125,1);
MaxPreBP = zeros(125,1);
MinPreBP = zeros(125,1);
for num = 1:max(Index)
    MaxPreRF(num) = max(rfPredictDis(Index == num));
    MinPreRF(num) = min(rfPredictDis(Index == num));
    MaxPreBP(num) = max(bpPredictDis(Index == num));
    MinPreBP(num) = min(bpPredictDis(Index == num));
end
