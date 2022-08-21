clear all;
clc;
close all;


load ('xp.mat');
load ('yp.mat');
load ('zp.mat');
load ('pitch.mat');


t = 1:0.01:600;
N = length(t);
number_of_state = 6;
number_of_measure = 4;
SAMPLE = 201;          %sample time
SAMPLE_TIME = 201;
%w = 0.55;             %wave
w = 0.02; %0.015
L = 1.39;              %mast length
G =  zeros(6,4);       %gain matrix

%Error covariance matrix
a = 0.0001;           %
b = 0.1;
Q =  zeros(6,6);      %Process model covariance matrix
R =  zeros(4,4);      %Measure model covariance matrix

Q(1,1) = a;        %initialization
Q(2,2) = a;
Q(3,3) = a;
Q(4,4) = a;
Q(5,5) = a;
Q(6,6) = a;

R(1,1) =  b; 
R(2,2) =  b;  
R(3,3) =  b;  
R(4,4) =  5;  

w1 = sqrt(a)*randn(6,N);
w2 = sqrt(b)*randn(4,N);

hx = zeros(4,N);      %state
z =  zeros(4,N);      %measure
x =  zeros(6,N);

p = 0;
r = 0;
pdot = 1;
rdot = 1;

fx = zeros(6,N);
fx(1,1) = p;
fx(2,1) = r;
fx(3,1) = pdot;
fx(4,1) = rdot;
fx(5,1) = w;
fx(6,1) = L;

%Jacobian matrix of process model
F =  zeros(6,6);
%Jacobian matrix of measure model
H =  zeros(4,6);

hx = zeros(4,N);
houyan_final = zeros(4,N);
model_output = zeros(4,N);
I = eye(6);
Pk_ =zeros(6,6); 
Pk_1=eye(6,6);
Xhat_k = zeros(6,N);
xkhat_ = zeros(6,N);
h_xkhat_ = zeros(4,N);
%xkhat_ = zeros(4,N);

for i = 2:N          %1352组数据后为有效数据
    %model
%     fx(1,i) = Xhat_k(1,i-1) + Xhat_k(3,i-1)/SAMPLE_TIME + w1(1,i-1);
%     fx(2,i) = Xhat_k(2,i-1) + Xhat_k(4,i-1)/SAMPLE_TIME + w1(2,i-1);
%     fx(3,i) = Xhat_k(3,i-1) - Xhat_k(5,i-1)^2 * Xhat_k(1,i-1)/SAMPLE_TIME+ w1(3,i-1);
%     fx(4,i) = Xhat_k(4,i-1) - Xhat_k(5,i-1)^2 * Xhat_k(2,i-1)/SAMPLE_TIME+ w1(4,i-1);
%     fx(5,i) = w+ w1(5,i-1);
%     fx(6,i) = L+ w1(6,i-1);
%     model_output(1,i) = L * sin(fx(1,i))+ w2(1,i-1);  % model value
%     model_output(2,i) = L * cos(fx(2,i))+ w2(2,i-1);
%     model_output(3,i) = L * cos(fx(1,i)) * cos(fx(2,i))+ w2(3,i-1);
%     model_output(4,i) = fx(1,i)+ w2(4,i-1);
    hx(1,i) = Xp(1352+i)-1.312;                               % measure value
    hx(2,i) = Yp(1352+i)-0.018;
    hx(3,i) = Zp(1352+i);
    hx(4,i) = pitch(1352+i)+0.1;
    
    
    %Get a priori estimate
    xkhat_(1,i) = Xhat_k(1,i-1) + Xhat_k(3,i-1)/SAMPLE_TIME ;
    xkhat_(2,i) = Xhat_k(2,i-1) + Xhat_k(4,i-1)/SAMPLE_TIME ;
    xkhat_(3,i) = Xhat_k(3,i-1) - Xhat_k(5,i-1)^2 * Xhat_k(1,i-1)/SAMPLE_TIME;
    xkhat_(4,i) = Xhat_k(4,i-1) - Xhat_k(5,i-1)^2 * Xhat_k(2,i-1)/SAMPLE_TIME;
    xkhat_(5,i) = w;
    xkhat_(6,i) = L;
    %Nonlinear measurement based on prior estimation
    h_xkhat_(1,i) = L * sin(xkhat_(1,i));
    h_xkhat_(2,i) = L * cos(xkhat_(2,i));
    h_xkhat_(3,i) = L * cos(xkhat_(1,i)) * cos(xkhat_(2,i));
    h_xkhat_(4,i) = xkhat_(1,i);
    %Update measurement Jacobian matrix
    H(1,1) = L*cos(Xhat_k(1,i));    
    H(2,2) = L*cos(Xhat_k(2,i));
    H(1,6) = sin(Xhat_k(1,i));
    H(2,6) = sin(Xhat_k(2,i));
    H(3,1) = -L*cos(Xhat_k(2,i))*sin(Xhat_k(1,i));
    H(3,2) = -L*cos(Xhat_k(1,i))*sin(Xhat_k(2,i));
    H(3,6) = cos(Xhat_k(2,i))*cos(Xhat_k(1,i));
    H(4,1) = 1.0; 
    %Update process model Jacobian matrix
    for j = 1:6
     F(j,j) = 1;
    end
    F(1,3) = 1/SAMPLE;%deltaT
    F(2,4) = 1/SAMPLE;
    F(3,1) = -1*w^2/SAMPLE;
    F(3,5) = -2*w*Xhat_k(1,i-1)/SAMPLE;%
    F(4,5) = -2*w*Xhat_k(2,i-1)/SAMPLE;%
    F(4,2) = -1*w^2/SAMPLE;
    %
    Pk_ = F*Pk_1*F.' +  Q;
    %gain of kalman
    K_k = Pk_ * H.' * inv(H*Pk_*H.' + R);
    
    Xhat_k(:,i) = xkhat_(:,i) +  K_k*(hx(:,i) - h_xkhat_(:,i));
    
    houyan_final(1,i) = L * sin(Xhat_k(1,i));
    houyan_final(2,i) = L * cos(Xhat_k(2,i));
    houyan_final(3,i) = L * cos(Xhat_k(1,i)) * cos(Xhat_k(2,i));
    houyan_final(4,i) = Xhat_k(1,i);
    
    Pk_1 = (I - K_k*H)*Pk_;
end
figure(1);
plot(hx(1,:));hold on;
plot(houyan_final(1,:));hold on;plot(h_xkhat_(1,:));hold on;
%plot(Xhat_k(2,:));hold on;plot(z(2,:));hold on;
legend('celiang','houyan*H','xianyanzhi*H');
figure(2);
plot(houyan_final(1,:));hold on;plot(houyan_final(2,:));hold on;
plot(houyan_final(4,:));hold on
%plot(Xhat_k(2,:));hold on;plot(z(2,:));hold on;
legend('x ','y ','pitch');


