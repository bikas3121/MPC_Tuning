% example_yalmip
yalmip('clear')
clear 
clc 
close all

%% Plant - Reconstruction filter
format long
Fs = 1e6;
Ts = 1/Fs;
Fc = 1e5;           % cutoff frequency
Wn = Fc/(Fs/2);
% Wn = 2*pi*Fc;
n = 3;              % filter order

[b1,a1] = butter(n, Wn,"low");
H1  = tf(b1,a1);
H = minreal(H1, 1e-5);
b  = [H.num{1}];
a  = [H.den{1}];
[Ah, Bh, Ch, Dh] = tf2ss(b1,a1);


%% Optimization probelm setup
% Define variables
Pf = sdpvar(n);
Pg = sdpvar(n);
Wf = sdpvar(1,n);
Wg = sdpvar(n,1);
L = sdpvar(n);
mu_e  = sdpvar(1);


% Constraints
mu_eta = 2^2;
MA = [Ah*Pf + Bh*Wf , Ah; L ,Pg*Ah];
MB = [Bh ; Wg];
MC = [Ch*Pf + Dh*Wf ,  Ch];
MP = [Pf, eye(n); eye(n), Pg];
MC_tilde = [Wf, zeros(1,n)];

C1 = [MP MA MB; MA' MP zeros(2*n,1); MB' zeros(1,2*n) eye(1)];
C2 = [mu_e MC Dh'; MC' MP zeros(2*n,1); Dh zeros(1,2*n) eye(1)];
C3 = [mu_eta MC_tilde; MC_tilde' MP];

BC1 = eye(2*n + 2*n + 1 );
BC2 = eye(1 + 2*n + 1 );
BC3 = eye(1 + 2*n);

% Constraints
% F = [Pf >= eye(n), Pg>= eye(n), C1 >= BC1, C2 >= BC2, C3>= BC3];
F = [Pf >= 0, Pg>= 0, C1 >= 0, C2 >= 0, C3>= 0];

ops = sdpsettings('solver','mosek');
ops.verbose = 1;
ops.showprogress = 1;
ops.debug = 1;
optimize(F, mu_e, ops)

%% Extract values
Pf = value(Pf);
Pg = value(Pg);
Wf = value(Wf);
Wg = value(Wg);
L = value(L);

%%  Transformation matrices
inv_Pg = pinv(Pg);
Sf = Pf - inv_Pg;
P1 = [Pf, Sf; Sf, Sf]; 
P = pinv(P1);
U = [Pf, eye(n); Sf, zeros(n) ];


%% Create dynamic file name to save 
[N,D] = rat(sqrt(mu_eta));
NSF_num = sprintf("NSF_num_%dkHz_%dMHz_%d%s%dMUeta",Fc/1e3,Fs/1e6,N,'|',D);
NSF_den = sprintf("NSF_den_%dkHz_%dMHz_%d%s%dMueta",Fc/1e3,Fs/1e6,N,'|',D);
%% Optimal NTF
format long
tol = 1e-22;
[Ar, Br, Cr, Dr] = ntf(Ah, Bh,  Pf, Pg, Wf, Wg, L); % state space
[br, ar] = ss2tf(Ar, Br, Cr, Dr);           % transfer function
Hr1 = tf(br,ar, Ts);
Hr = minreal(Hr1);
save(NSF_num, 'br')
save(NSF_den, 'ar')

Hf1= tf(ar, br, Ts);
Hf = minreal(Hf1);

%% Noise shaping filter
NSF = tf(ar-br,ar);
[Af, Bf, Cf, Df] = tf2ss(ar-br,ar);

% corresponding lowpass filter
LPF = tf(ar/br);
% LPF = minreal(LPF);
[A_olpf, B_olpf, C_olpf, D_olpf] = tf2ss(ar, br);
%% Closed loop system dynamics
Acl = [Ah, Bh*Cr; zeros(n) , Ar];
Bcl = [Bh; Br];
Ccl = [Ch, Dh*Cr];
Dcl = Dh;

%% 
MA1 = U'*P*Acl*U;
MB1 = U'*P*Bcl;
MC1 = Ccl*U;
MP1 = U'*P*U;

%% %% Plots
N = Fs/2;

% tf of H(z)
[hlp,wlp] = freqz(b1,a1, N, 'half', Fs);

%NSF derived directly from R_{opt}(z)
[hf,wf] = freqz(b1-a1,b1, N, 'half', Fs);

%NTF derived directly from R_{opt}(z)
[hn,wn] = freqz(a1,b1, N, 'half', Fs);




sl = length(hlp)/2;
figure(Name="Butterworth 2nd order")
plot(wlp(1:sl)/2*pi*1e-3, 20*log10(abs(hlp(1:sl))));
hold on 
plot(wf(1:sl)/2*pi*1e-3, 20*log10(abs(hf(1:sl))));
hold on 
plot(wn(1:sl)/2*pi*1e-3, 20*log10(abs(hn(1:sl))));

legend("LPF ($H(z)$)","NSF ($F(z)$)","NTF ($R(z)$)",'Interpreter','latex')
xlabel('Frequency (kHz)')
ylabel("Magnitude (dB)")
ylim([-50, 50])
grid minor


[hho,who] = freqz(ar,br, N, 'half', Fs);
[hfo,wfo] = freqz(ar - br,ar, N, 'half', Fs);
[hntf,wntf] = freqz(br,ar, N, 'half', Fs);
figure
plot(who(1:sl)/2*pi*1e-3, 20*log10(abs(hho(1:sl))));
hold on 
plot(wfo(1:sl)/2*pi*1e-3, 20*log10(abs(hfo(1:sl))));
hold on 
plot(wntf(1:sl)/2*pi*1e-3, 20*log10(abs(hntf(1:sl))));
legend("LPF ($H_{opt}(z))$","NSF ($F_{opt}(z)$)", "NTF ($R_{opt}(z)$)", 'Interpreter','latex')
xlabel('Frequency (kHz)')
ylabel("Magnitude (dB)")
ylim([-50,50])
grid minor



%%
% 
% figure
% plot(wlp1(1:sl)/2*pi*1e-3, 20*log10(abs(hlp1(1:sl))));
% hold on 
% plot(wf(1:sl)/2*pi*1e-3, 20*log10(abs(hf(1:sl))));
% 
% 
% 
% legend("$H_{MPC}(z)$", "$F_{opt}(z)$", 'Interpreter','latex')
% grid minor
% xlabel('Frequency (kHz)')
% ylabel("Magnitude (dB)")


