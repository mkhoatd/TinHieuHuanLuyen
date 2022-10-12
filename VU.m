clear; close all;
[signal_sM2, Fs_sM2] = audioread("studio_M2.wav");
[signal_sF2, Fs_sF2] = audioread("studio_F2.wav");
[signal_pM2, Fs_pM2] = audioread("phone_M2.wav");
[signal_pF2, Fs_pF2] = audioread("phone_F2.wav");

trainingData_sM2 = [0.45 0.48 0.77 0.79 0.88 0.92 1.32 1.37 1.53 1.59 1.93]; %u
trainingData_sF2 = [0.77 1.25 1.27 1.35 1.41 1.76 1.83 1.98 2.06 2.37]; %v
trainingData_pM2 = [0.53 1.05 1.12 1.24 1.31 1.46 1.68 1.97 2.06 2.12 2.17 2.30 2.43 2.50 2.52]; %v
trainingData_pF2 = [1.02 1.88 1.95 2.16 2.25 2.60 2.75 3.34 3.38 3.45 3.62 3.80 3.91 4.00 4.04]; %v
 
[frame_sM2, frameSize_sM2, frameCount_sM2] = chiaFrame(signal_sM2, Fs_sM2);
[frame_sF2, frameSize_sF2, frameCount_sF2] = chiaFrame(signal_sF2, Fs_sF2);
[frame_pM2, frameSize_pM2, frameCount_pM2] = chiaFrame(signal_pM2, Fs_pM2);
[frame_pF2, frameSize_pF2, frameCount_pF2] = chiaFrame(signal_pF2, Fs_pF2);

STE_sM2 = STE(frame_sM2, frameCount_sM2); 
STE_sF2 = STE(frame_sF2, frameCount_sF2);
STE_pM2 = STE(frame_pM2, frameCount_pM2);
STE_pF2 = STE(frame_pF2, frameCount_pF2);
[STE_sM2_V, STE_sM2_U] = chiaUV(STE_sM2, trainingData_sM2, 'U');
[STE_sF2_V, STE_sF2_U] = chiaUV(STE_sF2, trainingData_sF2, 'V');
[STE_pM2_V, STE_pM2_U] = chiaUV(STE_pM2, trainingData_pM2, 'V');
[STE_pF2_V, STE_pF2_U] = chiaUV(STE_pF2, trainingData_pF2, 'V');

STE_V = [STE_sM2_V STE_sF2_V STE_pM2_V STE_pF2_V];
STE_U = [STE_sM2_U STE_sF2_U STE_pM2_U STE_pF2_U];
T_STE = binsearch(STE_V, STE_U);

ZCR_sM2 = ZCR(frame_sM2, frameCount_sM2);
ZCR_sF2 = ZCR(frame_sF2, frameCount_sF2);
ZCR_pM2 = ZCR(frame_pM2, frameCount_pM2);
ZCR_pF2 = ZCR(frame_pF2, frameCount_pF2);
[ZCR_sM2_V, ZCR_sM2_U] = chiaUV(ZCR_sM2, trainingData_sM2, 'U');
[ZCR_sF2_V, ZCR_sF2_U] = chiaUV(ZCR_sF2, trainingData_sF2, 'V');
[ZCR_pM2_V, ZCR_pM2_U] = chiaUV(ZCR_pM2, trainingData_pM2, 'V');
[ZCR_pF2_V, ZCR_pF2_U] = chiaUV(ZCR_pF2, trainingData_pF2, 'V');

ZCR_V = [ZCR_sM2_V ZCR_sF2_V ZCR_pM2_V ZCR_pF2_V];
ZCR_U = [ZCR_sM2_U ZCR_sF2_U ZCR_pM2_U ZCR_pF2_U];
T_ZCR = binsearch(ZCR_U, ZCR_V);

Voiced_Unvoiced("studio_M2.wav", T_STE, T_ZCR, trainingData_sM2, "sm2", 'U');
Voiced_Unvoiced("studio_F2.wav", T_STE, T_ZCR, trainingData_sF2, "sf2", 'V');
Voiced_Unvoiced("phone_M2.wav", T_STE, T_ZCR, trainingData_pM2, "pm2", 'V');
Voiced_Unvoiced("phone_F2.wav", T_STE, T_ZCR, trainingData_pF2, "pf2", 'V');

function Voiced_Unvoiced(file, T_STE, T_ZCR, trainingData, name, firstUV)
    [signal, Fs] = audioread(file);
    t = (1:length(signal))/Fs;
    [frame, frameSize, frameCount] = chiaFrame(signal, Fs);
    ste = STE(frame, frameCount);
    zcr = ZCR(frame, frameCount);

    ste = normaliseByT(ste, T_STE);
    zcr = normaliseByT(zcr, T_ZCR);

    figure('Name', name);
    subplot(3,1,1);
    plot(t, signal);
    drawFrame(ste, frameSize, Fs, 'r');
    drawFrame(zcr, frameSize, Fs, 'g');

    P = ste - zcr;
    
    subplot(3,1,2);
    plot(t, signal);
    drawBorder(trainingData, 'red', firstUV);
    
    subplot(3,1,3);
    plot(t, signal);
    decision = zeros(1, frameCount);
    for i = 1 : length(decision)
        if P(i) > 0.015
            decision(i) = 1;
        end
    end

    border = [];
    for i = 1 : length(decision) - 1
    if decision(i) ~= decision(i+1) 
        frameIndex = i;
        time = frameIndex * 0.02;
            line([time time], [-1 1], 'Color', 'green');
            border = [border decision(i+1) time];
    end
    end

    for i = 1 : 2 : length(border)
        if(i <= length(border) - 2)
            if border(i) == 0
                line([border(i+1) border(i+3)], [1 1], 'Color', 'c', 'LineWidth', 1);
            else
                line([border(i+1) border(i+3)], [1 1], 'Color', 'r', 'LineWidth', 1);
            end
        end
    end
end

%chuan hoa diem
function result = normaliseMAX(frame)
    result = frame ./ max(abs(frame));
end

%chuan hoa dai
function result = normaliseByT(frame, T)
    min1 = min(frame);
    max1 = max(frame);
    for i = 1 : length(frame)
        if(frame(i) > T) result(i) = (frame(i) - T) / (max1 - T);
        else result(i) = (frame(i) - T) / (T - min1);
        end
    end
end

%chia frame
function [frame, frameSize, frameCount] = chiaFrame(signal, Fs)
frameSize = round(0.02 * Fs); 
frameCount = floor(length(signal) / frameSize);

temp = 0;
for i = 1 : frameCount
    frame(i,:) = signal(temp + 1 : temp + frameSize);
    temp = temp + frameSize;
end
end

%tinh va chuan hoa STE
function result = STE(frame, frameCount)
for i = 1 : frameCount
    result(i) = sum(frame(i,:).^2);
end

%chuẩn hóa STE về dải [0;1] (STE luôn dương)
result = normaliseMAX(result);
%result = (result - min(result)) ./ ((max(result) - min(result)));
end

%tinh va chuan hoa ZCR
function result = ZCR(frame, frameCount)
    for i = 1 : frameCount
        result(i) = sum(abs(sgn(frame(i,2:end)) - sgn(frame(i,1:end-1))));
    end
    
    result = normaliseMAX(result);
    %result = (result - min(result)) ./ ((max(result) - min(result)));
end

%signum
function result = sgn(frame)
    for i = 1 : length(frame)
        if(frame(i) >= 0) result(i) = 1;
        else result(i) = -1;
        end
    end
end

function drawFrame(frame, frameSize, Fs, color)
    wave = 0;
for i = 1 : length(frame)
    l = length(wave);
    wave(l : l + frameSize) = frame(i);
end

%trục x (t)
t1 = [0 : 1/Fs : length(wave)/Fs];
t1 = t1(1:end-1);

hold on;
plot(t1, wave, color);
hold off;
end

function drawBorder(border, color, firstUV)
    flag = 1;
    if firstUV == 'U'
        flag = -1;
    end
    for i = 1 : length(border)
        line([border(i) border(i)], [-1 1], 'Color','r');
        if(i <= length(border) - 1)
            if flag == -1
                line([border(i) border(i+1)], [1 1], 'Color', 'c', 'LineWidth', 1);
            else
                line([border(i) border(i+1)], [1 1], 'Color', 'r', 'LineWidth', 1);
            end
            flag = -flag;
        end
    end
end

function [voiced, unvoiced] = chiaUV(frame, trainingData, firstIsUorV)
    voiced = [];
    unvoiced = [];
    t = 0.02;
    if(firstIsUorV == 'U')
        for i = 1 : length(trainingData) - 1
            if mod(i,2) == 1
                unvoicedStartIndex = floor(trainingData(i)/t)+1;
                unvoicedEndIndex = floor(trainingData(i+1)/t);
                unvoiced = [unvoiced frame(unvoicedStartIndex:unvoicedEndIndex)];
            else
                voicedStartIndex = floor(trainingData(i)/t) + 1;
                voicedEndIndex = floor(trainingData(i+1)/t);
                voiced = [voiced frame(voicedStartIndex:voicedEndIndex)];
            end                
        end
    else
        for i = 1 : length(trainingData) - 1
            if mod(i,2) == 1
                voicedStartIndex = floor(trainingData(i)/t)+1;
                voicedEndIndex = floor(trainingData(i+1)/t);
                voiced = [voiced frame(voicedStartIndex:voicedEndIndex)];
            else
                unvoicedStartIndex = floor(trainingData(i)/t) + 1;
                unvoicedEndIndex = floor(trainingData(i+1)/t);
                unvoiced = [unvoiced frame(unvoicedStartIndex:unvoicedEndIndex)];
            end
        end
    end
end

function T = binsearch(g, f)
Nf = length(f);
Ng = length(g);
%Tìm threshold T theo binary search 
Tmax = max([g f]);
Tmin = min([g f]);
T = (Tmax + Tmin)/2;

p = length(find(g > T));
i = length(find(f < T));

j = -1; q = -1;
while i ~= j || p ~= q
    if 1/Nf * sum(f(f > T) - T) - 1/Ng * sum(T - g(g < T)) > 0
        Tmin = T;
    else 
        Tmax = T;
    end
    T = (Tmax + Tmin)/2;
    j = i; 
    q = p;
    p = length(find(g > T));
    i = length(find(f < T));
end
end
