clc; clear all; close all;

%% =======================================================================
%  Feature Extraction Script — v6
%  - 24 test conditions × repetitions (from the new layout)
%  - 2-class labels: Stable(1), Chatter(0) — Transition class removed
%  - Fixed 350 windows per signal (Short signals extracted as available)
%  - Raw signal (no noise removal)
%  - v4: added wRCMDE (Yang, Guo & Sun, 2022) and MPE (Liu et al., 2021)
%        features 38-42. All original v3 code/logic is unchanged below;
%        new code is clearly marked and appended only.
%  - v5: added CE (Liu et al., 2021, Eq. 9) and three additional MPE
%        scale factors (s = 1, 2, 3), features 43-46. All v3/v4
%        code/logic is unchanged below; new code is clearly marked and
%        appended only.
%  - v6: SUPERSEDES the multi-scale part of v4/v5. Per paper, wRCMDE is
%        the only one of the two that is genuinely a multi-scale feature
%        set in its source paper (Yang et al. use s = 1..4 jointly as 4
%        inputs to their SVM); MPE in Liu et al. is a single best-scale
%        index (s = 4) chosen from a sweep, not a multi-scale vector. To
%        match the 40-feature table (22 time + 14 freq + CE + WPEE + one
%        MPE + one wRCMDE = 40), both are now reduced to ONE feature
%        each: MPE_s1/s2/s3 (added in v5) are removed, keeping only the
%        paper-justified MPE at s = 4; wRCMDE_s1/s2/s3/s4 (added in v4)
%        are collapsed to a single chosen scale (see parameter block
%        below for which one and why this is a judgment call, not a
%        literal reproduction of Yang et al.). CE is unchanged. Total
%        feature count: 40.
%  - Output: one .mat file per signal (same name, containing SigData struct)
%% =======================================================================

% -----------------------------------------------------------------------
% Signal list: {filename, label_value, label_string}
%   1 = Stable (S)
%   0 = Chatter (U)
% -----------------------------------------------------------------------
signal_table = {
    % --- Test 1: Type A, L36, DOC0.5, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 2: Type A, L36, DOC1, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 3: Type A, L36, DOC2, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 4: Type A, L36, DOC4, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC4.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 5: Type A, L36, DOC6, WOC1, N2700, F270 → Stable ---
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC6.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 6: Type A, L36, DOC0.5, WOC2, N2700, F270 → Stable ---
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 7: Type A, L36, DOC1, WOC2, N2700, F270 → Stable ---
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 8: Type A, L36, DOC1.5, WOC2, N2700, F270 → Stable ---
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 9: Type A, L36, DOC2, WOC2, N2700, F270 → Chatter ---
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 10: Type A, L36, DOC4, WOC2, N2700, F270 → Chatter ---
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC4.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 11: Type A, L36, DOC1, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 12: Type A, L36, DOC1.5, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 13: Type A, L36, DOC2, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC2.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 14: Type A, L36, DOC3, WOC1, N1800, F180 → Stable ---
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC3.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 15: Type A, L36, DOC0.5, WOC1, N3600, F360 → Stable ---
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC0.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 16: Type A, L36, DOC1, WOC1, N3600, F360 → Stable ---
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.0_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 17: Type A, L36, DOC1.5, WOC1, N3600, F360 → Stable ---
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPA_L36_DOC1.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 18: Type A, L36, DOC2, WOC1, N3600, F360 → Chatter ---
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC2.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';

    % --- Test 19: Type A, L36, DOC3, WOC1, N3600, F360 → Chatter ---
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPA_L36_DOC3.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';
% ------------------------------------------------------------------------------------------------
    % --- Test 20: Type B, L56, DOC0.5, WOC1, N2700, F270 → Stable ---
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 21: Type B, L56, DOC1, WOC1, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 22: Type B, L56, DOC1.5, WOC1, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 23: Type B, L56, DOC2, WOC1, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 24: Type B, L56, DOC4, WOC1, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC1.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 25: Type B, L56, DOC6, WOC1, N2700, F270 → Chatter (as per table: R1,R3,R4,R5) ---
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC6.0_WOC1.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 26: Type B, L56, DOC0.5, WOC2, N2700, F270 → Stable ---
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 27: Type B, L56, DOC1, WOC2, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 28: Type B, L56, DOC1.5, WOC2, N2700, F270 → Stable ---
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC2.0_N2700_F270_R4.wav',  1, 'Stable';

    % --- Test 29: Type B, L56, DOC2, WOC2, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 30: Type B, L56, DOC4, WOC2, N2700, F270 → Chatter ---
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC4.0_WOC2.0_N2700_F270_R4.wav',  0, 'Chatter';

    % --- Test 31: Type B, L56, DOC0.5, WOC1, N1800, F180 → Stable ---
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 32: Type B, L56, DOC1, WOC1, N1800, F180 → Stable ---
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 33: Type B, L56, DOC1.5, WOC1, N1800, F180 → Stable ---
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N1800_F180_R4.wav',  1, 'Stable';

    % --- Test 34: Type B, L56, DOC2, WOC1, N1800, F180 → Chatter ---
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N1800_F180_R4.wav',  0, 'Chatter';

    % --- Test 35: Type B, L56, DOC3, WOC1, N1800, F180 → Chatter ---
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N1800_F180_R4.wav',  0, 'Chatter';

    % --- Test 36: Type B, L56, DOC0.5, WOC1, N3600, F360 → Stable ---
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC0.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 37: Type B, L56, DOC1, WOC1, N3600, F360 → Stable ---
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.0_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 38: Type B, L56, DOC1.5, WOC1, N3600, F360 → Stable ---
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R1.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R2.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R3.wav',  1, 'Stable';
    'S_WPB_L56_DOC1.5_WOC1.0_N3600_F360_R4.wav',  1, 'Stable';

    % --- Test 39: Type B, L56, DOC2, WOC1, N3600, F360 → Chatter (corrected L36 → L56) ---
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC2.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';

    % --- Test 40: Type B, L56, DOC3, WOC1, N3600, F360 → Chatter (corrected L36 → L56) ---
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R1.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R2.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R3.wav',  0, 'Chatter';
    'U_WPB_L56_DOC3.0_WOC1.0_N3600_F360_R4.wav',  0, 'Chatter';
};

% باقي الكود (استخراج الميزات، المعالجة، الحفظ) يبقى كما هو دون تغيير

%% =======================================================================
%  Window Settings
%  - DeltaT = 20 ms  (≈ one spindle revolution at 2700 RPM,
%    dominant speed across experiments — kept fixed)
%  - OverlapRatio = 0.5  (50% overlap, standard practice)
%  - TARGET_WINDOWS = 350  (fits comfortably in ~4 s signals)
%% =======================================================================
DeltaT           = 20e-3;
OverlapRatio     = 0.5;
TARGET_WINDOWS   = 350;
%% =======================================================================
%  [v6 — supersedes v4/v5 multi-scale parameter block] Entropy/Envelope
%  Feature Parameters
%  Three feature families are added below, computed per the cited
%  papers' equations and reported parameter choices. These are ADDED
%  features only — none of the 37 features above were modified.
%
%  (A) wRCMDE — weighted Refined Composite Multiscale Dispersion Entropy
%      Yang, B.; Guo, K.; Sun, J. "Chatter Detection in Robotic Milling
%      Using Entropy Features." Appl. Sci. 2022, 12, 8276.
%      https://doi.org/10.3390/app12168276
%      - DisEn defined by Eqs. (5)-(8); RCMDE by Eqs. (11)-(12);
%        wRCMDE (kurtosis-weighted RCMDE) by Eqs. (13)-(14).
%      - Parameters per paper Section 4.1: m = 5, c = 5, tau = 1.
%      - IN THE SOURCE PAPER, wRCMDE is genuinely a 4-DIMENSIONAL
%        feature (s = 1, 2, 3, 4 used together as separate SVM inputs,
%        per their Figure 5/6) — it is NOT reduced to one scale by Yang
%        et al. To match this project's 40-feature target (one row per
%        nonlinear descriptor), a SINGLE scale, s_wRCMDE (below), is
%        used instead. This is OUR choice, not a literal reproduction
%        of the source paper's feature set, and should be described as
%        such in the manuscript. s_wRCMDE = 1 is used as a reasonable
%        default (the finest, least coarse-grained scale, where
%        differences between classes tend to be largest before
%        coarse-graining smooths them out), but this has not been
%        empirically verified on your data. Before finalizing, consider
%        running the same kind of scale-factor comparison Liu et al.
%        perform for MPE (Section 5.2 of their paper) on your own
%        labeled windows, and report whichever scale gives the cleanest
%        class separation, citing that as your own analysis rather than
%        attributing the specific scale choice to Yang et al.
%
%  (B) MPE — Multi-scale Permutation Entropy
%      Liu, X.; Wang, Z.; Li, M.; Yue, C.; Liang, S.Y.; Wang, L.
%      "Feature extraction of milling chatter based on optimized
%      variational mode decomposition and multi-scale permutation
%      entropy." Int J Adv Manuf Technol 2021.
%      https://doi.org/10.1007/s00170-021-07027-0
%      - Coarse-graining per their Eq. (5) (same scheme as Costa et al.,
%        2002); base entropy is the classical permutation entropy of
%        Bandt & Pompe (2002).
%      - Parameters per paper Section 5.2: m = 6, tau = 1.
%      - UNLIKE wRCMDE above, Liu et al. sweep the scale factor only to
%        select the single best-performing value (s = 4), then use that
%        one value alone as their chatter index — MPE is genuinely a
%        single-scale feature in the source paper, so s_MPE = 4 below
%        is a direct, literal match to their method (no extra judgment
%        call needed here, unlike wRCMDE's scale choice above).
%
%  (C) CE — Crest factor of the envelope spectrum
%      Liu, X.; Wang, Z.; Li, M.; Yue, C.; Liang, S.Y.; Wang, L. (2021),
%      Int J Adv Manuf Technol, Eqs. (6)-(9). In the source paper, CE is
%      used as the PSO fitness function for selecting VMD parameters
%      [K, alpha]; here it is extracted directly from the raw window as
%      a stand-alone feature value, exactly as defined by Eq. (9):
%      CE = max(E)/RMS(E), where E is the FFT of the Hilbert-transform
%      envelope of the signal (Eqs. 6-8). No extra parameters (m, c, tau)
%      are required for CE.
%      IMPLEMENTATION NOTE: the envelope signal is non-negative by
%      construction, so its spectrum has a very large 0 Hz (DC) term
%      that is not related to periodic impulse content. Consistent with
%      standard envelope-spectrum/demodulation practice (the resonance-
%      demodulation literature underlying Eq. 9, e.g., Zhang et al.
%      2015, cited as Ref. [24] in Liu et al.), the DC component is
%      removed before the spectrum is taken, so CE reflects periodic
%      impulsiveness rather than the envelope's mean level. This is an
%      implementation choice not made explicit in the paper's text and
%      should be reported as such if asked by a reviewer.
%
%  NOTE ON WINDOW LENGTH: all three source-paper computations above use
%  signal segments of ~1 to ~10 s (thousands of samples). Your
%  per-window length here is DeltaT = 20 ms, i.e. only a few hundred
%  samples per window (fewer still after coarse-graining). With m = 5/
%  c = 5 (3125 possible dispersion patterns) or m = 6 (720 possible
%  permutation patterns), a 20 ms window may not contain enough points
%  to populate the pattern space reliably, which can bias the entropy
%  estimate toward lower/noisier values. Consider reporting this as a
%  limitation, or re-checking results with a longer DeltaT (or computing
%  these features once per full signal rather than per window) before
%  relying on them for your final feature set.
%% =======================================================================
m_DE       = 5;     % embedding dimension for dispersion entropy (DisEn/RCMDE)
c_DE       = 5;     % number of classes for dispersion entropy
tau_DE     = 1;     % time delay for dispersion entropy
s_wRCMDE   = 1;     % single chosen scale for wRCMDE — OUR choice (see note
                     % above); not the literal Yang et al. 4-feature set,
                     % and not yet empirically verified on your data

m_PE   = 6;     % embedding dimension for permutation entropy (MPE)
tau_PE = 1;     % time delay for permutation entropy
s_MPE  = 4;     % scale factor for MPE — literal match to Liu et al.'s
                % chosen best scale
%% =======================================================================
%  Main Processing Loop
%% =======================================================================
num_files = size(signal_table, 1);

for file_idx = 1:num_files

    filename    = signal_table{file_idx, 1};
    label_val   = signal_table{file_idx, 2};   % 0=Chatter, 1=Stable
    label_str   = signal_table{file_idx, 3};

    % -------------------------------------------------------------------
    % Check file existence
    % -------------------------------------------------------------------
    if ~exist(filename, 'file')
        warning('[SKIP] File not found: %s', filename);
        continue;
    end

    fprintf('Processing [%d/%d] (%s): %s\n', ...
        file_idx, num_files, label_str, filename);

    % -------------------------------------------------------------------
    % Read signal
    % -------------------------------------------------------------------
    [signal, fs] = audioread(filename);
    signal = signal(:, 1);          % Use channel 1 if stereo

    WindowSamples = round(DeltaT * fs);   % samples per window (e.g. 320 @ 16kHz)
    StepSamples   = round((1 - OverlapRatio) * WindowSamples);  % 50% overlap step

    % -------------------------------------------------------------------
    % Determine available windows from signal length
    % -------------------------------------------------------------------
    available_windows = floor((length(signal) - WindowSamples) / StepSamples) + 1;

    if available_windows <= 0
        warning('[SKIP] Signal too short to extract even one window: %s', filename);
        continue;
    end

    % -------------------------------------------------------------------
    % Decide number of windows to extract
    %
    %   Case A — signal long enough (normal case ~4 s):
    %            extract exactly TARGET_WINDOWS windows,
    %            starting from index 1.
    %
    %   Case B — signal shorter than needed for TARGET_WINDOWS:
    %            extract as many windows as available
    %            (Option 2 per specification).
    % -------------------------------------------------------------------
    if available_windows >= TARGET_WINDOWS
        num_windows = TARGET_WINDOWS;
    end

    % -------------------------------------------------------------------
    % Pre-allocate feature struct array
    % -------------------------------------------------------------------
    SigData = repmat(struct( ...
        'FileName',           [], ...
        'Label',              [], ...
        'Mean',               [], ...
        'Median',             [], ...
        'STD',                [], ...
        'Var',                [], ...
        'CoV',                [], ...
        'RMS',                [], ...
        'Peak',               [], ...
        'PTP',                [], ...
        'Skewness',           [], ...
        'Kurtosis',           [], ...
        'CrestFact',          [], ...
        'Avg_amp',            [], ...
        'Square_root_amp',    [], ...
        'Clear_fact',         [], ...
        'Shape_Fact',         [], ...
        'Imp_Fact',           [], ...
        'Skew_fact',          [], ...
        'Kurt_fact',          [], ...
        'ZeroCrossingRate',   [], ...
        'OSAF',               [], ...
        'TDE',                [], ...
        'EnR',                [], ...
        'Mean_of_freq',       [], ...
        'Centre_Freq',        [], ...
        'RMS_freq',           [], ...
        'Spectral_centroid',  [], ...
        'Spectral_bandwidth', [], ...
        'STDF',               [], ...
        'Mean_Square_Freq',   [], ...
        'Freq_Var',           [], ...
        'Median_Freq',        [], ...
        'Spectral_Rolloff',   [], ...
        'Spectral_Energy',    [], ...
        'Spectral_Flatness',  [], ...
        'Spectral_Entropy',   [], ...
        'Peak_Freq_Ratio',    [], ...
        'WPEE',               [],  ...
        'CE',                 [], ...
        'MPE',                [], ...
        'wRCMDE',             []  ...
        ), num_windows, 1);
  % 'LabelStr',           [], ...
    % -------------------------------------------------------------------
    % Energy of full original signal (used for EnR)
    % -------------------------------------------------------------------
    OriginalSigEnergy = sum(signal .^ 2);

    % -------------------------------------------------------------------
    % Window loop — exactly num_windows iterations
    % -------------------------------------------------------------------
    [~, name_only, ~] = fileparts(filename);

    for co = 1:num_windows

        % Start sample of this window
        i_start   = (co - 1) * StepSamples + 1;
        i_end     = i_start + WindowSamples - 1;

        % Safety guard (should not trigger for Case A)
        if i_end > length(signal)
            warning('[WINDOW] Window %d exceeds signal length in %s. Stopping early.', ...
                co, filename);
            SigData(co:end) = [];
            break;
        end

        % *** RAW signal window — no noise removal ***
        SigSaving = signal(i_start : i_end);

        %% ---- Frequency Domain Setup (raw spectrum) ----
        N  = length(SigSaving);
        Y  = fft(SigSaving);
        P2 = abs(Y / N);
        P1 = P2(1 : floor(N/2) + 1);
        P1(2:end-1) = 2 * P1(2:end-1);  % Single-sided amplitude correction
        s  = P1 .^ 2;                    % Power spectrum
        frequencies = fs * (0 : (length(P1)-1))' / N;

        %% ---- Metadata ----
        SigData(co).FileName  = name_only;
        SigData(co).Label     = label_val;    % 0 / 1
        % SigData(co).LabelStr  = label_str;    % 'Chatter' / 'Stable'

        %% ========== Time-Domain Features (1–22) ==========

        % [1]  Mean
        SigData(co).Mean             = mean(SigSaving);
        % [2]  Median
        SigData(co).Median           = median(SigSaving);
        % [3]  Standard Deviation
        SigData(co).STD              = std(SigSaving);
        % [4]  Variance
        SigData(co).Var              = var(SigSaving);
        % [5]  Coefficient of Variation
        SigData(co).CoV              = SigData(co).STD / (SigData(co).Mean + eps);
        % [6]  RMS
        SigData(co).RMS              = rms(SigSaving);
        % [7]  Peak
        SigData(co).Peak             = max(abs(SigSaving));
        % [8]  Peak-to-Peak
        SigData(co).PTP              = max(SigSaving) - min(SigSaving);
        % [9]  Skewness
        SigData(co).Skewness         = skewness(SigSaving);
        % [10] Kurtosis
        SigData(co).Kurtosis         = kurtosis(SigSaving);
        % [11] Crest Factor
        SigData(co).CrestFact        = SigData(co).Peak / (SigData(co).RMS + eps);
        % [12] Average Amplitude
        SigData(co).Avg_amp          = mean(abs(SigSaving));
        % [13] Square Root Amplitude
        SigData(co).Square_root_amp  = (mean(sqrt(abs(SigSaving)))) ^ 2;
        % [14] Clearance Factor
        SigData(co).Clear_fact       = SigData(co).Peak / (SigData(co).Square_root_amp + eps);
        % [15] Shape Factor
        SigData(co).Shape_Fact       = SigData(co).RMS / (SigData(co).Avg_amp + eps);
        % [16] Impulse Factor
        SigData(co).Imp_Fact         = SigData(co).Peak / (SigData(co).Avg_amp + eps);
        % [17] Skewness Factor
        SigData(co).Skew_fact        = SigData(co).Skewness / (SigData(co).RMS ^ 3 + eps);
        % [18] Kurtosis Factor
        SigData(co).Kurt_fact        = SigData(co).Kurtosis / (SigData(co).RMS ^ 4 + eps);
        % [19] Zero Crossing Rate
        SigData(co).ZeroCrossingRate = sum(abs(diff(sign(SigSaving)))) / (2 * N);

        % [20] One-Step Auto-correlation Function (OSAF)
        A = sum(SigSaving);
        B = sum(SigSaving .^ 2);
        C = sum(SigSaving(2:end) .* SigSaving(1:end-1));
        denom_osaf = N * B - A ^ 2;
        if denom_osaf ~= 0
            SigData(co).OSAF = (N * C - A ^ 2) / denom_osaf;
        else
            SigData(co).OSAF = NaN;
        end

        % [21] Time Domain Energy
        SigData(co).TDE = sum(SigSaving .^ 2);
        % [22] Energy Ratio
        SigData(co).EnR = SigData(co).TDE / (OriginalSigEnergy + eps);

        %% ========== Frequency-Domain Features (23–37) ==========

        % [23] Mean of Frequency
        SigData(co).Mean_of_freq       = sum(s) / length(s);
        % [24] Centre Frequency
        SigData(co).Centre_Freq        = sum(frequencies .* s) / (sum(s) + eps);
        % [25] RMS Frequency
        SigData(co).RMS_freq           = sqrt(sum((frequencies - SigData(co).Centre_Freq) .^ 2 .* s) / (sum(s) + eps));
        % [26] Spectral Centroid
        SigData(co).Spectral_centroid  = sum(frequencies .* P1) / (sum(P1) + eps);
        % [27] Spectral Bandwidth
        SigData(co).Spectral_bandwidth = sqrt(sum(((frequencies - SigData(co).Spectral_centroid) .^ 2) .* P1) / (sum(P1) + eps));
        % [28] Standard Deviation of Frequency (STDF)
        SigData(co).STDF               = sqrt(sum((frequencies - SigData(co).Mean_of_freq) .^ 2 .* s) / (sum(s) + eps));
        % [29] Mean Square Frequency
        SigData(co).Mean_Square_Freq   = sum((frequencies - SigData(co).Centre_Freq) .^ 2 .* s) / (sum(s) + eps);
        % [30] Frequency Variance
        SigData(co).Freq_Var           = sum((frequencies - SigData(co).Mean_of_freq) .^ 2 .* s) / (sum(s) + eps);

        % [31] Median Frequency
        cum_P1   = cumsum(P1);
        half_eng = cum_P1(end) / 2;
        [~, mid_idx] = min(abs(cum_P1 - half_eng));
        SigData(co).Median_Freq = frequencies(mid_idx);

        % [32] Spectral Roll-off (85%)
        roll_idx = find(cum_P1 >= 0.85 * cum_P1(end), 1, 'first');
        SigData(co).Spectral_Rolloff = frequencies(roll_idx);

        % [33] Spectral Energy
        SigData(co).Spectral_Energy = sum(s);

        % [34] Spectral Flatness
        SigData(co).Spectral_Flatness = exp(mean(log(P1 + eps))) / (mean(P1) + eps);

        % [35] Spectral Entropy
        p = P1 / (sum(P1) + eps);
        SigData(co).Spectral_Entropy = -sum(p .* log2(p + eps));

        % [36] Peak Frequency Ratio
        [~, max_idx] = max(P1);
        SigData(co).Peak_Freq_Ratio = P1(max_idx) / (sum(P1) + eps);

        % [37] Wavelet Packet Energy Entropy (WPEE)
        wavelet_name = 'db4';
        level        = 5;
        try
            tree     = wpdec(SigSaving, level, wavelet_name);
            nodes    = leaves(tree);
            energy_wp = zeros(length(nodes), 1);
            for k = 1:length(nodes)
                coeff        = wpcoef(tree, nodes(k));
                energy_wp(k) = sum(coeff .^ 2);
            end
            total_energy_wp = sum(energy_wp);
            prob_wp         = energy_wp / (total_energy_wp + eps);
            SigData(co).WPEE = -sum(prob_wp .* log2(prob_wp + eps));
        catch
            SigData(co).WPEE = NaN;
        end

          %% ========== [v6 — supersedes v4/v5 block] Features 38-40 ==========
        %  Single-scale wRCMDE, single-scale MPE, and CE. See the
        %  parameter block earlier in this script for why wRCMDE is a
        %  single chosen scale here (our choice, flagged) while MPE's
        %  single scale (s = 4) is a direct match to Liu et al.

        % [38] wRCMDE at scale factor s = s_wRCMDE (our chosen single
        %      scale — see parameter block note; NOT the literal 4-scale
        %      feature set Yang et al. use in their own SVM).
        %   Yang, Guo & Sun (2022), Appl. Sci. 12, 8276, Eqs. (5)-(14).
        try
            SigData(co).wRCMDE = localWRCMDE(SigSaving, m_DE, c_DE, tau_DE, s_wRCMDE);
        catch
            SigData(co).wRCMDE = NaN;
        end

        % [39] MPE at scale factor s = s_MPE (= 4, Liu et al.'s own
        %      chosen single best scale — a literal match, not a
        %      simplification).
        %   Liu, Wang, Li, Yue, Liang & Wang (2021), Int J Adv Manuf
        %   Technol, coarse-graining Eq. (5) + Bandt-Pompe (2002)
        %   permutation entropy.
        try
            SigData(co).MPE = localMPE(SigSaving, m_PE, tau_PE, s_MPE);
        catch
            SigData(co).MPE = NaN;
        end

        % [40] CE — Crest factor of the envelope spectrum
        %   Liu, Wang, Li, Yue, Liang & Wang (2021), Eqs. (6)-(9).
        try
            SigData(co).CE = localEnvelopeSpectrumCE(SigSaving);
        catch
            SigData(co).CE = NaN;
        end
    end % window loop

    % -------------------------------------------------------------------
    % Save — one .mat file per signal, same base name
    % -------------------------------------------------------------------
    save_name = [name_only '.mat'];
    save(save_name, 'SigData');
    fprintf('  Saved: %s (%d windows, label=%d [%s])\n', ...
        save_name, length(SigData), label_val, label_str);

end % file loop

fprintf('\nAll files processed.\n');


%% =======================================================================
%  [NEW in v4] Local Functions — Entropy Feature Implementations
%  Defined after script code (valid in MATLAB R2016b+ script files).
%  Each function implements the cited paper's equations exactly.
%% =======================================================================

function out = localDispersionEntropy(u, m, c, tau)
% Dispersion Entropy (DisEn).
% Yang, Guo & Sun (2022), Appl. Sci. 12, 8276, Eqs. (5)-(8).
    u = double(u(:));
    L = length(u);
    mu = mean(u);
    sg = std(u);
    if sg == 0
        out = 0;
        return;
    end

    y = normcdf(u, mu, sg);          % Eq. (5): normal-CDF mapping to [0,1]
    z = round(c .* y + 0.5);
    z(z < 1) = 1;
    z(z > c) = c;

    Nvec = L - (m - 1) * tau;
    if Nvec < 1
        out = NaN;
        return;
    end

    patternIdx = zeros(Nvec, 1);
    for i = 1:Nvec
        emb = z(i : tau : i + (m - 1) * tau);   % Eq. (6): embedding vector
        idx = 0;
        for d = 1:m
            idx = idx * c + (emb(d) - 1);        % unique base-c pattern id
        end
        patternIdx(i) = idx;                      % range: 0 .. c^m - 1
    end

    edges  = -0.5 : 1 : (c^m - 0.5);
    counts = histcounts(patternIdx, edges);
    p      = counts(counts > 0) / Nvec;           % Eq. (7)
    out    = -sum(p .* log(p));                   % Eq. (8)
end

function y = localCoarseGrainOffset(u, s, k)
% Refined-composite coarse-graining with starting offset k.
% Yang, Guo & Sun (2022), Eq. (11).
    u = double(u(:));
    L = length(u);
    starts = k : s : (L - s + 1);
    nWin = numel(starts);
    y = zeros(nWin, 1);
    for j = 1:nWin
        y(j) = mean(u(starts(j) : starts(j) + s - 1));
    end
end

function out = localRCMDE(u, m, c, tau, s)
% Refined Composite Multiscale Dispersion Entropy (RCMDE).
% Yang, Guo & Sun (2022), Eq. (12).
    deVals = zeros(s, 1);
    for k = 1:s
        yk = localCoarseGrainOffset(u, s, k);
        deVals(k) = localDispersionEntropy(yk, m, c, tau);
    end
    out = mean(deVals);
end

function out = localWRCMDE(u, m, c, tau, s)
% Weighted Refined Composite Multiscale Dispersion Entropy (wRCMDE).
% Yang, Guo & Sun (2022), Eqs. (13)-(14).
    KT  = kurtosis(double(u(:)));     % Eq. (13): Pearson (non-excess) kurtosis
    out = KT * localRCMDE(u, m, c, tau, s);
end

function y = localCoarseGrainSimple(x, s)
% Non-overlapping-average coarse-graining used for MPE.
% Liu et al. (2021), Eq. (5) (same scheme as Costa et al., 2002).
    x = double(x(:));
    n = length(x);
    nWin = floor(n / s);
    y = zeros(nWin, 1);
    for j = 1:nWin
        y(j) = mean(x((j - 1) * s + 1 : j * s));
    end
end

function idx = localPermRank(v)
% Lehmer-code-based unique index (1..m!) identifying an ordinal pattern.
    m = length(v);
    [~, ord] = sort(v);
    idx = 0;
    avail = ord;
    for i = 1:m
        rank = sum(avail(i + 1:end) < avail(i));
        idx = idx + rank * factorial(m - i);
    end
    idx = idx + 1;
end

function out = localPermEntropy(x, m, tau)
% Permutation Entropy (Bandt & Pompe, 2002), as used by Liu et al. (2021).
    x = double(x(:));
    L = length(x);
    N = L - (m - 1) * tau;
    if N < 1
        out = NaN;
        return;
    end
    pIdx = zeros(N, 1);
    for i = 1:N
        seg = x(i : tau : i + (m - 1) * tau);
        pIdx(i) = localPermRank(seg);
    end
    edges  = 0.5 : 1 : (factorial(m) + 0.5);
    counts = histcounts(pIdx, edges);
    p      = counts(counts > 0) / N;
    out    = -sum(p .* log(p));
end

function out = localMPE(x, m, tau, s)
% Multi-scale Permutation Entropy (MPE).
% Liu, Wang, Li, Yue, Liang & Wang (2021), Int J Adv Manuf Technol.
    xc  = localCoarseGrainSimple(x, s);
    out = localPermEntropy(xc, m, tau);
end

function out = localEnvelopeSpectrumCE(x)
% [NEW in v5] Crest factor of the envelope spectrum (CE).
% Liu, Wang, Li, Yue, Liang & Wang (2021), Int J Adv Manuf Technol,
% Eqs. (6)-(9):
%   xh(t)  = Hilbert transform of x(t)                       Eq. (6)
%   y(t)   = x(t) + j*xh(t)         (analytic signal)         Eq. (7)
%   A(t)   = sqrt(x(t)^2 + xh(t)^2) (envelope signal)         Eq. (8)
%   E      = FFT(A)                 (envelope spectrum)
%   CE     = max(E) / sqrt(mean(E.^2))                        Eq. (9)
%
% NOTE: the envelope A(t) is non-negative by construction, so its
% spectrum has a very large 0 Hz (DC) term unrelated to periodic
% impulse content; if left in, CE would be approximately constant
% (~sqrt(N)) for any signal and would not discriminate machining
% states. Consistent with standard envelope-spectrum/demodulation
% practice (the resonance-demodulation literature underlying Eq. 9,
% e.g., Zhang et al. 2015, cited as Ref. [24] in Liu et al.), the DC
% component is removed from the envelope before its spectrum is taken.
% This implementation choice is not made explicit in the paper's text
% and should be disclosed if asked.
    x = double(x(:));
    N = length(x);

    % --- Analytic signal via FFT-based Hilbert transform (Eqs. 6-7) ---
    Xf = fft(x);
    H  = zeros(N, 1);
    if mod(N, 2) == 0
        H(1)         = 1;
        H(2 : N/2)   = 2;
        H(N/2 + 1)   = 1;
    else
        H(1)               = 1;
        H(2 : (N+1)/2)     = 2;
    end
    xa = ifft(Xf .* H);

    % --- Envelope signal (Eq. 8), DC removed (see note above) ---
    A = abs(xa);
    A = A - mean(A);

    % --- Envelope spectrum (single-sided), DC bin dropped ---
    Ef    = fft(A);
    Ehalf = abs(Ef(1 : floor(N/2) + 1));
    if numel(Ehalf) > 1
        Ehalf = Ehalf(2:end);
    end

    % --- Crest factor of the envelope spectrum (Eq. 9) ---
    out = max(Ehalf) / (sqrt(mean(Ehalf .^ 2)) + eps);
end