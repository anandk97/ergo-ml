import numpy as np
import os
from PIL import Image
import re
import json 
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import csv
import sys
import pandas as pd

def img_curve_fit():
    img = Image.open("C:/Users/69516/OneDrive/research/JCATI/wrist torque computation/flexion moment curve.png")
    img_rgb = img.load()
    width, height = img.size

    p_curve = []
    for x in range(img.width):
        y_values = []
        for y in range(img.height):
            r, g, b = img_rgb[x, y]
            if r > 200 and g < 10 and b < 10:
                y_values.append(y)
        
        if y_values:
            avg_y = sum(y_values)/len(y_values)
            p_curve.append([x, avg_y])

    p_curve = np.array(p_curve)

    # translate pixel coordinate to graph coordinate
    X = (p_curve[:,0]-787)/398*50
    Y = -(p_curve[:,1]-1093)/339*5
    S = -8

    i = np.where((X[:-1] <= S) & (X[1:] >= S))[0][0]

    # 2. Fit the straight line
    def linear(x, m, b):
        return m * x + b

    params_line, _ = curve_fit(linear, X[:i+1], Y[:i+1])

    # 3. Fit the polynomial (without constraints) to get initial parameters
    def polynomial_no_c(x, a, b):
        return a * x**2 + b * x

    params_poly_initial, _ = curve_fit(polynomial_no_c, X[i+1:], Y[i+1:], p0=(1, 1))

    # Adjust the polynomial to meet the constraint
    mid_point = (X[i] + X[i+1]) / 2
    c_initial = S - params_poly_initial[0] * mid_point**2 - params_poly_initial[1] * mid_point

    # Refit with the constraint
    def polynomial(x, a, b, c):
        return a * x**2 + b * x + c

    params_poly, _ = curve_fit(polynomial, X[i+1:], Y[i+1:], p0=(params_poly_initial[0], params_poly_initial[1], c_initial))

    # # Plotting the data and the fitted curves
    # plt.scatter(X, Y, color='orange',alpha=0.1, label='Data Points')
    # plt.plot(X[:i+2], linear(X[:i+2], *params_line), 'b--', label='Fitted Line')
    # plt.plot(X[i-2:], polynomial(X[i-2:], *params_poly), 'r--', label='Fitted Polynomial')
    # plt.axvline(x=S, color='gray', linestyle='--')
    # plt.legend()
    # plt.xlabel('flexion angle (degree)')
    # plt.ylabel('flexion moment (Nm)')
    # plt.title('Curve Fitting')
    # plt.show()
    # plt.close()

    return params_line, params_poly, S

def moment_lookup(X, params_line, params_poly, S):
    
    def linear(x, m, b):
        return m * x + b

    def polynomial(x, a, b, c):
        return a * x**2 + b * x + c
    
    if np.isscalar(X):  # Check if X is a scalar
        if X <= S:
            return linear(X, *params_line)
        else:
            return polynomial(X, *params_poly)
    else:  # If X is an array
        Y = np.zeros_like(X)*1.0
        mask = X <= S
        Y[mask] = linear(X[mask], *params_line)
        Y[~mask] = polynomial(X[~mask], *params_poly)
        return Y

params_line, params_poly, S = img_curve_fit()

def correct_torque_outlier(torque,num_bins):
    
    counts, bins = np.histogram(torque,bins=num_bins)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    counts_percent = counts/np.sum(counts)

    sum = 0
    outlier_ratio = 0.01
    for idx in range(len(counts)):
        sum += counts[num_bins-1-idx]
        if sum >= np.sum(counts)*outlier_ratio:
            break
        else:
            outlier_thresh = (bins[num_bins-idx]+bins[num_bins-idx-1])/2
    
    torque[torque > outlier_thresh] = outlier_thresh
    return torque, outlier_thresh
            
dir = "C:/Data/Hand-intensive Manufacturing Processes Dataset"

# find all folders under dir
dict_torque_l = {"p1":np.empty((0,)),"p2":np.empty((0,)),"p3":np.empty((0,)),"p4":np.empty((0,)),"p5":np.empty((0,)),"p6":np.empty((0,)),"p7":np.empty((0,))}
dict_torque_r = {"p1":np.empty((0,)),"p2":np.empty((0,)),"p3":np.empty((0,)),"p4":np.empty((0,)),"p5":np.empty((0,)),"p6":np.empty((0,)),"p7":np.empty((0,))}
dict_torque_outlier_thresh_l = {"p1":0,"p2":0,"p3":0,"p4":0,"p5":0,"p6":0,"p7":0}
dict_torque_outlier_thresh_r = {"p1":0,"p2":0,"p3":0,"p4":0,"p5":0,"p6":0,"p7":0}
dict_torque_mean_l = {"p1":0,"p2":0,"p3":0,"p4":0,"p5":0,"p6":0,"p7":0}
dict_torque_mean_r = {"p1":0,"p2":0,"p3":0,"p4":0,"p5":0,"p6":0,"p7":0}

# get the torque for each person 
for folder in os.listdir(dir):

    parts = folder.split(" ")
    person = parts[0]
    tool_n_trial = " ".join(parts[1:])

    torque_l = np.load(dir + "/" + folder + "/processed torque_detail_l.npy")
    torque_r = np.load(dir + "/" + folder + "/processed torque_detail_r.npy")
    torque_sum_l = np.sum(torque_l, axis=1)
    torque_sum_r = np.sum(torque_r, axis=1)
    dict_torque_l[person] = np.append(dict_torque_l[person],torque_sum_l)
    dict_torque_r[person] = np.append(dict_torque_r[person],torque_sum_r)

# find torque mean and outlier_thresh for each person
for person in dict_torque_l.keys():
    torque_l = dict_torque_l[person]
    torque_r = dict_torque_r[person]
    num_bins = 20

    torque_l, outlier_thresh_l = correct_torque_outlier(torque_l,num_bins)
    torque_r, outlier_thresh_r = correct_torque_outlier(torque_r,num_bins)

    dict_torque_mean_l[person] = np.mean(torque_l)
    dict_torque_mean_r[person] = np.mean(torque_r)

    dict_torque_outlier_thresh_l[person] = outlier_thresh_l
    dict_torque_outlier_thresh_r[person] = outlier_thresh_r

# compute BACH score
for folder in os.listdir(dir):
    parts = folder.split(" ")
    person = parts[0]
    tool_n_trial = " ".join(parts[1:])

    gonio = pd.read_csv(dir + "/" + folder + "/processed gonio.csv", header=None).values
    gonio[gonio > 90] = 90
    gonio[gonio < -90] = -90
    
    torque_l = np.load(dir + "/" + folder + "/processed torque_detail_l.npy")
    torque_r = np.load(dir + "/" + folder + "/processed torque_detail_r.npy")
    outlier_thresh_l = dict_torque_outlier_thresh_l[person]
    outlier_thresh_r = dict_torque_outlier_thresh_r[person]

    torque_l = np.sum(torque_l, axis=1)
    torque_r = np.sum(torque_r, axis=1)
    torque_l[torque_l > outlier_thresh_l] = outlier_thresh_l
    torque_r[torque_r > outlier_thresh_r] = outlier_thresh_r

    torque_normalized_l = torque_l/dict_torque_mean_l[person]
    torque_normalized_r = torque_r/dict_torque_mean_r[person]

    wrist_factor_l = 11.85/moment_lookup(gonio[:,3], params_line, params_poly, S)
    wrist_factor_r = 11.85/moment_lookup(gonio[:,1], params_line, params_poly, S)

    score_BACH_l = torque_normalized_l*wrist_factor_l
    score_BACH_r = torque_normalized_r*wrist_factor_r
    