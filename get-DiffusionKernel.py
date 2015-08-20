#!/usr/bin/env python

# builds diffusion kernel K = exp(beta*H) with H = A-D from the list of pair-wise interactions
# input : 
# - list of pair-wise interactions in format: chr1 start1 end1 chr2 start2 end2 value 
# - list of promoters
# - project directory

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.mstats import mquantiles
import math
import scipy.linalg
import itertools
import copy
import random
import gzip
import rcca

#InteractionsFile=sys.argv[1]
#PromoterFile=sys.argv[2]
#PROJDIR=sys.argv[3]

InteractionsFileCaptureC='/srv/gsfs0/projects/kundaje/users/mtaranov/projects/dynamic3D/ContactsAfterIC/ChiCAGO_Calls_Adam/analysis/CaptureC_SC_bait-bait.bed.gz'
InteractionsFileHiC='/srv/gsfs0/projects/kundaje/users/mtaranov/projects/dynamic3D/ContactsAfterIC/P-P_from_HiC/output/PP_SC_RE100.bed.gz'
PromoterFile='/srv/gsfs0/projects/kundaje/users/mtaranov/projects/dynamic3D/FIT-HI-C/promoters_flexible_repl_combined/data/reference_genomes/hg19/PromoterCapture_Digest_Human_HindIII_baits_ID.bed'
PROJDIR='/srv/gsfs0/projects/kundaje/users/mtaranov/projects/dynamic3D/ContactsAfterIC/kCCA'

# builds adjacency matrix 
def BuildMatrixA(PromoterFile, InteractionsFile, datatype):

    REFrag_dict={}
    index=0
    # Assign indices to all promoter HindIII sites.
    for line in open(PromoterFile,'r'):
        words=line.rstrip().split()
        key=(words[0], words[1], words[2])
        if words[0] in ['chr1']: # only chr1
           REFrag_dict[key]=index
           index+=1

    # Initialize matrix (promoter x promoter)
    PPMatrix=np.zeros((len(REFrag_dict), len(REFrag_dict))) #  number of promoters in chr 1

    # Fill (promoter x promoter) matrix with q-values of promoter-promoter interaction
    for line in gzip.open(InteractionsFile,'r'):
        words=line.rstrip().split()
        if words[0] in ['chr1']: #only chr1
            i=REFrag_dict[(words[0], words[1], words[2])]
            j=REFrag_dict[(words[3], words[4], words[5])]
            if datatype == 'HiC':
                q_values=1.0-float(words[6]) # for HiC
            else:
                q_values=1.0-math.pow(10.0,-1*float(words[6]))  # for CaptureC
            if PPMatrix[i,j] != 0:
                PPMatrix[i,j]=PPMatrix[i,j]/2+q_values/2
                PPMatrix[j,i]=PPMatrix[j,i]/2+q_values/2
            else: 
                PPMatrix[i,j]=q_values
                PPMatrix[j,i]=q_values
            
    # take -1*log(Q) for non-zero entries
    #mask = PPMatrix != 0
    #PPMatrix[mask] = np.log10(PPMatrix[mask])*(-1)
    
    # list of non-zero q-values
    q_values=list(filter((0.0).__ne__,list(itertools.chain.from_iterable(np.array(PPMatrix).tolist()))))
    
    # Some tests:
    print "Some tests on adjacency matrix:"    
    # 1. Check if the matrix is symmetric:
    if (PPMatrix.transpose() == PPMatrix).all() == True:
        print "Adjacency matrix is symmetric"
    # 2. Print out average q-values:
    print "Average q-value with zeros: ", str(np.average(PPMatrix))
    print "Average q-value w/o zeros: ", np.mean(q_values)
    
    # Print distribution of q-values
    fig = plt.figure()
    plt.hist(q_values)
    fig.savefig(str(PROJDIR)+'/Q-value_Histogram.png', dpi=fig.dpi)

    return PPMatrix

def DiffusionKernel(AdjMatrix, beta):
    # 1.Computes Degree matrix  - diagonal matrix with diagonal entries = raw sums of adjacency matrix 
    DegreeMatrix = np.diag(np.sum(AdjMatrix, axis=1))
    # 2. Computes negative Laplacian H = AdjMatrix - DegreeMatrix
    H = np.subtract(AdjMatrix, DegreeMatrix)
    # 3. Computes matrix exponential: exp(beta*H)
    K = scipy.linalg.expm(beta*H)

   # tests:
   # plot cummulative (1-q)-value (raw sums) for all promoters
    fig = plt.figure()
    plt.plot(np.sum(PPMatrix, axis=1))
    plt.xlim(0, len(PPMatrix))
    fig.savefig(str(PROJDIR)+'/DegreeMatrix.png', dpi=fig.dpi)

    return K

def printMatrix(Matrix, outputName, ylabel, QuantileValue, LowerUpperLimit):
    #vmaxLim=mquantiles(Matrix,[0.99])[0]
    Lim=mquantiles(Matrix,[QuantileValue])[0]
    print Matrix.max()
    print np.shape(Matrix)
    print "Limit:", Lim
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=0.8)
    if LowerUpperLimit == 'lower':
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmin=Lim)
    else:
        m = ax.matshow(Matrix, origin="bottom", #norm=colors.LogNorm(),  #norm=colors.SymLogNorm(1),
               cmap="RdYlBu_r", vmax=Lim) # cmap="RdYlBu_r"
    

    ax.axhline(-0.5, color="#000000", linewidth=1, linestyle="--")
    ax.axvline(-0.5, color="#000000", linewidth=1, linestyle="--")

    cb = fig.colorbar(m)
    cb.set_label(ylabel)

    ax.set_ylim((-0.5, len(Matrix) - 0.5))
    ax.set_xlim((-0.5, len(Matrix) - 0.5))

    fig.savefig(outputName, figsize=(5,5), dpi=150)
    return

#test proportion is computed as total-train-vali
#nodes is an array of indices
def train_vali_test(PPMatrix, trainProportion,valiProportion):
    total_num=len(PPMatrix)
    train_num=int(trainProportion*total_num)
    vali_num=int(valiProportion*total_num)
    test_num=total_num-train_num-vali_num
    if test_num<=0:
        print "Nothing in the test set!!!"
    print "Training set: "+str(train_num)
    print "Validation set: "+str(vali_num)
    print "Test set: "+str(test_num)
    #decide the random split of nodes
    nodes = [i for i in range(len(PPMatrix))]
    shuffled_nodes=copy.copy(nodes)
    random.shuffle(shuffled_nodes)
    train_nodes=np.array(shuffled_nodes[:train_num])
    vali_nodes=np.array(shuffled_nodes[train_num:(train_num+vali_num)])
    test_nodes=np.array(shuffled_nodes[(train_num+vali_num):])
    #split each dataset based on the decided split
    def split_data(nodes , PPMatrix):
        #subSampledMatrix=PPMatrix[nodes, :]
        subSampledMatrix=PPMatrix[nodes, :math.ceil(len(PPMatrix)/20)]
        return subSampledMatrix
    train = split_data(train_nodes, PPMatrix)
    vali = split_data(vali_nodes, PPMatrix)
    test = split_data(test_nodes, PPMatrix)
    print "Training:", np.shape(train), "Validation:", np.shape(vali), "Test:", np.shape(test)
    return train, vali, test
    

def run_cca(data1, data2, test1, test2, numCC,reg):
    cca = rcca.CCA(kernelcca=False, numCC=numCC, reg=reg)
    cca.train([data1, data2])
    # Find canonical components
    # Test on held-out data
    corrs = cca.validate([test1, test2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot0 = ax.bar(np.arange(corrs[0].shape[0]), corrs[0], 0.3, color = "steelblue")
    plot1 = ax.bar(np.arange(corrs[1].shape[0])+0.35, corrs[1], 0.3, color="orangered")
    ax.legend([plot0[0], plot1[0]], ["Dataset 1", "Dataset 2"])
    ax.set_ylabel("Prediction correlation")
    ax.set_xticks(np.arange(0, corrs[0].shape[0], 20)+0.325)
    ax.set_xticklabels(["%d" % i for i in range(0, corrs[0].shape[0], 20)])
    ax.set_xlabel("Test data m=113")
    fig.savefig(str(PROJDIR)+'/Prediction.png', dpi=fig.dpi)

#printMatrix(trainCaptureC, str(PROJDIR)+'/trainCaptureC_chr1', '1-q_value', 1, 'upper')
#printMatrix(valiCaptureC, str(PROJDIR)+'/valiCaptureC_chr1', '1-q_value', 1, 'upper')
#printMatrix(testCaptureC, str(PROJDIR)+'/testCaptureC_chr1', '1-q_value', 1, 'upper')

#printMatrix(PPMatrix, str(PROJDIR)+'/PPMatrixHiC_chr1', '1-q_value', 1, 'upper')
#DiffKMatrix10=DiffusionKernel(PPMatrix, 10)
#DiffKMatrix5=DiffusionKernel(PPMatrix, 5)
#DiffKMatrix1=DiffusionKernel(PPMatrix, 1)
#Difference15=np.subtract(DiffKMatrix5, DiffKMatrix1)
#Difference110=np.subtract(DiffKMatrix10, DiffKMatrix1)
#printMatrix(Difference15, str(PROJDIR)+'/DiffKMatrix15HiC_chr1', 'Exp(5*H)-Exp(1*H)', 0.001, 'lower')
#printMatrix(Difference110, str(PROJDIR)+'/DiffKMatrix110HiC_chr1', 'Exp(10*H)-Exp(1*H)', 0.001, 'lower')


#train = train_vali_test(PPMatrix, 0.9, 0.1)
#printMatrix(train, str(PROJDIR)+'/train_chr1', '1-q_value', 1, 'upper')
#np.savetxt(output, PPMatrix)
    return cca.train([data1, data2])

PPMatrix = BuildMatrixA(PromoterFile, InteractionsFileCaptureC, 'CaptureC')
DiffKMatrix1=DiffusionKernel(PPMatrix, 1)
trainCaptureC, valiCaptureC, testCaptureC = train_vali_test(DiffKMatrix1, 0.7, 0.2)
run_cca(trainCaptureC, trainCaptureC, testCaptureC, testCaptureC, 2, 1)

