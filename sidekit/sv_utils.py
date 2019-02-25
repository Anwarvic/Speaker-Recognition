# -*- coding: utf-8 -*-
#
# This file is part of SIDEKIT.
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#
# SIDEKIT is a python package for speaker verification.
# Home page: http://www-lium.univ-lemans.fr/sidekit/
#    
# SIDEKIT is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as 
# published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.
#
# SIDEKIT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with SIDEKIT.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2014-2019 Anthony Larcher

:mod:`sv_utils` provides utilities to facilitate the work with SIDEKIT.
"""
import ctypes
import copy
import gzip
import multiprocessing
import numpy
import os
import pickle
import re
import scipy
import sys
if sys.version_info.major > 2:
    from functools import reduce

__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2014-2019 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = 'reStructuredText'


def save_svm(svm_file_name, w, b):
    """Save SVM weights and bias in PICKLE format
    
    :param svm_file_name: name of the file to write
    :param w: weight coefficients of the SVM to store
    :param b: biais of the SVM to store
    """
    if not os.path.exists(os.path.dirname(svm_file_name)):
            os.makedirs(os.path.dirname(svm_file_name))
    with gzip.open(svm_file_name, "wb") as f:
            pickle.dump((w, b), f)


def read_svm(svm_file_name):
    """Read SVM model in PICKLE format
    
    :param svm_file_name: name of the file to read from
    
    :return: a tupple of weight and biais
    """
    with gzip.open(svm_file_name, "rb") as f:
        (w, b) = pickle.load(f)
    return numpy.squeeze(w), b


def check_file_list(input_file_list, file_name_structure):
    """Check the existence of a list of files in a specific directory
    Return a new list with the existing segments and a list of indices 
    of those files in the original list. Return outputFileList and 
    idx such that inputFileList[idx] = outputFileList
    
    :param input_file_list: list of file names
    :param file_name_structure: structure of the filename to search for
    
    :return: a list of existing files and the indices 
        of the existing files in the input list
    """
    exist_files = numpy.array([os.path.isfile(file_name_structure.format(f)) for f in input_file_list])
    output_file_list = input_file_list[exist_files]
    idx = numpy.argwhere(numpy.in1d(input_file_list, output_file_list))
    return output_file_list, idx.transpose()[0]


def initialize_iv_extraction_weight(ubm, T):
    """
    Estimate matrices W and T for approximation of the i-vectors
    For more information, refers to [Glembeck09]_

    :param ubm: Mixture object, Universal Background Model
    :param T: Raw TotalVariability matrix as a ndarray
    
    :return:    
      W: fix matrix pre-computed using the weights from the UBM and the 
          total variability matrix
      Tnorm: total variability matrix pre-normalized using the co-variance 
          of the UBM
    """
    # Normalize the total variability matrix by using UBM co-variance
    
    sqrt_invcov = numpy.sqrt(ubm.get_invcov_super_vector()[:, numpy.newaxis])
    Tnorm = T * sqrt_invcov
    
    # Split the Total Variability matrix into sub-matrices per distribution
    Tnorm_c = numpy.array_split(Tnorm, ubm.distrib_nb())
    
    # Compute fixed matrix W
    W = numpy.zeros((T.shape[1], T.shape[1]))
    for c in range(ubm.distrib_nb()):
        W = W + ubm.w[c] * numpy.dot(Tnorm_c[c].transpose(), Tnorm_c[c])

    return W, Tnorm


def initialize_iv_extraction_eigen_decomposition(ubm, T):
    """Estimate matrices Q, D_bar_c and Tnorm, for approximation 
    of the i-vectors.
    For more information, refers to [Glembeck09]_
    
    :param ubm: Mixture object, Universal Background Model
    :param T: Raw TotalVariability matrix
    
    :return:
      Q: Q matrix as described in [Glembeck11]
      D_bar_c: matrices as described in [Glembeck11]
      Tnorm: total variability matrix pre-normalized using the co-variance of the UBM
    """
    # Normalize the total variability matrix by using UBM co-variance
    sqrt_invcov = numpy.sqrt(ubm.get_invcov_super_vector()[:, numpy.newaxis])
    Tnorm = T * sqrt_invcov
    
    # Split the Total Variability matrix into sub-matrices per distribution
    Tnorm_c = numpy.array_split(Tnorm, ubm.distrib_nb())
    
    # Compute fixed matrix Q
    W = numpy.zeros((T.shape[1], T.shape[1]))
    for c in range(ubm.distrib_nb()):
        W = W + ubm.w[c] * numpy.dot(Tnorm_c[c].transpose(), Tnorm_c[c])
    
    eigen_values, Q = scipy.linalg.eig(W)
    
    # Compute D_bar_c matrix which is the diagonal approximation of Tc' * Tc
    D_bar_c = numpy.zeros((ubm.distrib_nb(), T.shape[1]))
    for c in range(ubm.distrib_nb()):
        D_bar_c[c, :] = numpy.diag(reduce(numpy.dot, [Q.transpose(), Tnorm_c[c].transpose(), Tnorm_c[c], Q]))
    return Q, D_bar_c, Tnorm


def initialize_iv_extraction_fse(ubm, T):
    """Estimate matrices for approximation of the i-vectors.
    For more information, refers to [Cumani13]_
    
    :param ubm: Mixture object, Universal Background Model
    :param T: Raw TotalVariability matrix
    
    :return:
      Q: Q matrix as described in [Glembeck11]
      D_bar_c: matrices as described in [Glembeck11]
      Tnorm: total variability matrix pre-normalized using the co-variance of the UBM
    """
    # % Initialize the process
    # %init = 1;
    #
    #
    #   Extract i-vectors by using different methods
    #
    # %rank_T      = 10;
    # %featureSize = 50;
    # %distribNb   = 32;
    # %dictSize    = 5;
    # %dictSizePerDis=rank_T;  % a modifier par la suite
    #
    # %ubm_file    = 'gmm/world32.gmm';
    # %t_file      = 'mat/TV_32.matx';
    #
    #
    #
    #   Load data
    #
    #
    # % Load UBM for weight parameters that are used in the optimization
    # % function
    # %UBM = ALize_LoadModel(ubm_file);
    #
    # % Load meand from Minimum Divergence re-estimation
    # %sv_mindiv = ALize_LoadVect(minDiv_file)';
    #
    # % Load T matrix
    # %T = ALize_LoadMatrix(t_file)';
    #
    #
    # function [O,PI,Q] = factorized_subspace_estimation(UBM,T,dictSize,outIterNb,inIterNb)
    #
    #    rank_T      = size(T,2);
    #    distribNb   = size(UBM.W,2);
    #    featureSize = size(UBM.mu,1);
    #
    #    % Normalize matrix T
    #    sv_invcov = reshape(UBM.invcov,[],1);
    #    T_norm = T .* repmat(sqrt(sv_invcov),1,rank_T);
    #
    #     Tc{1,distribNb} = [];
    #     for ii=0:distribNb-1
    #         % Split the matrix in sub-matrices
    #         Tc{ii+1} = T_norm((ii*featureSize)+1:(ii+1)*featureSize,:);
    #     end
    #
    #     % Initialize O and Qc by Singular Value Decomposition
    #     init_FSE.Qc{distribNb} 	= [];
    #     init_FSE.O{distribNb}   = [];
    #     PI{distribNb}           = [];
    #
    #     for cc=1:distribNb
    #         init_FSE.Qc{cc} 	= zeros(featureSize,featureSize);
    #         init_FSE.O{cc}   = zeros(featureSize,featureSize);
    #         PI{cc}  = sparse(zeros(featureSize,dictSize*distribNb));      % a remplacer par une matrice sparse
    #     end
    #
    #     % For each distribution
    #     for cc=1:distribNb
    #         fprintf('Initilize matrice for distribution %d / %d\n',cc,distribNb);
    #         % Initialized O with Singular vectors from SVD
    #         [init_FSE.O{cc},~,V] = svd(UBM.W(1,cc)*Tc{cc});
    #         init_FSE.Qc{cc} = V'; 
    #     end
    #
    #
    #    % Concatenate Qc to create the matrix Q: A MODIFIER POUR DISSOCIER
    #    % dictSize DU NOMBRE DE DISTRIBUTIONS
    #    Q = [];
    #    for cc=1:distribNb
    #        Q = [Q;init_FSE.Qc{cc}(1:dictSize,:)];
    #    end
    #    O = init_FSE.O;
    #    clear 'init_FSE'
    #
    #
    #    % OUTER iteration process : update Q iteratively
    #    for it = 1:outIterNb
    #
    #        fprintf('Start iteration %d / %d for Q re-estimation\n',it,5);
    #
    #        % INNER iteration process: update PI and O iteratively
    #        for pioIT = 1:inIterNb
    #            fprintf('   Start iteration %d / %d for PI and O re-estimation\n',pioIT,10);
    #
    #            % Update PI
    #            %Compute diagonal terms of QQ'
    # %            diagQ = diag(Q*Q');
    #
    #            for cc=1:distribNb
    #
    #                % Compute optimal k and optimal v 
    #                % for each line f of PI{cc}
    #                A = O{cc}'*Tc{cc}*Q';
    #                f = 1;
    #                while (f < size(A,1)+1)
    #
    #                    if(f == 1)
    #                        A = O{cc}'*Tc{cc}*Q';       % equation (26)
    #                        PI{cc} = sparse(zeros(featureSize,dictSize*distribNb));
    #                    end
    #
    #                    % Find the optimal index k
    #                    [~,k] = max(A(f,:).^2);
    #                    k_opt = k;
    #
    #                    % Find the optimal value v
    #                    v_opt = A(f,k_opt);
    #
    #                    % Update the line of PI{cc} with the value v_opt in the
    #                    % k_opt-th column
    #                    PI{cc}(f,k_opt)     = v_opt;
    #
    #                    % if the column already has a non-zero element,
    #                    % update O and PI
    #                    I = find(PI{cc}(:,k_opt)~=0);
    #                    if size(I,1)>1
    #                        % get indices of the two lines of PI{cc} which
    #                        % have a non-zero element on the same column
    #                        a = I(1);
    #                        b = I(2);
    #
    #                        % Replace column O{cc}(:,a) and O{cc}(:,b)
    #                        Oa = (PI{cc}(a,k_opt)*O{cc}(:,a)+PI{cc}(b,k_opt)*O{cc}(:,b))/
    # (sqrt(PI{cc}(a,k_opt)^2+PI{cc}(b,k_opt)^2));
    #                        Ob = (PI{cc}(a,k_opt)*O{cc}(:,b)-PI{cc}(b,k_opt)*O{cc}(:,a))/
    # (sqrt(PI{cc}(a,k_opt)^2+PI{cc}(b,k_opt)^2));
    #                        O{cc}(:,a) = Oa;
    #                        O{cc}(:,b) = Ob;
    #
    #                        PI{cc}(a,k_opt) = sqrt(PI{cc}(a,k_opt)^2+PI{cc}(b,k_opt)^2);
    #                        PI{cc}(b,k_opt) = 0;
    #
    #                        f = 0;
    #                    end
    #                    f =f +1;
    #                end
    #            end
    #
    #            obj = computeObjFunc(UBM.W,Tc,O,PI,Q);
    #            fprintf('Objective Function after estimation of PI = %2.10f\n',obj);                
    #
    #            % Update O
    #            for cc=1:distribNb
    #
    #                % Compute 
    #                Z = PI{cc}*Q*Tc{cc}';
    #
    #                % Compute Singular value decomposition of Z
    #                [Uz,~,Vz] = svd(Z);
    #
    #                % Compute the new O{cc}
    #                O{cc} = Vz*Uz';
    #            end
    #            obj = computeObjFunc(UBM.W,Tc,O,PI,Q);
    #            fprintf('Objective Function after estimation of O = %2.10f\n',obj);
    #        end % END OF INNER ITERATION PROCESS
    #
    #
    #        % Update Q
    #        D = sparse(zeros(size(PI{cc},2),size(PI{cc},2)));
    #        E = zeros(size(PI{cc},2),rank_T);
    #        for cc=1:distribNb
    #            % Accumulate D
    #            D = D + UBM.W(1,cc) * PI{cc}'*PI{cc};
    #
    #            % Accumulate the second term
    #            E = E + UBM.W(1,cc) * PI{cc}'*O{cc}'*Tc{cc};
    #        end
    #        Q = D\E;
    #
    #        % Normalize rows of Q and update PI accordingly
    #
    #        % Compute norm of each row
    #        c1 = bsxfun(@times,Q,Q);
    #        c2 = sum(c1,2);
    #        c3 = sqrt(c2);
    #        Q = bsxfun(@rdivide,Q,c3);
    #
    #        % Update PI accordingly
    #        PI = cellfun(@(x)x*sparse(diag(c3)), PI, 'uni',false);
    #
    #        obj = computeObjFunc(UBM.W,Tc,O,PI,Q);
    #        fprintf('Objective Function after re-estimation of Q = %2.10f\n',obj);
    #    end
    # end
    pass


def clean_stat_server(statserver):
    """

    :param statserver:
    :return:
    """
    zero_idx = ~(statserver.stat0.sum(axis=1) == 0.)
    statserver.modelset = statserver.modelset[zero_idx]
    statserver.segset = statserver.segset[zero_idx]
    statserver.start = statserver.start[zero_idx]
    statserver.stop = statserver.stop[zero_idx]
    statserver.stat0 = statserver.stat0[zero_idx, :]
    statserver.stat1 = statserver.stat1[zero_idx, :]
    assert statserver.validate(), "Error after cleaning StatServer"
    print("Removed {} empty sessions in StatServer".format((~zero_idx).sum()))


def parse_mask(mask):
    """

    :param mask:
    :return:
    """
    if not set(re.sub("\s", "", mask)[1:-1]).issubset(set("0123456789-,")):
        raise Exception("Wrong mask format")
    tmp = [k.split('-') for k in re.sub(r"[\s]", '', mask)[1:-1].split(',')]
    indices = []
    for seg in tmp:
        if len(seg) == 1:
            seg += seg
        if len(seg) == 2:
            indices += list(range(int(seg[0]), int(seg[1])+1))
        else:
            raise Exception("Wrong mask format")
    return indices


def segment_mean_std_hdf5(input_segment, in_context=False):
    """
    Compute the sum and square sum of all features for a list of segments.
    Input files are in HDF5 format

    :param input_segment: list of segments to read from, each element of the list is a tuple of 5 values,
        the filename, the index of thefirst frame, index of the last frame, the number of frames for the
        left context and the number of frames for the right context
    :param in_context:
    :return: a tuple of three values, the number of frames, the sum of frames and the sum of squares
    """
    features_server, show, start, stop, in_context = input_segment

    if start is None or stop is None or not in_context:
        feat, _ = features_server.load(show,
                                       start=start,
                                       stop=stop)

    else:
        # Load the segment of frames plus left and right context
        feat, _ = features_server.load(show,
                                       start=start-features_server.context[0],
                                       stop=stop+features_server.context[1])
        # Get features in context
        feat, _ = features_server.get_context(feat=feat,
                                              label=None,
                                              start=features_server.context[0],
                                              stop=feat.shape[0]-features_server.context[1])

    return feat.shape[0], feat.sum(axis=0), numpy.sum(feat**2, axis=0)


def mean_std_many(features_server, seg_list, in_context=False, num_thread=1):
    """
    Compute the mean and standard deviation from a list of segments.

    :param features_server:
    :param seg_list: list of file names with start and stop indices
    :param in_context:
    :param num_thread:
    :return: a tuple of three values, the number of frames, the mean and the variance
    """
    if isinstance(seg_list[0], tuple):
        inputs = [(copy.deepcopy(features_server), seg[0], seg[1], seg[2], in_context) for seg in seg_list]
    elif isinstance(seg_list[0], str):
        inputs = [(copy.deepcopy(features_server), seg, None, None, in_context) for seg in seg_list]

    pool = multiprocessing.Pool(processes=num_thread)
    res = pool.map(segment_mean_std_hdf5, inputs)
    pool.terminate()
    total_N = 0
    total_F = 0
    total_S = 0
    for N, F, S in res:
        total_N += N
        total_F += F
        total_S += S
    return total_N, total_F / total_N, total_S / total_N


def serialize(M):
    M_shape = M.shape
    ct = ctypes.c_double
    if M.dtype == numpy.float32:
        ct = ctypes.c_float
    tmp_M = multiprocessing.Array(ct, M.size)
    M = numpy.ctypeslib.as_array(tmp_M.get_obj())
    return M.reshape(M_shape)
