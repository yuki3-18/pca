"""
* PCA_python
*
* @Author Zhihui Lu
* @Sponsor Tozawa
* @Version alpha_0.1
* @Date 2018/05/02
*
"""
import ioFunction_version_4_3 as IO
import argparse
import numpy as np
import PCA_sklearn


def main():
    parser = argparse.ArgumentParser(description='py, Dirout, EUDT_txt, num_case')

    parser.add_argument('--Dirout', '-i1', default='F:/PCA_python_debug/test',
                        help='Dirout_path')
    parser.add_argument('--EUDT_text', '-i2', default='F:/PCA_python_debug/input.txt',
                        help='EUDT(training_data_list)(.txt)')
    parser.add_argument('--num_case', '-i3', default='500', help='num of training data(int)',
                        type=int)

    args = parser.parse_args()

    case_size = int(512 * 512 * 1)

    # load data
    print('load data')
    case = np.zeros((args.num_case, case_size))

    with open(args.EUDT_text, 'rt') as f:
        i = 0
        for line in f:
            if i >= args.num_case:
                break
            line = line.split()
            case[i, :] = IO.read_raw(line[0], dtype='double')
            i += 1

    print(case.shape)

    # Prepare for pca
    print('process pca')
    pca = PCA_sklearn.PCA(n_components=args.num_case - 1, svd_solver='arpack')

    # Do pca and map to Principal component
    pca.fit(case)

    # mean_vector
    mean_vector = pca.mean_

    # components
    U = pca.components_

    # eigen_vector
    eigen_vector = pca.explained_variance_

    # output_result
    ratio = pca.explained_variance_ratio_  # CCR
    ratio = np.cumsum(ratio)
    np.savetxt(args.Dirout + '/CCR.txt', ratio, delimiter=',', fmt='%.6f')

    IO.write_raw(mean_vector, args.Dirout + '/mean.vect')  # mean

    IO.write_raw(eigen_vector, args.Dirout + '/eval.vect')  # eigen_vector
    np.savetxt(args.Dirout + '/eval.txt', eigen_vector, delimiter=',', fmt='%.6f')

    for i in range(0, args.num_case - 1):
        IO.write_raw(U[i, :].copy(), args.Dirout + '/vect_' + str(i).zfill(4) + '.vect')  # PCs


if __name__ == '__main__':
    main()
