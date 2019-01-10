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
import dataIO as io
import os, csv
import argparse
import numpy as np
import PCA_sklearn
import sklearn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimpleITK as sitk
import utils
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='py, Dirout, EUDT_txt, num_case')

    parser.add_argument('--Dirout', '-i1', default="E:/git/pca/output/CT/patch/z24",
                        help='Dirout_path')
    parser.add_argument('--EUDT_text', '-i2', default="E:/git/TFRecord_example/output/tfrecord/new/patch/th_150/list.txt",
                        help='EUDT(training_data_list)(.txt)')
    parser.add_argument('--num_case', '-i3', default='3039', help='num of training data(int)',
                        type=int)
    parser.add_argument('--components', '-i4', default='24', help='num of components(int)',
                        type=int)

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.Dirout)):
        os.makedirs(args.Dirout)

    patch_side = 9

    case_size = int(patch_side * patch_side * patch_side)
    num_test = 607
    num_train = args.num_case - num_test * 2
    num_generate = 5000

    # load data
    print('load data')
    case = np.zeros((args.num_case, 9, 9, 9))
    # test = np.zeros((num_test, 9, 9, 9))

    # list = io.load_list(args.EUDT_text)

    with open(args.EUDT_text, 'rt') as f:
        i = 0
        for line in f:
            if i >= args.num_case:
                break
            line = line.split()
            case[i, :] = IO.read_mhd_and_raw(line[0])
            i += 1


    # with open("E:/from_kubo/ConsoleApplication1/x64/Release/output/noise/filename.txt", 'rt') as f:
    #     i = 0
    #     for line in f:
    #         if i >= 100:
    #             break
    #         line = line.split()
    #         test[i, :] = IO.read_mhd_and_raw(line[0])
    #         i += 1

    case = np.reshape(case, [args.num_case, case_size])


    # normalization
    case_max = np.max(case,axis=1)
    case_min = np.min(case,axis=1)

    for i in range(args.num_case):
        case[i] = (case[i] - case_min[i]) / (case_max[i] - case_min[i])

    train = case[1214:3039, :]
    test = case[:num_test,:]
    train = np.reshape(train, [num_train, case_size])
    test = np.reshape(test, [num_test, case_size])
    print(case.shape)
    print(train.shape)
    print(test.shape)

    # print(test)

    # 寄与率
    PCA = sklearn.decomposition.PCA()
    PCA.fit(train)
    ev_ratio = PCA.explained_variance_ratio_
    ev_ratio = np. hstack([0, ev_ratio.cumsum()])
    plt.plot(ev_ratio)
    plt.grid()
    plt.show()

    # Prepare for pca
    print('process pca')
    pca = PCA_sklearn.PCA(n_components=args.components, svd_solver='arpack')

    # Do pca and map to Principal component
    pca.fit(train)

    # mean_vector
    mean_vector = pca.mean_

    # components
    U = pca.components_

    # eigen_vector
    eigen_vector = pca.explained_variance_

    # output_result
    ratio = pca.explained_variance_ratio_  # CCR
    ratio = np.cumsum(ratio)
    # np.savetxt(args.Dirout + '/CCR.txt', ratio, delimiter=',', fmt='%.6f')


    IO.write_raw(mean_vector, args.Dirout + '/mean.vect')  # mean

    IO.write_raw(eigen_vector, args.Dirout + '/eval.vect')  # eigen_vector
    # np.savetxt(args.Dirout + '/eval.txt', eigen_vector, delimiter=',', fmt='%.6f')

    for i in range(0, 2):
        IO.write_raw(U[i, :].copy(), args.Dirout + '/vect_' + str(i).zfill(4) + '.vect')  # PCs


    # projection
    z_train = pca.transform(train)
    z_test = pca.transform(test)

    mu = np.mean(z_train, axis=0)
    var = np.var(z_train, axis=0)

    print(mu.shape)
    print(var.shape)

    # inverse
    rec = pca.inverse_transform(z_test)

    # generalization
    test_img = test.reshape([num_test, patch_side, patch_side, patch_side])
    rec_img = rec.reshape([num_test, patch_side, patch_side, patch_side])
    L1 = []
    for j in range(len(rec)):
        # EUDT
        eudt_image = sitk.GetImageFromArray(rec_img[j])
        eudt_image.SetOrigin([0, 0, 0])
        eudt_image.SetSpacing([0.885, 0.885, 1])

        # calculate L1norm
        L1.append(utils.L1norm(test[j], rec[j]))
        # print('test{}'.format(j), test[j])
        # print('rec{}'.format(j),rec[j].shape)

        # output image
        io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.Dirout,'reconstruction/', 'EUDT', 'recon_{}'.format(j + 1))))

    # print('L1=', L1)
    # Generalizaton
    generalization = np.average(L1)
    print('generalization = %f' % generalization)
    # np.savetxt(os.path.join(args.Dirout, 'generalization.csv'), L1, delimiter=",")

    # bp
    generate_data = []
    specificity = []
    for k in range(num_generate):
        sample_z = np.random.normal(mu, var, (1, args.components))
        generate_data_single = pca.inverse_transform(sample_z)
        generate_data_single = generate_data_single[0, :]
        generate_data.append(generate_data_single)
        gen = np.reshape(generate_data_single, [9,9,9])

        # EUDT
        eudt_image = sitk.GetImageFromArray(gen)
        eudt_image.SetOrigin([0, 0, 0])
        eudt_image.SetSpacing([0.885, 0.885, 1])

        # output image
        io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.Dirout,'generate/', 'EUDT', 'spe_{}'.format(k + 1))))

    # generate_data = np.reshape(generate_data, [num_generate, 9 * 9 * 9])

        # Specificity
        case_min_specificity = 1.0
        # print(test.shape)
        # print(generate_data_single.shape)
        for image_index in range(test.shape[0]):
            # specificity = np.mean(abs(test[image_index] - generate_data))
            specificity_tmp = utils.L1norm(test[image_index], generate_data_single)
            # print('t=',test[image_index])
            # print('g=',generate_data_single)
            # print('t{}'.format(image_index),test[image_index].shape)
            if specificity_tmp < case_min_specificity:
                case_min_specificity = specificity_tmp
        # np.append(specificity, [case_min_specificity])
        #     print(specificity)
        specificity.append([case_min_specificity])

        # np.append(specificity, case_min_specificity)
        # print(specificity)

    # print(specificity)
    print('specificity = %f' % np.mean(specificity))
    # np.savetxt(os.path.join(args.Dirout, 'specificity.csv'), specificity, delimiter=",")


    # plot latent space
    plt.figure(figsize=(8, 6))
    # plt.subplot(2, 1, 1)
    plt.scatter(z_test[:, 0], z_test[:, 1])
    # plt.scatter(z_test[:, 0], z_test[:, 1], color='orange')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('latent distribution')
    plt.savefig(args.Dirout + "/latent.png")

    # for 3D
    # plt.figure(figsize=(8, 6))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(Xd[:,0], Xd[:,1], Xd[:,2] , marker="x")
    # ax.scatter(z_test[:,0], z_test[:,1],z_test[:,2], color='orange', marker="o")
    # plt.show()
    # plt.savefig(args.Dirout + "/latent_3d.png")
    # plt.subplot(2, 1, 2)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.xlabel('0')
    # plt.ylabel('1')
    # plt.show()

    # df = pd.DataFrame(case.data)
    # df.head()
    # sns.pairplot(df.iloc[:,:-1])

    # plot
    fig, axes = plt.subplots(ncols=10, nrows=2, figsize=(18, 4))

    test = np.reshape(test, [num_test, 9, 9, 9])
    rec = np.reshape(rec, [num_test, 9, 9, 9])
    print(test.shape)
    test = test[:,4,:]
    rec = rec[:,4,:]
    print(rec.shape)

    for i in range(10):
        axes[0, i].imshow(test[i, :].reshape(9, 9),cmap=cm.Greys_r)
        axes[0, i].set_title('original %d' % i)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)

        axes[1, i].imshow(rec[i, :].reshape(9, 9),cmap=cm.Greys_r)
        axes[1, i].set_title('reconstruction %d' % i)
        axes[1, i].get_xaxis().set_visible(False)
        axes[1, i].get_yaxis().set_visible(False)

    # plt.savefig(args.Dirout + "/reconstruction.png")

    # plt.show()

    #### display a 2D manifold of digits
    n = 13
    digit_size = 9
    figure1 = np.zeros((digit_size * n, digit_size * n))
    figure2 = np.zeros((digit_size * n, digit_size * n))
    figure3 = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    # generate_data = np.reshape(generate_data, [num_generate, 9 * 9 * 9])
    generate_data = np.reshape(generate_data[0], [9, 9, 9])
    print(generate_data.shape)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            digit_axial = generate_data[4, :, :]
            digit_coronal = generate_data[:, 4, :]
            digit_sagital = generate_data[:, :, 4]
            digit1 = np.reshape(digit_axial, [9, 9])
            digit2 = np.reshape(digit_coronal, [9, 9])
            digit3 = np.reshape(digit_sagital, [9, 9])
            # plt.imshow(digit, cmap='Greys_r')
            # plt.savefig(str(i) + '@' + str(j) + 'fig.png')
            figure1[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit1
            figure2[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit2
            figure3[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit3

    # set graph
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)

    # axial
    plt.figure(figsize=(10, 10))
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure1, cmap='Greys_r')
    # plt.savefig(args.Dirout + "/digit_axial.png")
    # plt.show()

    # coronal
    plt.figure(figsize=(10, 10))
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure2, cmap='Greys_r')
    # plt.savefig(args.Dirout + "/digit_coronal.png")
    # plt.show()

    # sagital
    plt.figure(figsize=(10, 10))
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure3, cmap='Greys_r')
    # plt.savefig(args.Dirout + "/digit_sagital.png")
    # plt.show()



if __name__ == '__main__':
    main()
