import os
import math
import numpy as np
import argparse
import random

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
from torch.autograd import Variable
from source.Z_TimeSignalAnalysis.params import opt as args

# from Util.data import path_NIC
from BrainWaveletModel import Model, LineTransformer
import scipy.io as scio

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def train(args):
    DataNorm = LineTransformer()
    input_dir = args.input_dir
    test_input_dir = args.test_input_dir

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    ckpt_dir = args.save_model_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logfile = open(args.log_dir + '/log.txt', 'w')

    total_epoch = args.epochs
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    init_scale_list = args.scale_list
    lambda_d = args.lambda_d

    epoch = 1
    model = Model(wavelet_affine=False)

    # 导入数据集
    logfile.write('Load data starting...' + '\n')
    logfile.flush()
    print('Load data starting...')
    data = scio.loadmat(args.pre_data_root)

    # fMRI_Dataset
    # batch_size = args.batchSize
    # num_node = args.num_ROI
    timeseries = torch.from_numpy(data['timeseries'])

    # normalize signal data
    # Q1 = torch.quantile(timeseries, 0.25)
    # Q2 = torch.quantile(timeseries, 0.50)
    # Q3 = torch.quantile(timeseries, 0.75)
    # IQR = Q3-Q1
    #
    # timeseries = (timeseries - Q2)/IQR

    max_signal = torch.ceil(torch.quantile(abs(timeseries), 0.95))
    timeseries = timeseries.cuda()
    # timeseries = torch.clamp(timeseries, min=1e-6).cuda()
    # size = timeseries.size()
    # timeseries_tmp = timeseries.reshape(-1, timeseries.shape[-1])
    # timeseries = DataNorm.transform(timeseries_tmp).reshape(size)

    all_label = torch.from_numpy(data['label'].squeeze(1)).cuda()
    tr_set = torch.from_numpy(data['tr_set'][1][0].astype(int)) - 1
    te_set = torch.from_numpy(data['te_set'][1][0].astype(int)) - 1
    # val_set = torch.from_numpy(data['val_set'][1][0].astype(int).squeeze(0)) - 1
    test_dataset = TensorDataset(timeseries[te_set.tolist()], all_label[te_set.tolist()])
    all_dataset_index = torch.cat([tr_set, te_set], dim=-1).tolist()
    all_train_dataset = TensorDataset(timeseries[tr_set.tolist()], all_label[tr_set.tolist()])
    all_dataset = TensorDataset(timeseries[all_dataset_index], all_label[all_dataset_index])

    train_loader = DataLoader(all_dataset, batch_size=args.patch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.patch_size, shuffle=False)

    logfile.write('Load all data succeed!' + '\n')
    logfile.flush()
    print('Load all data succeed!')
    max_step = train_loader.__len__()
    max_step = 20000 if 20000 < max_step else max_step
    print("train step:", max_step, end=" ")
    logfile.write("train step:" + str(max_step))
    test_max_step = test_loader.__len__()
    test_max_step = 1000 if 1000 < test_max_step else test_max_step
    print("test step:", test_max_step)
    logfile.write(" test step:" + str(test_max_step) + '\n')

    # 导入预训练的模型参数
    if args.load_addi_model is not None:
        model_dict = model.state_dict()
        checkpoint = torch.load(args.load_addi_model)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        print('Load pre-trained model [' + args.load_addi_model + '] succeed!')
        logfile.write('Load pre-trained model [' + args.load_addi_model + '] succeed!' + '\n')
        logfile.flush()
    else:
        if args.rate_model is not None:
            model_dict = model.state_dict()
            checkpoint = torch.load(args.rate_model)
            part_dict_org = checkpoint['state_dict']
            print("pre train args.rate_model", end=" ")
            logfile.write("pre train args.rate_model" + ' ')
            part_dict = {k: v for k, v in part_dict_org.items() if "coding" in k}
            for name, param in part_dict.items():
                print("pretrain " + name, end=" ")
                logfile.write("pretrain " + name + ' ')
            model_dict.update(part_dict)
            model.load_state_dict(model_dict)
            print('Load pre-trained part model [' + args.rate_model + '] succeed!')
            logfile.write('Load pre-trained part model [' + args.rate_model + '] succeed!' + '\n')
            logfile.flush()
        if args.post_model is not None:
            model_dict = model.state_dict()
            checkpoint = torch.load(args.post_model)
            part_dict_org = checkpoint['state_dict']
            print("pre train args.post_model", end=" ")
            logfile.write("pre train args.post_model" + ' ')
            part_dict = {k: v for k, v in part_dict_org.items() if "post" in k and "coding" not in k}
            for name, param in part_dict.items():
                print("pretrain " + name, end=" ")
                logfile.write("pretrain " + name + ' ')
            model_dict.update(part_dict)
            model.load_state_dict(model_dict)
            print('Load pre-trained part model [' + args.post_model + '] succeed!')
            logfile.write('Load pre-trained part model [' + args.post_model + '] succeed!' + '\n')
            logfile.flush()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    model.train()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    if args.load_addi_model is None and args.load_affi_model is None and (
            args.post_model is None or args.rate_model is None):
        # 没有预训练模型作为熵编码和后处理模块的初始化。先使用传统的97小波，训练熵编码和后处理模块
        while True:
            if epoch > 10:
                break

            ori_all = 0.
            post_all = 0.
            bpp_all = 0.
            loss_all = 0.
            scale_all = 0
            for batch_idx, input_signal in enumerate(train_loader):
                if batch_idx > max_step - 1:
                    break

                # input_img_v = to_variable(input_img[0]) * 255.
                input_signal_v = input_signal[0][:, :, :160]
                input_signal_v = input_signal_v.unsqueeze(1)
                input_signal_v = input_signal_v.float()
                size = input_signal_v.size()
                # 因为只训练后处理和熵编码模块不涉及到量化之前的网络，因此下面的量化直接使用round便可
                mes_loss, bits, reference_signal_v = model(input_signal_v, alpha=0, train=0, scale_init=init_scale_list[0],
                                                          wavelet_trainable=0)
                bpp = torch.sum(bits) / size[0]

                # 计算信号的PSNR
                # max_signal = input_signal_v.view(-1, size[3]).max(dim=-1).values
                # max_signal = max_signal.mean()
                # mse_ori = torch.mean(mse_ori)
                # psnr_ori = 10. * torch.log10(max_signal * max_signal / mse_ori)
                # mse_post = torch.mean(mse_post)
                # psnr_post = 10. * torch.log10(255. * 255. / mse_post)
                # 获取信号的最大值
                # 可以根据实际情况选择 max() 还是 max(dim=-1)，这里我们取信号的绝对最大值


                # 计算均方误差 MSE
                mse_ori = torch.mean((input_signal_v - reference_signal_v) ** 2)

                # 计算 PSNR
                psnr_ori = 10. * torch.log10(max_signal**2 / mse_ori)
                scale = torch.sum(mes_loss) / size[0]

                loss = bpp + lambda_d * mse_ori
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
                opt.step()

                ori_all += psnr_ori.item()
                # post_all += psnr_post.item()
                bpp_all += bpp.item()
                loss_all += loss.item()
                scale_all += scale.item()

                if batch_idx % 5 == 0:
                    logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                                  + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                    logfile.flush()
                    print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                          + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

            if epoch % 1 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save({'epoch': epoch, 'state_dict': model.module.state_dict()},
                               ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_post_entropy.pth',
                               _use_new_zipfile_serialization=False)
                # else:
                #     torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                #                ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_post_entropy.pth',
                #                _use_new_zipfile_serialization=False)
                logfile.write('psnr_mean: ' + str(ori_all / max_step) + '\n')
                # logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
                logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
                logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
                logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
                logfile.flush()
                print('psnr_mean: ' + str(ori_all / max_step))
                # print('post_mean: ' + str(post_all / max_step))
                print('bpp_mean: ' + str(bpp_all / max_step))
                print('loss_mean: ' + str(loss_all / max_step))
                print('scale_mean: ' + str(scale_all / max_step))

            epoch = epoch + 1

    alpha = np.array(args.alpha_start, dtype=np.float32)
    alpha = torch.from_numpy(alpha)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        alpha = alpha.unsqueeze(0)
        alpha = torch.repeat_interleave(alpha, repeats=torch.cuda.device_count(), dim=0)

    epoch = 11
    for name, param in model.named_parameters():
        param.requires_grad = True
    # 端到端训练熵编码，后处理部分，可训练的小波 soft to hard
    if (args.load_addi_model is None) and (args.load_affi_model is None):
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        while True:
            if epoch > 25:
                break

            print('train soft additive wavelet')
            logfile.write('train soft additive wavelet ' + '\n')
            logfile.flush()

            print("alpha:", alpha[0].item())
            logfile.write('alpha: ' + str(alpha[0].item()) + '\n')
            logfile.flush()
            model.train()

            ori_all = 0.
            post_all = 0.
            bpp_all = 0.
            loss_all = 0.
            scale_all = 0.

            for batch_idx, input_signal in enumerate(train_loader):

                if batch_idx > max_step - 1:
                    break

                input_signal_v = input_signal[0][:, :, :160].float()
                input_signal_v = input_signal_v.unsqueeze(1)
                size = input_signal_v.size()
                mes_loss, bits, reference_signal_v = model(input_signal_v, alpha=alpha, train=1,
                                                          scale_init=init_scale_list[0], wavelet_trainable=1)
                bpp = torch.sum(bits) / size[0]

                # 计算信号的PSRN
                # 获取信号的最大值
                # 可以根据实际情况选择 max() 还是 max(dim=-1)，这里我们取信号的绝对最大值
                # max_signal = input_signal_v.abs().max()

                # 计算均方误差 MSE
                mse_ori = torch.mean((input_signal_v - reference_signal_v) ** 2)

                # 计算 PSNR
                psnr_ori = 10. * torch.log10(max_signal**2 / mse_ori)
                scale = torch.sum(mes_loss) / size[0]

                loss = bpp + lambda_d * mse_ori
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
                opt.step()

                ori_all += psnr_ori.item()
                # post_all += psnr_post.item()
                bpp_all += bpp.item()
                loss_all += loss.item()
                scale_all += scale.item()

                if batch_idx % 5 == 0:
                    logfile.write('Train Epoch: [' + '{:2d}'.format(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                                  + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                    logfile.flush()
                    print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                          + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

            if epoch % 1 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict()},
                               ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_soft.pth',
                               _use_new_zipfile_serialization=False)
                # else:
                #     torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict()},
                #                ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_soft.pth',
                #                _use_new_zipfile_serialization=False)
                logfile.write('psnr_mean: ' + str(ori_all / max_step) + '\n')
                # logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
                logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
                logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
                logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
                logfile.flush()
                print('psnr_mean: ' + str(ori_all / max_step))
                # print('post_mean: ' + str(post_all / max_step))
                print('bpp_mean: ' + str(bpp_all / max_step))
                print('loss_mean: ' + str(loss_all / max_step))
                print('scale_mean: ' + str(scale_all / max_step))

            epoch = epoch + 1
            if alpha[0] < args.alpha_end:
                alpha += 2.0

    epoch = 25
    # 端到端训练熵编码，后处理部分，可训练的小波 soft to hard. 如果有预训练模型，可以省去这一阶段的训练。如果没有预训练模型，需要进行第二次soft to hard 来达到较好的效果。
    if args.wavelet_affine:
        if args.load_affi_model is None:
            if torch.cuda.device_count() > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            model = Model(wavelet_affine=True)  # 使用additive wavelet 初始化affine wavelet

            for k, v in list(state_dict.items()):
                if 'wavelet_transform.lifting' in k:
                    state_dict.pop(k)
                    state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.0.lifting0')] = v
                    state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.0.lifting1')] = v
                    state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.0.lifting2')] = v

                    state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.1.lifting0')] = v
                    state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.1.lifting1')] = v
                    state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.1.lifting2')] = v

                    # state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.2.lifting0')] = v
                    # state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.2.lifting1')] = v
                    # state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.2.lifting2')] = v
                    #
                    # state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.3.lifting0')] = v
                    # state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.3.lifting1')] = v
                    # state_dict[k.replace('wavelet_transform.lifting', 'wavelet_transform.3.lifting2')] = v

            model_dict = model.state_dict()
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print('Load pre-trained model [ additive ] ')
            logfile.write('Load pre-trained model [ additive ] ' + '\n')
            logfile.flush()
        else:
            model = Model(wavelet_affine=True)  # 导入初始化affine wavelet
            model_dict = model.state_dict()
            checkpoint = torch.load(args.load_affi_model)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch'] + 1
            model.load_state_dict(state_dict)
            print('Load pre-trained part model [' + args.load_affi_model + '] succeed!')
            logfile.write('Load pre-trained part model [' + args.load_affi_model + '] succeed!' + '\n')
            logfile.flush()

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()

    alpha = np.array(args.alpha_start, dtype=np.float32)
    alpha = torch.from_numpy(alpha)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        alpha = alpha.unsqueeze(0)
        alpha = torch.repeat_interleave(alpha, repeats=torch.cuda.device_count(), dim=0)

    for name, param in model.named_parameters():
        param.requires_grad = True
    if args.wavelet_affine:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    else:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    while True:
        if epoch > 35:
            break

        print('train soft affine wavelet')
        logfile.write('train soft affine wavelet ' + '\n')
        logfile.flush()

        print("alpha:", alpha[0].item())
        logfile.write('alpha: ' + str(alpha[0].item()) + '\n')
        logfile.flush()
        model.train()

        ori_all = 0.
        post_all = 0.
        bpp_all = 0.
        loss_all = 0.
        scale_all = 0.

        for batch_idx, input_signal in enumerate(train_loader):

            if batch_idx > max_step - 1:
                break

            input_signal_v = input_signal[0][:, :, :160].float()
            input_signal_v = input_signal_v.unsqueeze(1)
            size = input_signal_v.size()
            mes_loss, bits, reference_signal_v = model(input_signal_v, alpha=alpha, train=1, scale_init=init_scale_list[0],
                                                      wavelet_trainable=1)
            bpp = torch.sum(bits) / size[0]

            # 计算信号的PSRN
            # 获取信号的最大值
            # 可以根据实际情况选择 max() 还是 max(dim=-1)，这里我们取信号的绝对最大值
            # max_signal = input_signal_v.abs().max()

            # 计算均方误差 MSE
            mse_ori = torch.mean((input_signal_v - reference_signal_v) ** 2)

            # 计算 PSNR
            psnr_ori = 10. * torch.log10(max_signal**2 / mse_ori)
            scale = torch.sum(mes_loss) / size[0]

            loss = bpp + lambda_d * mse_ori
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            ori_all += psnr_ori.item()
            # post_all += psnr_post.item()
            bpp_all += bpp.item()
            loss_all += loss.item()
            scale_all += scale.item()

            if batch_idx % 5 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                              + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item())  + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                      + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item())  + '/' + str(bpp.item()) + '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_soft.pth',
                           _use_new_zipfile_serialization=False)
            # else:
            #     torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict()},
            #                ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_soft.pth',
            #                _use_new_zipfile_serialization=False)
            logfile.write('psnr_mean: ' + str(ori_all / max_step) + '\n')
            # logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
            logfile.flush()
            print('psnr_mean: ' + str(ori_all / max_step))
            # print('post_mean: ' + str(post_all / max_step))
            print('bpp_mean: ' + str(bpp_all / max_step))
            print('loss_mean: ' + str(loss_all / max_step))
            print('scale_mean: ' + str(scale_all / max_step))

        epoch = epoch + 1
        if alpha[0] < args.alpha_end:
            alpha += 2.0

    epoch = 36
    # soft then hard  固定小波变换部分，量化使用round即可
    for name, param in model.named_parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if 'scale_net' in name:
            param.requires_grad = False
        if 'wavelet_transform' in name:
            param.requires_grad = False

    if args.wavelet_affine:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6)
    else:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    while True:
        if epoch > 40:
            break

        print('train hard affine wavelet')
        logfile.write('train hard affine wavelet ' + '\n')
        logfile.flush()

        model.train()

        ori_all = 0.
        post_all = 0.
        bpp_all = 0.
        loss_all = 0.
        scale_all = 0.

        for batch_idx, input_signal in enumerate(train_loader):

            if batch_idx > max_step - 1:
                break

            input_signal_v = input_signal[0][:, :, :160].float()
            input_signal_v = input_signal_v.unsqueeze(1)
            size = input_signal_v.size()
            mes_loss, bits, reference_signal_v = model(input_signal_v, alpha=0, train=0, scale_init=init_scale_list[
                random.randint(0, len(init_scale_list) - 1)], wavelet_trainable=1)

            bpp = torch.sum(bits) / size[0]

            # 计算信号的PSRN
            # 获取信号的最大值
            # 可以根据实际情况选择 max() 还是 max(dim=-1)，这里我们取信号的绝对最大值
            # max_signal = input_signal_v.abs().max()

            # 计算均方误差 MSE
            mse_ori = torch.mean((input_signal_v - reference_signal_v) ** 2)

            # 计算 PSNR
            psnr_ori = 10. * torch.log10(max_signal**2 / mse_ori)
            scale = torch.sum(mes_loss) / size[0]

            loss = bpp + lambda_d * mse_ori
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            ori_all += psnr_ori.item()
            # post_all += psnr_post.item()
            bpp_all += bpp.item()
            loss_all += loss.item()
            scale_all += scale.item()

            if batch_idx % 5 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                              + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item())  + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                      + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item())  + '/' + str(bpp.item()) + '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_hard.pth',
                           _use_new_zipfile_serialization=False)
            # else:
            #     torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict()},
            #                ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_hard.pth',
            #                _use_new_zipfile_serialization=False)
            logfile.write('psnr_mean: ' + str(ori_all / max_step) + '\n')
            # logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
            logfile.flush()
            print('psnr_mean: ' + str(ori_all / max_step))
            # print('post_mean: ' + str(post_all / max_step))
            print('bpp_mean: ' + str(bpp_all / max_step))
            print('loss_mean: ' + str(loss_all / max_step))
            print('scale_mean: ' + str(scale_all / max_step))

        epoch = epoch + 1
    model_static = model.state_dict()
    torch.save(model_static, "D:/Users/qiqi/PycharmProject/iwave-main/source/Z_TimeSignalAnalysis/out/model/BWare_off_normalize.pth")
    logfile.close()


if __name__ == "__main__":
    print(args)
    train(args)
