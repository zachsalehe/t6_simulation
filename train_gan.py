import os
import glob
import numpy as np
import cv2
import torch
import torchvision
import scipy.io as sio
import lpips

import data_load as dl
import gan


# *
def define_loss(device):
    print('using lpips loss')

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    def gen_loss(in1, in2): 
        return torch.mean(loss_fn_alex(in1, in2, normalize=True))
        
    return gen_loss, loss_fn_alex


def load_dataset(data_dir):
    ct = torchvision.transforms.Compose([dl.ToTensor()])

    dataset_list = []
    dataset_list_test = []

    for i in range(50):
        img_dir = data_dir + "exp{:03d}/".format(i)

        if i % 5 != 4:
            dataset_list.append(dl.GetSampleBatch(img_dir, ct, bsize=16))
        else:
            dataset_list_test.append(dl.GetSampleBatch(img_dir, ct, bsize=16))

    return dataset_list, dataset_list_test


def train(device):
    data_dir = "data/t6_imgs5/rdata2/"
    save_dir = "data/t6_imgs5/rsaved/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_img_noisy = np.load(data_dir + "exp040/noisy0000.npy")
    test_img_clean = np.load(data_dir + "exp040/clean.npy")
    test_exp = np.load(data_dir + "exp040/exp.npy")[0]

    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    lambda_gp = 10
    num_epochs_gain = 10
    num_epochs = 10
    n_critic = 5

    gen_loss, loss_fn_alex = define_loss(device)

    dataset_list, dataset_list_test = load_dataset(data_dir)
    train_dataset = torch.utils.data.ConcatDataset(dataset_list)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataset = torch.utils.data.ConcatDataset(dataset_list_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    generator = gan.NoiseGenerator(device, r=320, c=320, n=1)
    generator.cuda(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    discriminator = gan.Discriminator(n=1)
    discriminator.cuda(device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    kld_list = []

    for epoch in range(num_epochs_gain):
        print(f"gain epoch {epoch} / {num_epochs_gain}")

        for i, sample in enumerate(train_loader):
            noisy = sample["noisy_imgs"][0].to(device)
            clean = sample["clean_imgs"][0].to(device)
            exp = sample["exp"].to(device)[0]

            gen_noisy = generator(clean, exp, False)

            loss_l2 = torch.mean((noisy - gen_noisy) ** 2)

            optimizer_g.zero_grad()
            loss_l2.backward()
            optimizer_g.step()

    generator.g.requires_grad = False
    generator.h.requires_grad = False

    for epoch in range(num_epochs):
        print(f"epoch {epoch} / {num_epochs}")

        for i, sample in enumerate(train_loader):
            noisy = sample["noisy_imgs"][0].to(device)
            clean = sample["clean_imgs"][0].to(device)
            exp = sample["exp"][0].to(device)

            gen_noisy = generator(clean, exp, True)

            noisy = gan.split_into_patches(noisy / (2**12-1)).to(device)
            gen_noisy = gan.split_into_patches(gen_noisy / (2**12-1)).to(device)

            noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(noisy, norm="ortho")))
            gen_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

            real_validity = discriminator(noisy)
            fake_validity = discriminator(gen_noisy)

            gp = gan.compute_gradient_penalty(discriminator, noisy.data, gen_noisy.data)

            loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()

            if i % n_critic == 0:
                gen_noisy = generator(clean, exp, True)

                gen_noisy = gan.split_into_patches(gen_noisy / (2**12-1)).to(device)

                gen_noisy = torch.abs(torch.fft.fftshift(torch.fft.fft2(gen_noisy, norm="ortho")))

                fake_validity = discriminator(gen_noisy)

                loss_lpips = gen_loss(noisy, gen_noisy)
                loss_g = -torch.mean(fake_validity) + loss_lpips

                loss_g.backward()
                optimizer_g.step()

            if i % 10 == 0:
                with torch.no_grad():
                    test_in = torch.tensor(test_img_clean[None].transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
                    test_in_exp = torch.tensor(test_exp, dtype=torch.float32, device=device)
                    test_out = generator(test_in, test_in_exp, True).detach().cpu().numpy().transpose(0, 2, 3, 1)[0]

                    out_img = np.clip(np.concatenate((test_out, test_img_noisy), axis=1) / 16, 0, 255).astype(np.uint8)
                    cv2.imwrite("test.png", out_img)

            print(f"epoch {epoch} / {num_epochs}, batch {i} / {len(train_loader)}, " + \
                    f"d loss {loss_d.item()}, g loss {loss_g.item()}, lpips loss {loss_lpips.item()}")

        total_kld = 0

        for i, sample in enumerate(test_loader):
            with torch.no_grad():
                noisy = sample["noisy_imgs"][0].to(device)
                clean = sample["clean_imgs"][0].to(device)
                exp = sample["exp"][0].to(device)

                gen_noisy = generator(clean, exp, True)

                noisy = gan.split_into_patches(noisy / (2**12-1)).to(device)
                gen_noisy = gan.split_into_patches(gen_noisy / (2**12-1)).to(device)

                noisy_np = noisy.detach().cpu().numpy()
                gen_noisy_np = gen_noisy.detach().cpu().numpy()
                kld = gan.cal_kld(noisy_np, gen_noisy_np)
                total_kld += kld
                print(f"exp {exp.item()} kld {kld}")

        kld_list.append(total_kld)
        print(f"total kld {total_kld}")

        with torch.no_grad():
            test_in = torch.tensor(test_img_clean[None].transpose(0, 3, 1, 2), dtype=torch.float32, device=device)
            test_in_exp = torch.tensor(test_exp, dtype=torch.float32, device=device)
            test_out = generator(test_in, test_in_exp, True).detach().cpu().numpy().transpose(0, 2, 3, 1)[0]

            out_img = np.clip(np.concatenate((test_out, test_img_noisy), axis=1) / 16, 0, 255).astype(np.uint8)
            cv2.imwrite(save_dir + f"{epoch:03d}_{total_kld:.5f}.png", out_img)

    np.save(save_dir + "klds.npy", np.array(kld_list))
    torch.save(generator.state_dict(), save_dir + "generator.pt")
    torch.save(discriminator.state_dict(), save_dir + "discriminator.pt")

    params = {}
    params["g"] = generator.g.detach().cpu().numpy()
    params["h"] = generator.h.detach().cpu().numpy()
    params["shot"] = generator.shot.detach().cpu().numpy()
    params["read"] = generator.read.detach().cpu().numpy()
    params["row"] = generator.row.detach().cpu().numpy()
    params["rowt"] = generator.rowt.detach().cpu().numpy()
    params["quant"] = generator.quant.detach().cpu().numpy()
    params["dark"] = generator.dark.detach().cpu().numpy()
    sio.savemat(save_dir + "params.mat", params)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit()
    else:
        device = torch.device("cuda")

    # torch.manual_seed(452)

    train(device)
# *
