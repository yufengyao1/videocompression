import cv2
import torch
import numpy as np
from tqdm import tqdm
from loss import C3DLoss
from datasets import VideoDatasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from c3d_gan import C3dGenerator, C3dDescriminator
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    G_model = C3dGenerator().to(device)
    G_model.load_state_dict(torch.load('weights/com_796.pth', map_location='cuda'))  # 196
    D_model = C3dDescriminator().to(device)
    model_test = C3dGenerator()

    LR_G, LR_D = 3e-4, 1e-5
    optimizer_G = torch.optim.Adam(G_model.parameters(), lr=LR_G)
    optimizer_D = torch.optim.Adam(D_model.parameters(), lr=LR_D)

    criterion = C3DLoss()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = VideoDatasets(transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    for epoch in range(0, 2000):
        train_loss, val_loss, g_losses, d_losses, l2_losses = 0, 0, [], [], []
        for img_true in tqdm(train_loader, leave=False):
            img_true = img_true.to(device)
            img_generated = G_model(img_true)  # 生成图

            # l2_loss = criterion(img_true, img_generated)

            des_true = D_model(img_true)
            des_generated = D_model(img_generated)

            G_loss = -1/torch.mean(torch.log(1.-des_generated))
            D_loss = -torch.mean(torch.log(des_true)+torch.log(1-des_generated))

            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)

            optimizer_D.zero_grad()
            D_loss.backward()

            optimizer_G.step()
            optimizer_D.step()

            g_losses.append(G_loss.item())
            d_losses.append(D_loss.item())
            # l2_losses.append(l2_loss.item())

        g_loss = np.mean(g_losses)
        d_loss = np.mean(d_losses)
        dis_loss = np.mean(l2_losses)

        torch.save(G_model.state_dict(), "weights/{0}_{1}.pth".format("gan", epoch))
        print('epoch:{}, dis_loss:{:.6f}, g_loss:{:.6f}, d_loss:{:.6f}'.format(epoch, dis_loss, g_loss, d_loss))

        # 测试
        model_test.load_state_dict(torch.load("weights/{0}_{1}.pth".format("gan", epoch), map_location='cpu'))
        model_test.eval()

        dataset = VideoDatasets()
        input = dataset.__getitem__(0).unsqueeze(0)

        frames = torch.permute(input, (0, 2, 4, 3, 1)).detach().numpy()[0]*255
        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/tmp/{i}.jpg", frame)

        pred = model_test(input)
        pred = torch.permute(pred, (0, 2, 4, 3, 1)).detach().numpy()[0]*255
        pred.astype(int)

        for i, frame in enumerate(pred):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/tmp/{i}_new.jpg", frame)
