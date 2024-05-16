import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DatasetFromHdf5
# from model import PanNet,summaries
from model import KNLNet
import numpy as np
import scipy.io as sio
import shutil
from torch.utils.tensorboard import SummaryWriter
import time


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True  
cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

lr = 1e-4
epochs = 1000
ckpt_step = 50
batch_size = 32

model = KNLNet().cuda()

PLoss = nn.L1Loss(size_average=True).cuda()
# Sparse_loss = SparseKLloss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   # optimizer 1

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                              gamma=0.1)  # lr = lr* 1/gamma for each step_size = 180




writer = SummaryWriter("train_logs/ "+model_folder)
def save_checkpoint(model, epoch):  # save model function

    model_out_path = model_folder + "{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr":lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def train(training_data_loader, validate_data_loader,start_epoch=0):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=2)
    print('Start training...')


    time_s = time.time()
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            GT, LRHSI, HRMSI  = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

            optimizer.zero_grad()  # fixed

            output_HRHSI = model(LRHSI,HRMSI)
            time_e = time.time()
            Pixelwise_Loss =PLoss(output_HRHSI, GT)


            Myloss = Pixelwise_Loss
            epoch_train_loss.append(Myloss.item()) 

            Myloss.backward() 
            optimizer.step() 

            if iteration % 10 == 0:

                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Myloss.item()))

        print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        lr_scheduler.step()  

        t_loss = np.nanmean(np.array(epoch_train_loss)) 
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  
        print(time_e - time_s)
        if epoch % ckpt_step == 0:  
            save_checkpoint(model, epoch)

        if epoch % 50== 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT,  LRHSI, HRMSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                    output_HRHSI = model(
                        LRHSI, HRMSI)
                    time_e = time.time()
                    Pixelwise_Loss = PLoss(output_HRHSI, GT)
                    MyVloss = Pixelwise_Loss
                    epoch_val_loss.append(MyVloss.item())
            LRHSI1 = LRHSI[0, [10, 20, 30], ...].float().permute(1, 2, 0).cpu().numpy()
            axes[0, 0].imshow(LRHSI1)
            axes[0, 1].imshow(HRMSI[0, ...].permute(1, 2, 0).cpu().numpy())
            axes[1, 0].imshow(output_HRHSI[0, [10, 20, 30], ...].permute(1, 2, 0).cpu().detach().numpy())
            axes[1, 1].imshow(GT[0, [10, 20, 30], ...].permute(1, 2, 0).cpu().numpy())
            plt.pause(0.1)
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/loss', v_loss, epoch)
            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
    writer.close()  # close tensorboard



if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    train_set = DatasetFromHdf5('train_cave.h5')  
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                        pin_memory=True, drop_last=True) 
    validate_set = DatasetFromHdf5('validation_cave.h5')  
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                        pin_memory=True, drop_last=True)  
    train(training_data_loader, validate_data_loader)

