from collections import defaultdict

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable
from tqdm.auto import tqdm
git
import Capsnet
import Loss
from torch.utils.tensorboard import SummaryWriter


INPUT_SIZE = (1, 28, 28)
transforms=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(INPUT_SIZE[1:], padding=2),
    torchvision.transforms.ToTensor()
])

train_data=torchvision.datasets.MNIST('../data',train=True,download=True,transform=transforms)
test_data=torchvision.datasets.MNIST('../data',train=False,download=True,transform=transforms)

print('训练集的大小：%d'%len(train_data))
print('测试集的大小：%d'%len(test_data))

batchsize=64*8

train_loader=torch.utils.data.DataLoader(train_data,batchsize,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batchsize,shuffle=False)

model=Capsnet.CapsNet().cuda()
#model.load_state_dict(torch.load('model/checkpoint-0.992800-0016.pth')['model_state_dict'])
print('Number of Parameters: %d' % model.n_parameters())
print(model)

Loss=Loss.Capsule_Loss()


def exponential_decay(optimizer, learning_rate, global_step, decay_steps, decay_rate, staircase=False):
    if (staircase):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
    else:
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step / decay_steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = decayed_learning_rate

    return optimizer

learning_rate=0.001
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9,0.999),
    eps=1e-08,
)
global_epoch = 0
global_step = 0
best_tst_accuracy = 0.0
history = defaultdict(lambda:list())
COMPUTE_TRN_METRICS = False
writer=SummaryWriter("logs")
n_epochs = 1500  # Number of epochs not specified in the paper

def save_checkpoint(epoch,train_accuracy,test_accuracy,model,optimizer,path=None):
    if(path is None):
        path='model/checkpoint-%f-%04d.pth' % (test_accuracy, epoch)
        state={
            'epoch': epoch,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
    torch.save(state,path)

def show_example(x,y,x_reconstruction,y_pred):
    x=x.squeeze().cpu().data.numpy()
    y=y.cpu().data.numpy()
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)
    y_pred = y_pred.cpu().data.numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.show()

def test(model, loader):
    metrics = defaultdict(lambda:list())
    for batch_id, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x = Variable(x).float().cuda()
        y = Variable(y).cuda()
        y_pred, x_reconstruction = model(x, y)
        _, y_pred = torch.max(y_pred, -1)
        metrics['accuracy'].append((y_pred == y).cpu().data.numpy())
    metrics['accuracy'] = np.concatenate(metrics['accuracy']).mean()
    return metrics


for epoch in range(n_epochs):
    print('Epoch %d (%d/%d):' % (global_epoch + 1, epoch + 1, n_epochs))

    for batch_id, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer = exponential_decay(optimizer, learning_rate, global_epoch, 1,
                                      0.90)  # Configurations not specified in the paper

        x = Variable(x).float().cuda()
        y = Variable(y).cuda()

        y_pred, x_reconstruction = model(x, y)
        loss, margin_loss, reconstruction_loss = Loss(x, y, x_reconstruction, y_pred.cuda())

        history['margin_loss'].append(margin_loss.cpu().data.numpy())
        history['reconstruction_loss'].append(reconstruction_loss.cpu().data.numpy())
        history['loss'].append(loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Total_loss", loss, global_step)
        writer.add_scalar("margin_loss", margin_loss, global_step)

        global_step += 1

    trn_metrics = test(model, train_loader) if COMPUTE_TRN_METRICS else None
    tst_metrics = test(model, test_loader)

    print('Margin Loss: %f' % history['margin_loss'][-1])
    print('Reconstruction Loss: %f' % history['reconstruction_loss'][-1])
    print('Loss: %f' % history['loss'][-1])
    print('Train Accuracy: %f' % (trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0))
    print('Test Accuracy: %f' % tst_metrics['accuracy'])
    print('Example:')
    idx = np.random.randint(0, len(x))
    #show_example(x[idx], y[idx], x_reconstruction[idx], y_pred[idx])

    if (tst_metrics['accuracy'] >= best_tst_accuracy):
        best_tst_accuracy = tst_metrics['accuracy']
        save_checkpoint(
            global_epoch + 1,
            trn_metrics['accuracy'] if COMPUTE_TRN_METRICS else 0.0,
            tst_metrics['accuracy'],
            model,
            optimizer
        )
    global_epoch += 1



writer.close()


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean

n_points_avg = 10
n_points_plot = 1000
plt.figure(figsize=(20, 10))

curve = np.asarray(history['loss'])[-n_points_plot:]
avg_curve = compute_avg_curve(curve, n_points_avg)
plt.plot(avg_curve, '-g')

curve = np.asarray(history['margin_loss'])[-n_points_plot:]
avg_curve = compute_avg_curve(curve, n_points_avg)
plt.plot(avg_curve, '-b')

curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
avg_curve = compute_avg_curve(curve, n_points_avg)
plt.plot(avg_curve, '-r')

plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
plt.show()