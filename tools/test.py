import os
import time
import torch
def evaluation(net, test_loader, loss_func, device, epoch, save_dir):
    t0 = time.time()
    correct = 0
    total = 0
    avg_loss = 0
    n_samples = 1000
    for i, (images, labels) in enumerate(test_loader):
        if i == n_samples:
            break
        images = images.to(device)
        labels = labels.to(device)

        out = net(images)
        _, predicted = torch.max(out, 1)
        loss = loss_func(out, labels).to(device)
        avg_loss += loss.item()
        correct += (predicted==labels).sum()
        total += labels.size(0)
    avg_loss = avg_loss / n_samples
    with open(os.path.join(save_dir,'log_validation.csv'), 'a') as f:
        f.write('%d, %.4f, %.4f\n' % (epoch+1, (100.0 * correct) / (total + 1), avg_loss))
    print('Percent Accuracy: %.3f, Loss: %.4f ' % (((100.0 * correct) / (total + 1)), avg_loss))