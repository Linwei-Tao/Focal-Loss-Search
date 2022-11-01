import torch.nn as nn
import utils
import torch
import hfai.nccl.distributed as dist
import hfai
import time


def model_train(train_queue, model, lossfunc, optimizer, name, args, gumbel_training=True, epoch=0, start_step=0,
                loss_scaler=None, local_rank=0, save_path=None, scheduler=None, mode=None):
    # set model to training model
    model.train()
    for step, batch in enumerate(train_queue):
        if dist.get_rank() == 0:
            print("step: ", step)
        if step < start_step:
            continue

        # data to CUDA
        samples, labels = [x.cuda(non_blocking=True) for x in batch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            nll, loss = lossfunc(outputs, labels)
        loss_scaler.scale(loss).backward()

        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 保存
        if dist.get_rank() == 0 and local_rank == 0 and hfai.client.receive_suspend_command():
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'step': step + 1
            }
            torch.save(state, save_path / '{}_latest.pt'.format(mode))
            time.sleep(5)
            hfai.client.go_suspend()
