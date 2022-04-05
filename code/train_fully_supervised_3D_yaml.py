import logging
import os
import random
import sys

import numpy as np
import torch
from torch.backends import cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision.utils import make_grid
from tqdm import tqdm

from config_yaml import load_config
from networks.net_factory_3d import net_factory_3d
from utils import losses
from dataloader.utils import _loader_classes

def train(config):
    # create learning rate adjustment strategy
    lr_scheduler_config = config['lr_scheduler']
    base_lr = lr_scheduler_config['base_lr']

    # create the model
    model_config = config['model']
    model = net_factory_3d(net_type=model_config['name'],
                           in_chns=model_config['in_channels'],
                           class_num=model_config['class_num'])
    # create loss criterion

    # create evaluation metric
    # create data loaders
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']
    logging.info('Creating training and validation set loaders...')
    ## get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logging.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)
    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"
    train_datasets = dataset_class.create_datasets(loaders_config, phase='train')
    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')
    num_workers = loaders_config.get('num_workers', 1)
    logging.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['basic']['device'].type == 'cpu':
        logging.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()
    logging.info(f'Batch size for train/val loader: {batch_size}')
    trainloader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers),
    # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
    valloader = DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # create the optimizer
    ## sets the model in training mode
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    # Create loss criterion
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    # Create learning rate adjustment strategy
    # ...
    # Create trainer
    trainer_config = config['trainer']
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)
    max_iterations = trainer_config.pop('max_num_iterations', 30000)
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    # if resume is not None:
    #     logger.info(f"Loading checkpoint '{resume}'...")
    #     state = utils.load_checkpoint(resume, self.model, self.optimizer)
    #     logger.info(
    #         f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
    #         f"Best val score: {state['best_eval_score']}."
    #     )
    #     self.best_eval_score = state['best_eval_score']
    #     self.num_iterations = state['num_iterations']
    #     self.num_epochs = state['num_epochs']
    #     self.checkpoint_dir = os.path.split(resume)[0]
    # elif pre_trained is not None:
    #     logger.info(f"Logging pre-trained model from '{pre_trained}'...")
    #     utils.load_checkpoint(pre_trained, self.model, None)
    #     if 'checkpoint_dir' not in kwargs:
    #         self.checkpoint_dir = os.path.split(pre_trained)[0]
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = _split_training_batch(sampled_batch)
            # forward pass
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                # ...

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
        writer.close()
        return "Training Finished!"


if __name__ == '__main__':
    # load configuration
    config = load_config()
    # snapshot_path
    snapshot_path = "../model/{}/{}".format(config.basic.exp, config.model.name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config['snapshot_path'] = snapshot_path
    # log experiment configuration
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    basic_config = config['basic']
    # seed config
    seed = basic_config.pop('seed', None)
    if seed is not None:
        logging.info(f'Seed the RNG for all devices with {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    deterministic = basic_config.pop('deterministic', False)
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        logging.warning('Using CuDNN deterministic setting. This may slow down the training!')
        cudnn.benchmark = False
        cudnn.deterministic = True
    # start training
    train(config)







