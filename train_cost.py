import torch, numpy, argparse, pdb, os, time, math, random
import utils
from dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import importlib
import models
import torch.nn as nn
import utils

#################################################
# Train an action-conditional forward model
#################################################

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-v', type=int, default=4)
parser.add_argument('-dataset', type=str, default='i80')
parser.add_argument('-data_dir', type=str, default='traffic-data/state-action-cost/data_i80_v0/')
parser.add_argument('-model_dir', type=str, default='models/')
parser.add_argument('-ncond', type=int, default=20, help='number of conditioning frames')
parser.add_argument('-npred', type=int, default=20, help='number of predictions to make with unrolled fwd model')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-layers', type=int, default=3)
parser.add_argument('-nfeature', type=int, default=256)
parser.add_argument('-n_hidden', type=int, default=256)
parser.add_argument('-dropout', type=float, default=0.0, help='regular dropout')
parser.add_argument('-lrt', type=float, default=0.0001)
parser.add_argument('-grad_clip', type=float, default=5.0)
parser.add_argument('-epoch_size', type=int, default=1000)
parser.add_argument('-mfile', type=str, default='model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=0-seed=1.model')
#parser.add_argument('-mfile', type=str, default='model=fwd-cnn-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-gclip=5.0-warmstart=0-seed=1.step200000.model')
parser.add_argument('-debug', action='store_true')
parser.add_argument('-enable_tensorboard', action='store_true',
                    help='Enables tensorboard logging.')
parser.add_argument('-tensorboard_dir', type=str, default='models',
                    help='path to the directory where to save tensorboard log. If passed empty path' \
                         ' no logs are saved.')
parser.add_argument('-use_colored_lane', type=bool, default=False)
parser.add_argument('-pred_from_h', type=bool, default=False)
parser.add_argument('-detach_h', type=bool, default=False)
parser.add_argument('-random_action', type=bool, default=False)
opt = parser.parse_args()

os.system('mkdir -p ' + opt.model_dir)

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
dataloader = DataLoader(None, opt, opt.dataset, use_colored_lane=opt.use_colored_lane)




# specific to the I-80 dataset
opt.n_inputs = 4
opt.n_actions = 2
opt.height = 117
opt.width = 24
if opt.layers == 3:
    opt.h_height = 14
    opt.h_width = 3
elif opt.layers == 4:
    opt.h_height = 7
    opt.h_width = 1
opt.hidden_size = opt.nfeature*opt.h_height*opt.h_width

model = torch.load(opt.model_dir + opt.mfile)
cost = models.CostPredictor(opt).cuda()
model = model['model']
model.opt.detach_h = opt.detach_h
model.intype('gpu')
optimizer = optim.Adam(cost.parameters(), opt.lrt)
opt.model_file = opt.model_dir + opt.mfile + '.cost'
print(f'[will save as: {opt.model_file}]')

# Load normalisation stats
stats = torch.load('traffic-data/state-action-cost/data_i80_v0/data_stats.pth')
model.stats = stats  # used by planning.py/compute_uncertainty_batch

def train(nbatches, npred):
    model.train()
    total_loss = 0
    if model.opt.use_colored_lane:
        n_channels = 4
    else:
        n_channels = 3
    for i in range(nbatches):
        optimizer.zero_grad()
        inputs, actions, targets, _, car_sizes = dataloader.get_batch_fm('train', npred)
        #pdb.set_trace()
        if opt.random_action:
            actions = torch.normal(model.stats['a_mean'], model.stats['a_std'], size=actions.size())
        pred, _ = model(inputs[: -1], actions, targets, z_dropout=0)
        if model.opt.output_h and opt.pred_from_h:
            pred_cost = cost(pred[0].view(opt.batch_size*opt.npred, 1, n_channels, opt.height, opt.width), pred[1].view(opt.batch_size*opt.npred, 1, 4), hidden=pred[2].view(opt.batch_size*opt.npred, 1, -1))
        else:
            pred_cost = cost(pred[0].view(opt.batch_size*opt.npred, 1, n_channels, opt.height, opt.width), pred[1].view(opt.batch_size*opt.npred, 1, 4))
        pred_images = pred[0]
        pred_states = pred[1]
        proximity_cost, _ = utils.proximity_cost(pred_images[:, :, :n_channels].contiguous(), pred_states.data,
                                                 car_sizes,
                                                 green_channel=3 if model.opt.use_colored_lane else 1,
                                                 unnormalize=True, s_mean=model.stats['s_mean'],
                                                 s_std=model.stats['s_std'])
        if model.opt.use_colored_lane:
            orientation_cost, confidence_cost = utils.orientation_and_confidence_cost(
                                                pred_images[:, :, :n_channels].contiguous(), pred_states.data, car_size=car_sizes,
                                                unnormalize=True, s_mean=model.stats['s_mean'],
                                                s_std=model.stats['s_std'], pad=1)
            loss = F.binary_cross_entropy(pred_cost.view(opt.batch_size, opt.npred, 3),
                                          torch.stack([proximity_cost, orientation_cost, confidence_cost], dim=-1).detach())
        else:
            lane_cost, prox_map_l = utils.lane_cost(pred_images[:, :, :n_channels].contiguous(), car_sizes)
            loss = F.binary_cross_entropy(pred_cost.view(opt.batch_size, opt.npred, 2),
                                          torch.stack([proximity_cost, lane_cost], dim=-1).detach())

        if not math.isnan(loss.item()):
            loss.backward(retain_graph=False)
            if not math.isnan(utils.grad_norm(model).item()):
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer.step()
            total_loss += loss.item()
        del inputs, actions, targets

    total_loss /= nbatches
    return total_loss

def test(nbatches, npred):
    model.train()
    total_loss = 0
    if model.opt.use_colored_lane:
        n_channels = 4
    else:
        n_channels = 3
    for i in range(nbatches):
        inputs, actions, targets, _, car_sizes = dataloader.get_batch_fm('valid', npred)
        pred, _ = model(inputs[: -1], actions, targets, z_dropout=0)
        if model.opt.output_h and opt.pred_from_h:
            pred_cost = cost(pred[0].view(opt.batch_size*opt.npred, 1, n_channels, opt.height, opt.width), pred[1].view(opt.batch_size*opt.npred, 1, 4), hidden=pred[2].view(opt.batch_size*opt.npred, 1, -1))
        else:
            pred_cost = cost(pred[0].view(opt.batch_size * opt.npred, 1, n_channels, opt.height, opt.width),
                             pred[1].view(opt.batch_size * opt.npred, 1, 4))
        pred_images = pred[0]
        pred_states = pred[1]
        proximity_cost, _ = utils.proximity_cost(pred_images[:, :, :n_channels].contiguous(), pred_states.data,
                                                 car_sizes,
                                                 green_channel=3 if model.opt.use_colored_lane else 1,
                                                 unnormalize=True, s_mean=model.stats['s_mean'],
                                                 s_std=model.stats['s_std'])
        if model.opt.use_colored_lane:
            orientation_cost, confidence_cost = utils.orientation_and_confidence_cost(
                                                pred_images[:, :, :n_channels].contiguous(), pred_states.data, car_size=car_sizes,
                                                unnormalize=True, s_mean=model.stats['s_mean'],
                                                s_std=model.stats['s_std'], pad=1)
            loss = F.binary_cross_entropy(pred_cost.view(opt.batch_size, opt.npred, 3),
                                          torch.cat([proximity_cost, orientation_cost, confidence_cost], dim=-1))
        else:
            lane_cost, prox_map_l = utils.lane_cost(pred_images[:, :, :n_channels].contiguous(), car_sizes)
            loss = F.binary_cross_entropy(pred_cost.view(opt.batch_size, opt.npred, 2),
                                          torch.cat([proximity_cost, lane_cost], dim=-1))
        if not math.isnan(loss.item()):
            total_loss += loss.item()
        del inputs, actions, targets

    total_loss /= nbatches
    return total_loss

writer = utils.create_tensorboard_writer(opt)


print('[training]')
n_iter = 0
for i in range(200):
    t0 = time.time()
    train_loss = train(opt.epoch_size, opt.npred)
    valid_loss = test(int(opt.epoch_size / 2), opt.npred)
    n_iter += opt.epoch_size
    model.intype('cpu')
    torch.save({'model': cost,
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter}, opt.model_file + '.model')
    if (n_iter/opt.epoch_size) % 10 == 0:
        torch.save({'model': cost,
                    'optimizer': optimizer.state_dict(),
                    'n_iter': n_iter}, opt.model_file + f'.step{n_iter}.model')
        torch.save(model, opt.model_file + f'.step{n_iter}.model')
    model.intype('gpu')
    if writer is not None:
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/valid', valid_loss, i)
    log_string = f'step {n_iter} | train: {train_loss} | valid: {valid_loss}' 
    print(log_string)
    utils.log(opt.model_file + '.log', log_string)

if writer is not None:
    writer.close()
