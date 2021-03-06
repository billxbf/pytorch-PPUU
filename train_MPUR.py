import math
from collections import OrderedDict

import numpy
import os
import ipdb
import random
import torch
import torch.optim as optim
from os import path

import planning
import utils
from dataloader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a policy / controller
#################################################

opt = utils.parse_command_line()

# Create file_name
opt.model_file = path.join(opt.model_dir, 'policy_networks', 'MPUR-' + opt.policy)
utils.build_model_file_name(opt)

os.system('mkdir -p ' + path.join(opt.model_dir, 'policy_networks'))

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Define default device
opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_cuda else 'cpu')
if torch.cuda.is_available() and opt.no_cuda:
    print('WARNING: You have a CUDA device, so you should probably run without -no_cuda')

# load the model
model_path = path.join(opt.model_dir, opt.mfile)
if path.exists(model_path):
    model = torch.load(model_path)
elif path.exists(opt.mfile):
    model = torch.load(opt.mfile)
else:
    raise runtime_error(f'couldn\'t find file {opt.mfile}')

if type(model) is dict: model = model['model']
if not hasattr(model.encoder, 'n_channels'):
    model.encoder.n_channels = 3
model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch
if opt.value_model != '':
    value_function = torch.load(path.join(opt.model_dir, 'value_functions', opt.value_model)).to(opt.device)
    model.value_function = value_function

# Create policy
if os.path.isfile(opt.model_file + '.model'):
    filename = opt.model_file + '.model'
    print(f'[loading previous checkpoint: {filename}]')
    checkpoint = torch.load(filename)
    model = checkpoint['model']
    model.cuda()
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)
    optimizer.load_state_dict(checkpoint['optimizer'])
    n_iter = checkpoint['n_iter']
    utils.log(opt.model_file + '.log', '[resuming from checkpoint]')
else:
    model.create_policy_net(opt)
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!
    n_iter = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80_000/opt.epoch_size, gamma=0.1)

# data
opt.dataset = f"traffic-data/state-action-cost-{opt.ksize}-{opt.position_threshold}/data_i80_v0/"

# Load normalisation stats
stats = torch.load(opt.dataset+'data_stats.pth')
model.stats = stats  # used by planning.py/compute_uncertainty_batch

if 'ten' in opt.mfile:
    p_z_file = opt.model_dir + opt.mfile + '.pz'
    p_z = torch.load(p_z_file)
    model.p_z = p_z

# Send to GPU if possible
model.to(opt.device)
model.policy_net.stats_d = {}
for k, v in stats.items():
    if isinstance(v, torch.Tensor):
        model.policy_net.stats_d[k] = v.to(opt.device)

if opt.learned_cost!= 'False':
    print('[loading cost regressor]')
    cost_name=''
    value = ['1.0', '0.5', '0.1', '0.0']
    tf = ['0', '1']
    cost_name+='-random='+tf[int(opt.learned_cost[0])]
    cost_name+='-std_v='+value[int(opt.learned_cost[1])]+'-std_r='+value[int(opt.learned_cost[2])]
    cost_name+='-c_dropout='+tf[int(opt.learned_cost[3])]

    model.cost = torch.load(path.join(opt.model_dir,'cost_models', opt.mfile + '.cost'+cost_name+'.model'))['model']


dataloader = DataLoader(None, opt, opt.dataset, use_colored_lane=model.opt.use_colored_lane,
                            use_offroad_map=model.opt.use_offroad_map if hasattr(model.opt,'use_offroad_map') else False,
                        use_kinetic_model=model.opt.use_kinetic_model if hasattr(model.opt,'use_kinetic_model') else False,
                        iterate_all=model.opt.iterate_all if hasattr(model.opt,'iterate_all') else False)
model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred, pad=opt.pad, offroad_range=opt.offroad_range)
model.eval()


def start(what, nbatches, npred, track=False, pad=1, offroad_range=1.0):
    train = True if what is 'train' else False
    model.train()
    model.policy_net.train()
    n_updates, grad_norm = 0, 0
    if opt.track_grad_norm:
        a_grad_norm = [0, 0, 0]
    if opt.use_colored_lane:
        total_losses = dict(
            proximity=0,
            uncertainty=0,
            position=0,
            orientation=0,
            action=0,
            policy=0,
        )
        if opt.track_grad_norm:
            total_grads = dict(
                proximity_position_orientation=0,
                position=0,
                orientation=0,
            )
    else:
        total_losses = dict(
            proximity=0,
            uncertainty=0,
            lane=0,
            offroad=0,
            action=0,
            policy=0,
        )
        if opt.track_grad_norm:
            total_grads = dict(
                proximity_lane=0,
                lane=0,
            )
    for j in range(nbatches):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(what, npred)
        pred, actions = planning.train_policy_net_mpur(
            model, inputs, targets, car_sizes, n_models=10, lrt_z=opt.lrt_z,
            n_updates_z=opt.z_updates, infer_z=opt.infer_z, pad=pad, offroad_range=offroad_range)
        if opt.use_colored_lane:
            pred['policy'] = pred['proximity'] + \
                             opt.u_reg * pred['uncertainty'] + \
                             opt.lambda_o * pred['orientation'] + \
                             opt.lambda_a * pred['action'] + \
                             opt.lambda_l * pred['position']
        else:
            pred['policy'] = pred['proximity'] + \
                             opt.u_reg * pred['uncertainty'] + \
                             opt.lambda_l * pred['lane'] + \
                             opt.lambda_a * pred['action'] + \
                             opt.lambda_o * pred['offroad']

        if not math.isnan(pred['policy'].item()):
            if train:
                optimizer.zero_grad()
                pred['policy'].backward()  # back-propagation through time!
                grad_norm += utils.grad_norm(model.policy_net).item()
                torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), opt.grad_clip)
                optimizer.step()
            elif track:
                if opt.use_colored_lane:
                    optimizer.zero_grad()
                    pred['policy'] = opt.lambda_o * pred['orientation']
                    pred['policy'].backward()  # back-propagation through time!
                    a_grad_norm[0] += utils.a_grad_norm(model.policy_net).item()
                    optimizer.zero_grad()
                    pred['policy'] = opt.lambda_l * pred['position']
                    pred['policy'].backward()  # back-propagation through time!
                    a_grad_norm[1] += utils.a_grad_norm(model.policy_net).item()
                    optimizer.zero_grad()
                    pred['policy'] = pred['proximity'] + opt.lambda_o * pred['orientation'] + opt.lambda_l * pred['position']
                    pred['policy'].backward()  # back-propagation through time!
                    a_grad_norm[2] += utils.a_grad_norm(model.policy_net).item()
                    total_grads['orientation'] += a_grad_norm[0]
                    total_grads['position'] += a_grad_norm[1]
                    total_grads['proximity_position_orientation'] += a_grad_norm[2]
                else:
                    optimizer.zero_grad()
                    pred['policy'] = opt.lambda_l * pred['lane']
                    pred['policy'].backward()  # back-propagation through time!
                    a_grad_norm[0] += utils.a_grad_norm(model.policy_net).item()
                    optimizer.zero_grad()
                    pred['policy'] = pred['proximity'] + opt.lambda_l * pred['lane']
                    pred['policy'].backward()  # back-propagation through time!
                    a_grad_norm[1] += utils.a_grad_norm(model.policy_net).item()
                    total_grads['lane'] += a_grad_norm[0]
                    total_grads['proximity_lane'] += a_grad_norm[1]
            for loss in total_losses: total_losses[loss] += pred[loss].item()
            n_updates += 1
        else:
            print('warning, NaN')  # Oh no... Something got quite fucked up!
            ipdb.set_trace()

        if j == 0 and opt.save_movies and train:
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                state_img = pred['state_img'][b]
                state_vct = pred['state_vct'][b]
                utils.save_movie(opt.model_file + f'.mov/sampled/mov{b}', state_img, state_vct, None, actions[b])

        del inputs, actions, targets, pred

    for loss in total_losses:
        total_losses[loss] /= n_updates
    if track:
        for grad in total_grads:
            total_grads[grad] /= n_updates
    if train: print(f'[avg grad norm: {grad_norm / n_updates:.4f}]')
    return total_losses


print('[training]')
utils.log(opt.model_file + '.log', f'[job name: {opt.model_file}]')
if opt.use_colored_lane:
    losses = OrderedDict(
        p='proximity',
        c='position',
        o='orientation',
        u='uncertainty',
        a='action',
        π='policy',
    )
    if opt.track_grad_norm:
        grads = OrderedDict(
        p='proximity_position_orientation',
        c='position',
        o='orientation',
    )
else:
    losses = OrderedDict(
        p='proximity',
        l='lane',
        o='offroad',
        u='uncertainty',
        a='action',
        π='policy',
    )
    if opt.track_grad_norm:
        grads = OrderedDict(
        p='proximity_lane',
        l='lane',
    )

writer = utils.create_tensorboard_writer(opt)

for i in range(500):
    train_losses = start('train', opt.epoch_size, opt.npred, pad=opt.pad, offroad_range=opt.offroad_range)
    a_grad = []
    if opt.track_grad_norm:
        valid_losses = start('valid', opt.epoch_size // 2, opt.npred, track=True, pad=opt.pad, offroad_range=opt.offroad_range)
    else:
        with torch.no_grad():  # Torch, please please please, do not track computations :)
            valid_losses = start('valid', opt.epoch_size // 2, opt.npred, pad=opt.pad, offroad_range=opt.offroad_range)
    scheduler.step()
    if writer is not None:
        for key in train_losses:
            writer.add_scalar(f'Loss/train_{key}', train_losses[key], i)
        for key in valid_losses:
            writer.add_scalar(f'Loss/valid_{key}', valid_losses[key], i)

    n_iter += opt.epoch_size
    model.to('cpu')
    torch.save(dict(
        model=model,
        optimizer=optimizer.state_dict(),
        opt=opt,
        n_iter=n_iter,
    ), opt.model_file + '.model')
    if (n_iter / opt.epoch_size) % 10 == 0:
        torch.save(dict(
            model=model,
            optimizer=optimizer.state_dict(),
            opt=opt,
            n_iter=n_iter,
        ), opt.model_file + f'step{n_iter}.model')

    model.to(opt.device)

    log_string = f'step {n_iter} | '
    log_string += 'train: [' + ', '.join(f'{k}: {train_losses[v]:.4f}' for k, v in losses.items()) + '] | '
    log_string += 'valid: [' + ', '.join(f'{k}: {valid_losses[v]:.4f}' for k, v in losses.items()) + ']'
    if opt.track_grad_norm:
        log_string += ' | grad: [' + ', '.join(f'{k}: {a_grad[v]:.4f}' for k, v in grads.items()) + ']'
    print(str(train_losses) + '\n' + str(valid_losses))
    try:
        print(log_string)
    except Exception:
        print("Print error")

    utils.log(opt.model_file + '.log', log_string)

if writer is not None:
    writer.close()
