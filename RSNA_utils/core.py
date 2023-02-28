import gc
from collections import defaultdict

from tqdm.auto import tqdm
import torch
import wandb
import numpy as np
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from .utils import AverageMeter, make_dist_plot, class2dict
from .metric import get_df_scores, get_scores_attn, get_scores


def get_optimizer(model, cfg):

    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps, betas=cfg.betas,
                                weight_decay=cfg.weight_decay)
    else:
        raise ValueError('Error in "get_optimizer" function:',
                         f'Wrong optimizer name. Choose one from ["Adam", "AdamW"] ')

    return optimizer


def get_scheduler(cfg, scheduler_name, optimizer, num_train_steps, cycles):
    if scheduler_name == 'linear':
        scheduler = get_linear_schedule_with_warmup(
                                                    optimizer, num_warmup_steps=cfg.num_warmup_steps,
                                                    num_training_steps=num_train_steps)
    elif scheduler_name == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
                                                    optimizer, num_warmup_steps=cfg.num_warmup_steps,
                                                    num_training_steps=num_train_steps, num_cycles=cycles)

    elif scheduler_name == 'cosine_restart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=cfg.min_lr,
                                                                   T_0=int(num_train_steps // cycles), T_mult=1)

    elif scheduler_name == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=int(num_train_steps*cfg.onecycle_m),
                                                  pct_start=cfg.onecycle_start)

    elif scheduler_name == 'simple_cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(num_train_steps // cycles),
                                                               eta_min=cfg.min_lr, last_epoch=-1)

    elif scheduler_name == 'cosine_warmup_ext':
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=num_train_steps // 2.5,
                                                  cycle_mult=1.75,
                                                  max_lr=cfg.lr,
                                                  min_lr=cfg.min_lr,
                                                  warmup_steps=cfg.num_warmup_steps,
                                                  gamma=0.7)


    else:
        raise ValueError('Error in "get_scheduler" function:',
                         f'Wrong scheduler name. Choose one from ["linear", "cosine", "cosine_restart", "onecycle" ]')

    return scheduler


def train_fn(cfg, fold, train_loader, valid_loader, model, criterion,
             optimizer, scheduler, device, epoch, LOGGER, best_score,
             _global_step, _it4eval, save_path, lastval_path, val_folds, best_raw_score, save_path_raw):

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Ep.{epoch} Train ')

    for step, batch in pbar:
        _global_step += 1

        if 'MIP' in cfg.model:
            img, label, ext, lens = batch
            lens = lens.tolist()
        elif 'atten' in cfg.dataset:
            img, lengths, label = batch
        else:
            img, label, ext = batch

        #print(img.shape)

        img = img.to(device)
        label = label.float().unsqueeze(1).to(device)

        batch_size = img.size(0)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            if cfg.deep_supervision:
                y_pred, sv_pred = model(img)
            elif 'MIP' in cfg.model:
                y_pred = model(img, lens)
            elif 'atten' in cfg.dataset:
                y_pred = model(img, lengths)
            else:
                y_pred = model(img)

            if cfg.loss == 'LMF_EXT':
                ext = ext.float().unsqueeze(1).to(device)
                loss = criterion(y_pred, label, ext)
            else:
                if cfg.deep_supervision:
                    sv_loss = None
                    for sv in sv_pred:
                        if sv_loss is None:
                            sv_loss = criterion(sv, label)
                        else:
                            sv_loss += criterion(sv, label)

                    sv_loss /= len(model.sup_inds)
                    loss = criterion(y_pred, label) * 0.6 + 0.4 * sv_loss
                else:
                    loss = criterion(y_pred, label)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()
        wl = torch.where(label == 1, cfg.pos_wgt, 1).flatten().float().mean().item()
        losses.update(loss.item() / wl, batch_size)

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                         lr=f'{current_lr:0.8f}',
                         gpu_mem=f'{mem:0.2f} GB',
                         global_step=f'{_global_step}')

        if cfg.wandb:
            try:
                wandb.log({f"[fold{fold}] loss": losses.val,
                           f"[fold{fold}] lr": current_lr})
            except:
                pass

        if cfg.use_restart:
            if step > 100 and _global_step % _it4eval == 0 and \
                    len(train_loader) - step > 100 and epoch > cfg.rest_epoch:
                print(f'Eval on step: {_global_step}')

                avg_val_loss, predictions = valid_fn(cfg=cfg, valid_loader=valid_loader, model=model,
                                                     epoch=epoch, criterion=criterion, device=cfg.device)

                prediction = predictions[0].copy()

                if cfg.deep_supervision_out:
                    print(f'SCORES FOR DEEP SUPERVISION:')
                    val_folds['prediction'] = predictions[0] * 0.7 + 0.3 * predictions[1]
                    get_df_scores(val_folds)
                    print()
                # scoring
                if cfg.deep_supervision_out:
                    val_folds['dsv_prediction'] = predictions[1].copy()
                val_folds['prediction'] = prediction
                if 'atten' in cfg.dataset:
                    val_score, params, raw_score = get_scores_attn(prediction, val_folds['cancer'].values)
                else:
                    val_score, params, raw_score = get_df_scores(val_folds)

                if cfg.VALID_SINGLE:
                    print('='*50)
                    print(f'SINGLE METRICS:')
                    val_score, params, raw_score = get_scores(predictions[0].copy(), val_folds['cancer'].values,
                                                              agg='Single',
                                                              best_cur=0)
                    print(f'Best params: {params}')
                    print(f'Best RAW score: {raw_score}')
                    if cfg.deep_supervision_out:
                        print(f'SINGLE FOR DSV:')
                        val_score_ds, params_ds, raw_score_ds = get_scores(predictions[1].copy(),
                                                                           val_folds['cancer'].values,
                                                                           agg='Single', best_cur=0)
                        print(f'Best params: {params_ds}')
                        print(f'Best RAW score: {raw_score_ds}')
                        print('=' * 50)

                if cfg.dist_plot_valid:
                    print(f'PLOT FOR BASE')
                    make_dist_plot(val_folds)
                    if cfg.deep_supervision_out:
                        print(f'PLOT FOR DSV')
                        hz_folds = val_folds.copy()
                        hz_folds['prediction'] = hz_folds['dsv_prediction']
                        make_dist_plot(hz_folds)
                        del hz_folds

                LOGGER.info(
                    f'avg_val_loss: {avg_val_loss:.5f} '
                    f'val_score: {val_score:.5f} '
                    f'raw_score: {raw_score:.5f}')

                if raw_score >= best_raw_score:
                    LOGGER.info(f'Best RAW Score Updated {best_raw_score:0.5f} -->> {raw_score:0.5f} | Model Saved')
                    LOGGER.info(f'Best params: {params}')
                    best_raw_score = raw_score
                    val_folds.to_csv(save_path_raw.split('.pth')[0] + '_df.csv', index=False)
                    torch.save({'model': model.state_dict(),
                                'prediction': prediction,
                                'params': params,
                                'raw_score': raw_score,
                                'scheduler': scheduler.state_dict(),
                                'optimizer': optimizer.state_dict()}, save_path_raw)

                if val_score >= best_score:
                    LOGGER.info(f'||||| Best Score Updated {best_score:0.5f} -->> {val_score:0.5f} | Model Saved |||||')
                    LOGGER.info(f'Best params: {params}')
                    best_score = val_score

                    val_folds.to_csv(save_path.split('.pth')[0] + '_df.csv', index=False)
                    torch.save({'model': model.state_dict(),
                                'prediction': prediction,
                                'params': params,
                                'raw_score': raw_score,
                                'scheduler': scheduler.state_dict(),
                                'optimizer': optimizer.state_dict()}, save_path)

                else:
                    LOGGER.info(f'Score NOT updated. Current best: {best_score:0.4f}')
                    if cfg.use_restart and best_score - val_score > cfg.rest_thr_:
                        loaded_check = torch.load(save_path, #lastval_path,  # not best
                                                  map_location=torch.device('cpu'))
                        model.to(torch.device('cpu'))
                        model.load_state_dict(loaded_check['model'])
                        model.to(device)
                        LOGGER.info('Loaded model FROM LAST VALIDATION')

                torch.save({'model': model.state_dict(),
                            'prediction': prediction,
                            'params': params,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict()}, lastval_path)

                if cfg.wandb:
                    try:
                        wandb.log({
                                   f"[fold{fold}] avg_train_loss": losses.avg,
                                   f"[fold{fold}] avg_val_loss": avg_val_loss,
                                   f"[fold{fold}] val_score": val_score,
                                   f"[fold{fold}] raw_score": raw_score})
                    except:
                        pass

                model.train()
                
    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, best_score, best_raw_score


@torch.no_grad()
def valid_fn(cfg, valid_loader, model, epoch, criterion, device):

    losses = AverageMeter()
    model.eval()
    prediction = np.array([], dtype=np.float32)
    if cfg.deep_supervision_out:
        prediction_sv = np.array([], dtype=np.float32)

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Ep.{epoch} Eval ')

    for step, batch in pbar:

        if 'MIP' in cfg.model:
            img, label, ext, lens = batch
            lens = lens.tolist()
        elif 'atten' in cfg.dataset:
            img, lengths, label = batch
        else:
            img, label, ext = batch

        img = img.to(device)
        label = label.float().unsqueeze(1).to(device)

        batch_size = img.size(0)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            if cfg.deep_supervision:
                y_pred, sv_pred = model(img)
            elif 'MIP' in cfg.model:
                y_pred = model(img, lens)
            elif 'atten' in cfg.dataset:
                y_pred = model(img, lengths)
            else:
                y_pred = model(img)

            if cfg.loss == 'LMF_EXT':
                ext = ext.float().unsqueeze(1).to(device)
                loss = criterion(y_pred, label, ext)
            else:
                if cfg.deep_supervision:
                    sv_loss = None
                    for sv in sv_pred:
                        if sv_loss is None:
                            sv_loss = criterion(sv, label)
                        else:
                            sv_loss += criterion(sv, label)

                    sv_loss /= len(model.sup_inds)
                    loss = criterion(y_pred, label) * 0.6 + 0.4 * sv_loss
                else:
                    loss = criterion(y_pred, label)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        # losses.update(loss.item(), batch_size)
        wl = torch.where(label == 1, cfg.pos_wgt, 1).flatten().float().mean().item()
        losses.update(loss.item() / wl, batch_size)
        
        out = torch.sigmoid(y_pred).detach().cpu().flatten().numpy()
        if cfg.deep_supervision_out:
            sv = torch.stack([torch.sigmoid(x) for x in sv_pred]).mean(dim=0).detach().cpu().flatten().numpy()
            prediction_sv = np.append(prediction_sv, sv)
        prediction = np.append(prediction, out)

        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0

        pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')

    torch.cuda.empty_cache()
    gc.collect()
    if cfg.deep_supervision_out:
        return losses.avg, [prediction, prediction_sv]
    return losses.avg, [prediction]


def get_batch_inds(lengths):
    batch_inds = []
    for i in range(len(lengths)):
        ln = int(lengths[i])
        done = sum(lengths[:i])
        ind_ar = []
        for ind in range(0, ln, 1):
            ind_ar.append((min(ind + done, ln + done - 3), min(ln + done, done + ind + 3)))
        ind_ar = np.unique(np.array(ind_ar), axis=0)
        for br in range(len(ind_ar)):
            batch_inds.append((i, ind_ar[br]))
    return np.array(batch_inds)


@torch.no_grad()
def valid_fn_patient(cfg, valid_loader, model, epoch, criterion, device):
    losses = AverageMeter()
    model.eval()
    outs = np.array([])
    lnz = np.array([])
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Ep.{epoch} Eval ')

    for step, batch in pbar:

        img, lengths, label, difficult = batch
        lnz = np.concatenate([lnz, lengths.numpy()])
        batch_inds = get_batch_inds(lengths)
        for i in range(0, len(batch_inds), cfg.valid_bs):
            start_idx = i
            stop_idx = min(i + cfg.valid_bs, len(batch_inds))

            cur_batch, cur_label, cur_dif = [], [], []
            for item in batch_inds[start_idx:stop_idx]:
                cur_batch.append(img[item[1][0]:item[1][1]])
                cur_label.append(label[item[0]])
                cur_dif.append(difficult[item[0]])

            cur_batch = torch.stack(cur_batch).to(device)
            
            cur_label = torch.tensor(cur_label).float().unsqueeze(1).to(device)
            batch_size = cur_batch.size(0)

            with torch.no_grad():
                y_pred = model(cur_batch)
                if cfg.loss == 'diff_bce':
                    cur_dif = torch.tensor(cur_dif).float().unsqueeze(1).to(device)
                    loss = criterion(y_pred, cur_label, cur_dif)
                else:
                    loss = criterion(y_pred, cur_label)

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

            losses.update(loss.item(), batch_size)
            out = torch.sigmoid(y_pred).detach().cpu().flatten().numpy()
            outs = np.concatenate([outs, out])

            mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0

            pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                             gpu_mem=f'{mem:0.2f} GB')

    torch.cuda.empty_cache()
    gc.collect()

    scores_mb = defaultdict(list)
    batch_places = get_batch_inds(lnz)[:, 0]
    for in_ in range(len(outs)):
        scores_mb[batch_places[in_]].append(outs[in_])
    mean_preds = np.array([np.mean(v) for v in scores_mb.values()])
    median_preds = np.array([np.median(v) for v in scores_mb.values()])
    max_preds = np.array([np.max(v) for v in scores_mb.values()])

    prediction = np.stack([mean_preds, median_preds, max_preds],axis=1)

    return losses.avg, prediction
