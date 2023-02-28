import gc
import time

import torch
import wandb

from .core import get_optimizer, get_scheduler, train_fn, valid_fn, valid_fn_patient
from .utils import set_seed, make_dist_plot, class2dict
from .data import prepare_loaders, prepare_loaders_fulltrain
from .models import get_model
from .losses import get_loss_func
from .metric import get_df_scores, get_scores_attn, get_scores


def train_loop(CFG, folds, fold, LOGGER):

    whole_time_start = time.time()
    if CFG.finetune_change_seed:
        CFG.seed *= 2
    set_seed(CFG.seed)

    LOGGER.info(f"========== Fold: {fold} training ==========")

    train_loader, valid_loader, valid_folds = prepare_loaders(CFG, folds, fold)

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================
    model = get_model(CFG)
    if CFG.finetune and CFG.finetune_fold == fold:
        loaded_check = torch.load(CFG.finetune_path, map_location=torch.device('cpu'))
        model.to(torch.device('cpu'))
        model.load_state_dict(dict([(n, p) for n, p in loaded_check['model'].items() if 'proj_head' not in n]),
                              strict=False)

        best_score = loaded_check['params']['score']
        if 'raw_score' in loaded_check.keys():
            best_raw_score = loaded_check['raw_score']
        else:
            best_raw_score = 0
        print(f'Loaded from checkpoint')
        print(f'Best score set to: {best_score:.5f}')
    else:
        best_score = 0
        best_raw_score = 0
    model.to(CFG.device)

    optimizer = get_optimizer(model, CFG)

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    if CFG.finetune and CFG.finetune_sched_opt:
        #loaded_opt_sched = torch.load(CFG.finetune_sched_opt)
        optimizer.load_state_dict(loaded_check['optimizer'])
        scheduler.load_state_dict(loaded_check['scheduler'])

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    _global_step = 0

    save_path = CFG.save_path + f"{CFG.model}_fold{fold}_best.pth"
    last_path = CFG.save_path + f"{CFG.model}_fold{fold}_last.pth"
    last_val_path = CFG.save_path + f"{CFG.model}_fold{fold}_last_val_rest.pth"
    save_path_raw = CFG.save_path + f"{CFG.model}_fold{fold}_best_raw.pth"

    if CFG.finetune:
        torch.save({'model': model.state_dict(),
                    'prediction': loaded_check['prediction'],
                    'params': loaded_check['params'],
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, last_path)

        torch.save({'model': model.state_dict(),
                    'prediction': loaded_check['prediction'],
                    'params': loaded_check['params'],
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, save_path)

        torch.save({'model': model.state_dict(),
                    'prediction': loaded_check['prediction'],
                    'params': loaded_check['params'],
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, last_val_path)

    already_deleted_ext = False

    CFG.iter4eval = int(len(train_loader) // CFG.iter4eval)

    for epoch in range(1, CFG.epochs+1):

        if CFG.change_pos_wgt and epoch > 1:
            CFG.pos_wgt += 0.5
            criterion = get_loss_func(CFG)

        if CFG.finetune:
            if epoch < CFG.finetune_epoch:
                continue

        if CFG.use_external and epoch > CFG.dnt_use_ext_after and not already_deleted_ext:
            print(f'Data size before: {len(train_loader.dataset)}')
            train_loader.dataset.data = train_loader.dataset.data[train_loader.dataset.data.fold != 999].reset_index(drop=True)
            print(f'Data size after: {len(train_loader.dataset)}')
            print(f'CREATED NEW LOADERS WITHOUT EXTERNAL DATA')
            already_deleted_ext = True

        if epoch == CFG.epochs-5:  # 4
            CFG.iter4eval = int(len(train_loader) // 6)
        elif epoch == CFG.epochs-4:  # 5
            CFG.iter4eval = int(len(train_loader) // 6)
            CFG.rest_thr_ -= 0.005
            if CFG.change_aug:
                train_loader.dataset.use_aug_prob += 0.05  # 0.9
                print(f'Epoch {epoch}. Augmentations changed to {train_loader.dataset.use_aug_prob}.')

        elif epoch == CFG.epochs-3:  # 7
            CFG.iter4eval = int(len(train_loader) // 4)  # 1035
            CFG.rest_thr_ -= 0.005
            if CFG.change_aug:
                train_loader.dataset.use_aug_prob += 0.05  # 0.95
                print(f'Epoch {epoch}. Augmentations changed to {train_loader.dataset.use_aug_prob}.')
        elif epoch == CFG.epochs-1:  # 8
            if CFG.change_aug:
                train_loader.dataset.use_aug_prob += 0.05  # 0.95
                print(f'Epoch {epoch}. Augmentations changed to {train_loader.dataset.use_aug_prob}.')
            CFG.iter4eval = int(len(train_loader) // 4)  # 1035
            CFG.rest_thr_ -= 0.005
        elif epoch == CFG.epochs:  # 10
            CFG.iter4eval = int(len(train_loader) // 4)  # 1553
            CFG.rest_thr_ -= 0.005

        if 'stripes' in CFG.dataset:
            train_loader.dataset.epoch = epoch
        start_time = time.time()
        print(f'Epoch {epoch}/{CFG.epochs} | Fold {fold}')

        # train
        avg_loss, best_score, best_raw_score = train_fn(cfg=CFG, fold=fold, train_loader=train_loader,
                                                        valid_loader=valid_loader, model=model, criterion=criterion,
                                                        optimizer=optimizer, scheduler=scheduler, device=CFG.device,
                                                        epoch=epoch, LOGGER=LOGGER, best_score=best_score,
                                                        _global_step=_global_step, _it4eval=CFG.iter4eval,
                                                        save_path=save_path,  lastval_path=last_val_path,
                                                        val_folds=valid_folds, best_raw_score=best_raw_score,
                                                        save_path_raw=save_path_raw)

        _global_step += len(train_loader)
        # eval

        avg_val_loss, predictions = valid_fn(cfg=CFG, valid_loader=valid_loader, model=model,
                                             epoch=epoch, criterion=criterion, device=CFG.device)
        prediction = predictions[0].copy()

        if CFG.deep_supervision_out:
            print(f'SCORES FOR DEEP SUPERVISION:')
            valid_folds['prediction'] = predictions[0] * 0.7 + 0.3 * predictions[1]
            get_df_scores(valid_folds)
            print()
        # scoring
        valid_folds['prediction'] = prediction
        if CFG.deep_supervision_out:
            valid_folds['dsv_prediction'] = predictions[1].copy()
        if 'atten' in CFG.dataset:
            val_score, params, raw_score = get_scores_attn(prediction, valid_folds['cancer'].values)
        else:
            val_score, params, raw_score = get_df_scores(valid_folds)

        if CFG.VALID_SINGLE:
            print('='*50)
            print(f'SINGLE METRICS:')
            val_score, params, raw_score = get_scores(predictions[0].copy(), valid_folds['cancer'].values, agg='Single',
                                                      best_cur=0)
            print(f'Best params: {params}')
            print(f'Best RAW score: {raw_score}')
            if CFG.deep_supervision_out:
                print(f'SINGLE FOR DSV:')
                val_score_ds, params_ds, raw_score_ds = get_scores(predictions[1].copy(), valid_folds['cancer'].values,
                                                                   agg='Single', best_cur=0)
                print(f'Best params: {params_ds}')
                print(f'Best RAW score: {raw_score_ds}')
                print('=' * 50)

        if CFG.dist_plot_valid:
            print(f'PLOT FOR BASE')
            make_dist_plot(valid_folds)
            if CFG.deep_supervision_out:
                print(f'PLOT FOR DSV')
                hz_folds = valid_folds.copy()
                hz_folds['prediction'] = hz_folds['dsv_prediction']
                make_dist_plot(hz_folds)
                del hz_folds

        torch.save({'model': model.state_dict(),
                    'prediction': prediction,
                    'params': params,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, last_path)
                    
        #torch.save({'scheduler': scheduler.state_dict(),
        #            'optimizer': optimizer.state_dict()}, CFG.save_path + f"last_sched_opt_f{fold}_ep{epoch}.pt")
        
        valid_folds.to_csv(last_path.split('.pth')[0] + '_df.csv', index=False)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch} - avg_train_loss: {avg_loss:.5f}  '
            f'avg_val_loss: {avg_val_loss:.5f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch} - Score: {val_score:.5f} | RAW score: {raw_score:.5f}')

        if raw_score >= best_raw_score:
            LOGGER.info(f'Best RAW Score Updated {best_raw_score:0.5f} -->> {raw_score:0.5f} | Model Saved')
            LOGGER.info(f'Best params: {params}')
            best_raw_score = raw_score
            valid_folds.to_csv(save_path_raw.split('.pth')[0] + '_df.csv', index=False)
            torch.save({'model': model.state_dict(),
                        'prediction': prediction,
                        'params': params,
                        'raw_score': raw_score,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()}, save_path_raw)

        if val_score >= best_score:
            LOGGER.info(f'||||||||| Best Score Updated {best_score:0.5f} -->> {val_score:0.5f} | Model Saved |||||||||')
            LOGGER.info(f'Best params: {params}')
            best_score = val_score
            valid_folds.to_csv(save_path.split('.pth')[0] + '_df.csv', index=False)
            torch.save({'model': model.state_dict(),
                        'prediction': prediction,
                        'params': params,
                        'raw_score': raw_score,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()}, save_path)
        else:
            LOGGER.info(f'Score NOT updated. Current best: {best_score:0.4f}')
            if CFG.use_restart and best_score - val_score > CFG.rest_thr_ and epoch > CFG.rest_epoch:
                loaded_check = torch.load(save_path, #last_val_path,  # not best
                                          map_location=torch.device('cpu'))
                model.to(torch.device('cpu'))
                model.load_state_dict(loaded_check['model'])
                model.to(CFG.device)
                LOGGER.info('Loaded model FROM LAST VALIDATION')

        torch.save({'model': model.state_dict(),
                    'prediction': prediction,
                    'params': params,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, last_val_path)

        if CFG.save_for_future and epoch == CFG.save_future_epoch:
            torch.save({'model': model.state_dict(),
                        'prediction': prediction,
                        'params': params,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       CFG.save_path + f"{CFG.model}_fold{fold}_epoch{epoch}_FORFINETUNE.pth")

        if CFG.wandb:
            try:
                wandb.log({f"[fold{fold}] val_score": val_score,
                           f"[fold{fold}] raw_score": raw_score,
                           f"[fold{fold}] avg_train_loss": avg_loss,
                           f"[fold{fold}] avg_val_loss": avg_val_loss})
            except:
                pass

    # Load and re save best valid
    loadeed_check = torch.load(save_path, map_location=torch.device('cpu'))
    final_v_path = CFG.save_path + f"{CFG.model}_fold{fold}_final_loop_best_{best_score:0.4f}.pth"
    torch.save(loadeed_check, final_v_path)
    if not CFG.patient_wise:
        valid_folds['prediction'] = loadeed_check['prediction']
    else:
        valid_folds[['prediction_mean', 'prediction_median', 'prediction_max']] = loadeed_check['prediction']
    valid_folds.to_csv(final_v_path.split('.pth')[0] + '_df.csv', index=False)

    loadeed_check = torch.load(save_path_raw, map_location=torch.device('cpu'))
    final_vraw_path = CFG.save_path + f"{CFG.model}_fold{fold}_final_loop_best_RAW_{best_raw_score:0.4f}.pth"
    torch.save(loadeed_check, final_vraw_path)
    # END OF TRAINING

    LOGGER.info(f'FOLD {fold} TRAINING FINISHED. BEST SCORE: {best_score:0.4f} | BEST RAW SCORE: {best_raw_score:.4f}',
                f'SAVED HERE: {final_v_path}')

    torch.cuda.empty_cache()
    gc.collect()

    time_elapsed = time.time() - whole_time_start
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))


def train_loop_full_set(CFG, folds, LOGGER):

    CFG.use_restart = False
    whole_time_start = time.time()
    if CFG.finetune:
        CFG.seed *= 2
    set_seed(CFG.seed)

    LOGGER.info(f"========== Full set training ==========")

    train_loader, valid_folds = prepare_loaders_fulltrain(CFG, folds)

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================
    model = get_model(CFG)
    model.to(CFG.device)

    optimizer = get_optimizer(model, CFG)

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    _global_step = 0
    best_score = 0

    save_path = CFG.save_path + f"{CFG.model}_full_train_best.pth"
    last_path = CFG.save_path + f"{CFG.model}_full_train_last"

    already_deleted_ext = False

    for epoch in range(1, CFG.epochs+1):

        if CFG.use_external and epoch > CFG.dnt_use_ext_after and not already_deleted_ext:
            print(f'Data size before: {len(train_loader.dataset)}')
            train_loader.dataset.data = train_loader.dataset.data[train_loader.dataset.data.fold != 999].reset_index(drop=True)
            print(f'Data size after: {len(train_loader.dataset)}')
            print(f'CREATED NEW LOADERS WITHOUT EXTERNAL DATA')
            already_deleted_ext = True

        if CFG.change_aug:
            if epoch == 5:  # 6
                train_loader.dataset.use_aug_prob = 0.8  # 0.9
                # Randzoom
                train_loader.dataset.transforms_0.transforms[2].prob = 0.35
                train_loader.dataset.transforms_1.transforms[2].prob = 0.35

                # Rotate
                train_loader.dataset.transforms_0.transforms[3].prob = 0.35
                train_loader.dataset.transforms_1.transforms[3].prob = 0.35

                # OneOf Elastic Affine
                train_loader.dataset.transforms_0.transforms[4].transforms[0].prob = 0.35
                train_loader.dataset.transforms_0.transforms[4].transforms[1].prob = 0.35
                train_loader.dataset.transforms_1.transforms[4].transforms[0].prob = 0.35
                train_loader.dataset.transforms_1.transforms[4].transforms[1].prob = 0.35

                # Contrast
                train_loader.dataset.transforms_0.transforms[5].prob = 0.4
                train_loader.dataset.transforms_1.transforms[5].prob = 0.4

                # Coarse Dropout
                if 'stripes' not in CFG.dataset:
                    train_loader.dataset.transforms_0.transforms[6].prob = 0.35
                    train_loader.dataset.transforms_1.transforms[6].prob = 0.35
                else:
                    train_loader.dataset.stripe_prob = 0.65

                print(f'Epoch {epoch}. Augmentations changed.')

            elif epoch == 7:  # 8
                train_loader.dataset.use_aug_prob = 0.75  # 0.95
                # Randzoom
                train_loader.dataset.transforms_0.transforms[2].prob = 0.5
                train_loader.dataset.transforms_1.transforms[2].prob = 0.5

                # Rotate
                train_loader.dataset.transforms_0.transforms[3].prob = 0.5
                train_loader.dataset.transforms_1.transforms[3].prob = 0.5

                # OneOf Elastic Affine
                train_loader.dataset.transforms_0.transforms[4].transforms[0].prob = 0.5
                train_loader.dataset.transforms_0.transforms[4].transforms[1].prob = 0.5
                train_loader.dataset.transforms_1.transforms[4].transforms[0].prob = 0.5
                train_loader.dataset.transforms_1.transforms[4].transforms[1].prob = 0.5

                # Contrast
                train_loader.dataset.transforms_0.transforms[5].prob = 0.6
                train_loader.dataset.transforms_1.transforms[5].prob = 0.6

                # Coarse Dropout
                if 'stripes' not in CFG.dataset:
                    train_loader.dataset.transforms_0[6].prob = 0.35
                    train_loader.dataset.transforms_1[6].prob = 0.35
                else:
                    train_loader.dataset.stripe_prob = 0.75
                print(f'Epoch {epoch}. Augmentations changed.')

        start_time = time.time()
        print(f'Epoch {epoch}/{CFG.epochs}')

        # train

        avg_loss, best_score, best_raw_score = train_fn(cfg=CFG, fold=0, train_loader=train_loader,
                                                        valid_loader=None, model=model, criterion=criterion,
                                                        optimizer=optimizer, scheduler=scheduler, device=CFG.device,
                                                        epoch=epoch, LOGGER=LOGGER, best_score=best_score,
                                                        _global_step=_global_step, _it4eval=10e9,
                                                        save_path=save_path, lastval_path=None,
                                                        val_folds=valid_folds, best_raw_score=0,
                                                        save_path_raw=None)

        torch.save({'model': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()}, last_path + f'_epoch{epoch}.pth')

        _global_step += len(train_loader)
        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch} - avg_train_loss: {avg_loss:.5f}  '
            f'time: {elapsed:.0f}s')
        if CFG.wandb:
            try:
                wandb.log({f"[fold0] avg_train_loss": avg_loss})
            except:
                pass

    # Load and re save best valid
    final_v_path = CFG.save_path + f"{CFG.model}_full_train_final_loop.pth"
    torch.save({'model': model.state_dict()}, final_v_path)
    valid_folds.to_csv(final_v_path.split('.pth')[0] + '_df.csv', index=False)
    # END OF TRAINING

    LOGGER.info(f'FULL TRAIN TRAINING FINISHED ',
                f'SAVED HERE: {final_v_path}')

    torch.cuda.empty_cache()
    gc.collect()

    time_elapsed = time.time() - whole_time_start
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
