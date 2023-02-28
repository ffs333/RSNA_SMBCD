import os
import glob
import warnings

import wandb
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import get_logger, class2dict
from pipeline import train_loop, train_loop_full_set

from cfg import CFG
warnings.filterwarnings("ignore")


def main(CFG):
    if CFG.debug:
        CFG.wandb = False
    if CFG.wandb:
        wandb.init(project=CFG.wandb_project,
                   name=CFG.exp_name,
                   config=class2dict(CFG),
                   group=CFG.wb_group,
                   job_type="train",
                   dir=CFG.base_path)

    IMG_PATH = 'image_path/'

    train = pd.read_csv(f'{CFG.base_path}/train.csv')

    train['age'].fillna(int(train['age'].mean()), inplace=True)
    train['age'] = train['age'] / 100

    pathes = []
    for i in range(len(train)):
        sample = train.loc[i]
        path = IMG_PATH + f'/{sample["patient_id"]}/{sample["image_id"]}.png'
        pathes.append(path)
    train['path'] = pathes
    print(train.shape)

    bad_mb = [1743461841]
    # bad images

    split_cols = ['cancer', 'laterality', 'view', 'age', 'implant', 'machine_id', 'difficult_negative_case']
    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[split_cols])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    train = train[~train.image_id.isin(np.array(bad_mb))].reset_index(drop=True)

    if CFG.debug:
        train = train.sample(n=500, random_state=0).reset_index(drop=True)

    files_ = glob.glob(IMG_PATH + '/*/*.png')

    hzzz = []
    for i in range(len(train)):
        if not os.path.exists(train.loc[i, 'path']):
            hzzz.append(train.loc[i, 'path'])
            print(train.loc[i, 'path'])
            print('Label: ', train.loc[i, 'cancer'])
    train = train[~train.path.isin(hzzz)].reset_index(drop=True)

    print(f'Total files: {len(files_)} | Not Found Files: {len(hzzz)} | Rest train shape: {len(train)}')


    if CFG.use_external:
        external_df = pd.read_csv('path_to_external/external_train.csv')
        external_df['laterality'] = external_df['laterality'].apply(lambda x: x[0])

        IMG_PATH_EXT = 'external_image_path/'
        all_fols = os.listdir(IMG_PATH_EXT)

        external_df['path'] = ['' for _ in range(len(external_df))]
        external_df['image_id'] = ['' for _ in range(len(external_df))]

        for p in external_df['patient_id'].unique():
            p_df = external_df[external_df.patient_id == p]
            p_df_index = p_df.index
            p_df = p_df.reset_index(drop=True)

            ids = [x.split('_')[-1] for x in all_fols if p in x]

            if len(ids) != len(p_df):
                print(f'ALARM ON {p}')

            pathes, img_ids = [], []
            for id in ids:
                pth_ = glob.glob(IMG_PATH_EXT + f'{p}_{id}/*.png')[0]
                img_ids.append(id)
                pathes.append(pth_)

            external_df.loc[p_df_index, 'path'] = pathes
            external_df.loc[p_df_index, 'image_id'] = img_ids

        external_df['fold'] = (np.ones(len(external_df)) * 999).astype('int')
        external_df['difficult_negative_case'] = [False for _ in range(len(external_df))]

        external_df = pd.concat([external_df for _ in range(CFG.external_multiplier)]).reset_index(drop=True)

        print(external_df.shape)
        external_df.head()

        train = pd.concat([train, external_df]).reset_index(drop=True)
        print(train.shape)

    if CFG.train:
        if CFG.FULL_TRAIN:
            LOGGER = get_logger(CFG.base_path + 'results/' + CFG.exp_name + '/train_full')
            train_loop_full_set(CFG, train, LOGGER)
        else:
            for fold_ in CFG.trn_fold:
                LOGGER = get_logger(CFG.base_path + 'results/' + CFG.exp_name + f'/train_f{fold_}')
                train_loop(CFG, train, fold_, LOGGER)


if __name__ == "__main__":
    main(CFG)
