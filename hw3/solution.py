#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import os
from timeit import default_timer as timer

import lightgbm as lgb
import numpy as np
import pandas as pd


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    data = data.sort_values('QueryId', kind='stable')
    features = [column for column in data.columns if column not in ('QueryId', 'Label')]
    queries = np.random.RandomState(42).permutation(data['QueryId'].unique())
    valid_mask = data['QueryId'].isin(queries[:int(np.ceil(0.2 * len(queries)))])
    train_data, valid_data = data[~valid_mask], data[valid_mask]
    train_pool = lgb.Dataset(
        train_data[features], label=train_data['Label'],
        group=train_data.groupby('QueryId', sort=False).size().to_numpy(),
    )
    valid_pool = lgb.Dataset(
        valid_data[features], label=valid_data['Label'],
        group=valid_data.groupby('QueryId', sort=False).size().to_numpy(),
        reference=train_pool,
    )
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'label_gain': [0, 1, 3, 7],
        'lambdarank_truncation_level': 13,
        'learning_rate': 0.03,
        'num_leaves': 15,
        'min_data_in_leaf': 50,
        'lambda_l2': 5.0,
        'feature_fraction': 0.9,
        'seed': 42,
        'num_threads': 4,
        'deterministic': True,
        'force_col_wise': True,
        'verbosity': -1,
    }
    model = lgb.train(
        params, train_pool, num_boost_round=2000,
        valid_sets=[valid_pool], valid_names=['valid'],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    best_iteration = model.best_iteration
    print(f'Validation nDCG@10 = {model.best_score["valid"]["ndcg@10"]:.6f}; '
          f'iterations = {best_iteration}')

    full_pool = lgb.Dataset(
        data[features], label=data['Label'],
        group=data.groupby('QueryId', sort=False).size().to_numpy(),
    )
    model = lgb.train(params, full_pool, num_boost_round=best_iteration)
    model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    features = [column for column in test.columns if column not in ('ObjectId', 'QueryId')]
    model = lgb.Booster(model_file=model_file)
    scores = model.predict(test[features], num_threads=4)

    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
