import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
from matplotlib import rcParams

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score as r2
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from credit_default import ReadWriteDataset, print_time, export_to_excel, \
    memory_compression, DataProcessing

import featuretools as ft
import featuretools.variable_types as vtypes
from itertools import combinations

warnings.filterwarnings("ignore")

SEED = 2021
USE_CORES = os.cpu_count() - 1
PATH_FILES = r'D:\python-txt\credit'
PATH_EXPORT = r'D:\python-txt\credit'
FILE_LEARN0 = os.path.join(PATH_FILES, 'to_learn_all.csv')
FILE_LEARN1 = os.path.join(PATH_FILES, 'to_learn_ai.csv')
FILE_LEARN2 = os.path.join(PATH_FILES, 'to_learn_cla.csv')
FILE_LEARN3 = os.path.join(PATH_FILES, 'to_learn_cs.csv')


class NewTargetFeature:
    """Генерация новых фич для заполнения пропусков"""

    def __init__(self):
        """ Инициализация класса """
        self.cat_features = []
        self.exclude_columns = ['Learn']
        self.learn_columns = []
        self.dummy = []
        self.comment = []

    def transform(self, df_in, dummy_cols=[], exclude_cols=[]):
        """
        Преобразование данных
        :param df_in: входной ДФ
        :param dummy_cols: категорийные колонки для преобразования в признаки
        :param exclude_cols: колонки не участвующие в обработке
        :return: ДС с новыми признаками
        """
        self.dummy = dummy_cols
        df = df_in.copy(deep=True)

        # Добавление новых признаков
        # df = DataProcessing.new_features(df)

        # деление категорий по столбцам
        if self.dummy:
            df_dummy = pd.get_dummies(df[self.dummy], columns=self.dummy)
            df = pd.concat([df, df_dummy], axis=1)
            self.cat_features.extend(df_dummy.columns.values)
            self.exclude_columns.extend(self.dummy)

        # для категорийных признаков установим тип данных 'category'
        for cat_column in self.cat_features:
            if cat_column in df.columns:
                df[cat_column] = df[cat_column].astype('category')

        # колонки, которые нужно исключить из обучения
        self.exclude_columns.extend(exclude_cols)
        # колонки для обучения = собраны из ДФ - колонки с категориями в dummy
        self.learn_columns = [col_name for col_name in df.columns.values
                              if col_name not in self.exclude_columns]
        return df


def process_model(use_model=RandomForestRegressor(random_state=SEED),
                  params={'max_depth': [7]}, folds_range=[], fold_single=5,
                  verbose=0, build_model=False, show_preds=False):
    """
    Поиск лучшей модели
    :param show_preds: отображать распределение
    :param use_model: модель для обучения и предсказаний
    :param params: параметры для модели
    :param folds_range: диапазон фолдов на сколько разбивать обучающие данные
    :param fold_single: на сколько фолдов разбивать данные финальной модели
    :param verbose: = 1 - отображать процесс
    :param build_model: = True - строить модель и выгружать предсказания
    :return: параметры модели, feat_imp_df - датафрейм с фичами
    """

    def iter_folds(n_fold, verb=0, cat=''):
        """
        Итерация для поиска лучшей модели для заданного количества флодов
        :param n_fold: количество фолдов
        :param verb: = 1 - отображать процесс
        :type cat: префикс для catBoost
        :return: r2_score_train, r2_score_valid, модель
        """
        skf = StratifiedKFold(n_splits=n_fold, random_state=SEED, shuffle=True)
        # skf = KFold(n_splits=n_fold, random_state=SEED, shuffle=True)
        if cat == 'cb_':
            search_cv = use_model.grid_search(params, X_train, y_train, cv=skf,
                                              stratified=True, refit=True)
            use_model.fit(X_train, y_train)
            y_train_pred = use_model.predict(X_train)
            y_valid_pred = use_model.predict(X_valid)
            best_ = search_cv['params']
            search_cv = use_model
            print(f'best_score: {use_model.best_score_}')
        else:
            search_cv = GridSearchCV(use_model, params, cv=skf, scoring='r2',
                                     verbose=verb, n_jobs=USE_CORES)
            search_cv.fit(X_train, y_train)
            best_ = search_cv.best_params_
            best_tree_cv = search_cv.best_estimator_
            y_train_pred = best_tree_cv.predict(X_train)
            y_valid_pred = best_tree_cv.predict(X_valid)
        r2_score_train = r2(y_train, y_train_pred)
        r2_score_valid = r2(y_valid, y_valid_pred)
        print(f'folds={n_fold:2d}, r2_train={r2_score_train:0.7f},'
              f' r2_valid={r2_score_valid:0.7f}'
              f' best_params={best_}')
        return r2_score_train, r2_score_valid, best_, search_cv, \
               y_train_pred, y_valid_pred

    submit_prefix = ''
    prefixes = {'import RandomForestRegressor': 'rf_',
                'import ExtraTreesRegressor': 'et_',
                'import GradientBoostingRegressor': 'gb_',
                'LightGBM regressor': 'lg_',
                'None': 'cb_'}
    for model_name, prefix in prefixes.items():
        if model_name in str(use_model.__doc__):
            submit_prefix = prefix
            break

    if folds_range:
        print('Поиск лучших параметров...')
    start_time_cv = time.time()
    best_folds = []
    for folds in folds_range:
        r2_trn, r2_vld, best_prms, search, _, _ = iter_folds(folds, verbose,
                                                             submit_prefix)
        best_folds.append([r2_trn, r2_vld, folds, best_prms, search])
    if folds_range:
        # печатаем модели в порядке убывания рейтинга
        best_folds.sort(key=lambda x: (-x[1], x[2]))
        print()
        fold_single = best_folds[0][2]
        time_stamp = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        file_name = os.path.join(PATH_EXPORT, 'log_work.csv')
        with open(file_name, 'a') as log:
            for line in best_folds:
                print(line[:4])
                log.write(f'{time_stamp}, {line[:5]}\n')
            log.write('\n')
        print_time(start_time_cv)
    # построение лучшей модели
    if build_model:
        start_time_cv = time.time()
        print('Обучение модели...')
        r2_trn, r2_vld, prms, search, trn_p, vld_p = iter_folds(fold_single,
                                                                verbose,
                                                                submit_prefix)
        print(prms)
        print()
        if submit_prefix == 'cb_':
            max_depth = prms['depth']
        else:
            max_depth = search.best_params_['max_depth']

        if show_preds:
            evaluate_preds(y_train, trn_p, y_valid, vld_p)

        feat_imp = search.feature_importances_
        feat_imp_df = pd.DataFrame({'features': X.columns.values,
                                    'importances': feat_imp})
        feat_imp_df.sort_values('importances', ascending=False, inplace=True)

        # сохранение результатов итерации в файл
        file_name = os.path.join(PATH_EXPORT, 'results_model.csv')
        if os.path.exists(file_name):
            file_df = pd.read_csv(file_name)
            file_df.time_stamp = pd.to_datetime(file_df.time_stamp,
                                                format='%y-%m-%d %H:%M:%S')
            file_df.time_stamp = file_df.time_stamp.dt.strftime(
                '%y-%m-%d %H:%M:%S')
            if 'comment' not in file_df.columns:
                file_df['comment'] = ''
        else:
            file_df = pd.DataFrame()
        date_now = datetime.now()
        time_stamp = date_now.strftime('%y-%m-%d %H:%M:%S')
        features_list = feat_imp_df.to_dict(orient='split')['data']
        temp_df = pd.DataFrame({'time_stamp': time_stamp,
                                'mdl': submit_prefix[:2].upper() + ' ai',
                                'max_depth': max_depth,
                                'folds': fold_single,
                                'r2_train': r2_trn,
                                'r2_valid': r2_vld,
                                'best_params': [prms],
                                'features': [features_list],
                                'column_dummies': [features_gen.dummy],
                                'model_columns': [mdl_columns],
                                'category_columns': [learn_cats],
                                'learn_exclude': [lrn_exclude],
                                'comment': [features_gen.comment]
                                })

        file_df = file_df.append(temp_df)
        file_df.r2_train = file_df.r2_train.round(7)
        file_df.r2_valid = file_df.r2_valid.round(7)
        file_df.to_csv(file_name, index=False)
        file_df.name_export_to_excel = 'results_model'
        # экспорт в эксель
        export_to_excel(file_df)
        print_time(start_time_cv)
        return feat_imp_df
    else:
        return best_folds


def find_depth(use_model, not_sklearn=0, show_plot=True):
    print(use_model)
    # Подберем оптимальное значение глубины обучения дерева.
    scores = pd.DataFrame(columns=['max_depth', 'train_score', 'valid_score'])
    max_depth_values = range(2, 11)
    for max_depth in max_depth_values:
        print(f'max_depth = {max_depth}')
        if not_sklearn == 1:
            find_model = use_model(random_state=SEED, max_depth=max_depth,
                                   num_leaves=63)
        elif not_sklearn == 2:
            find_model = use_model(silent=True, random_state=SEED,
                                   max_depth=max_depth,
                                   cat_features=learn_cats,
                                   eval_metric='RMSE',
                                   early_stopping_rounds=30, )
        else:
            find_model = use_model(random_state=SEED, max_depth=max_depth,
                                   criterion='mse')

        find_model.fit(X_train, y_train)

        y_train_pred = find_model.predict(X_train)
        y_valid_pred = find_model.predict(X_valid)
        train_score = r2(y_train, y_train_pred)
        valid_score = r2(y_valid, y_valid_pred)

        print(f'\ttrain_score = {train_score:.7f}')
        print(f'\tvalid_score = {valid_score:.7f}\n')

        scores.loc[len(scores)] = [max_depth, train_score, valid_score]
    scores.max_depth = scores.max_depth.astype(int)
    scores_data = pd.melt(scores,
                          id_vars=['max_depth'],
                          value_vars=['train_score', 'valid_score'],
                          var_name='dataset_type',
                          value_name='score')
    if show_plot:
        # Визуализация
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='max_depth', y='score', hue='dataset_type',
                     data=scores_data)
        plt.show()
    print(scores.sort_values('valid_score', ascending=False))
    print()
    print('Наилучший результат с параметрами:')
    print(scores.loc[scores.valid_score.idxmax()])
    print()


def evaluate_preds(train_true, train_pred, test_true, test_pred):
    """
    Отображение теста и предсказаний
    :param train_true:
    :param train_pred:
    :param test_true:
    :param test_pred:
    :return: None
    """
    print("Train R2:\t" + str(round(r2(train_true, train_pred), 5)))
    print("Valid R2:\t" + str(round(r2(test_true, test_pred), 5)))

    check_train = pd.DataFrame(
        {"true": train_true, "pred": train_pred.flatten()})
    check_test = pd.DataFrame({"true": test_true, "pred": test_pred.flatten()})

    plt.figure(figsize=(15, 12))
    plt.title('Тренировочные предсказания vs Валидационные предсказания')
    sns.scatterplot(data=check_train, x="pred", y="true", alpha=.5,
                    label="Train data")
    sns.scatterplot(data=check_test, x="pred", y="true", alpha=.5,
                    label="Valid data")
    plt.show()


TargetEnc = [
    'TargetEnc_grp_purpose',
    'TargetEnc_debt_group',
    'TargetEnc_rest_bal_group',
    'TargetEnc_NumberOfCreditProblems',
    'TargetEnc_HomeOwnership',
    'TargetEnc_grp_purpose_debt_group',
    'TargetEnc_grp_purpose_rest_bal_group',
    'TargetEnc_grp_purpose_NumberOfCreditProblems',
    'TargetEnc_grp_purpose_HomeOwnership',
    'TargetEnc_debt_group_rest_bal_group',
    'TargetEnc_debt_group_NumberOfCreditProblems',
    'TargetEnc_debt_group_HomeOwnership',
    'TargetEnc_rest_bal_group_NumberOfCreditProblems',
    'TargetEnc_rest_bal_group_HomeOwnership',
    'TargetEnc_NumberOfCreditProblems_HomeOwnership'
]


def make_model(target_enc_feats=[], es_groups=[]):
    """
    Построение модели
    :param es_groups: список признаков для добавления новых фич
    :param target_enc_feats: колонка с таргетенкодингом
    :param multy_feature: колонка на которую умножается target_enc_feature
    :return: None
    """
    global features_gen, df_lrn, mdl_columns, test_df, X, y
    global X_train, X_valid, y_train, y_valid, learn_cats, lrn_exclude

    # это передается в метод
    num_pos = 0

    df_lrn = pd.DataFrame()

    # чтение предобработанного датасета
    # для моделей по AnnualIncome, CurrentLoanAmount, CreditScore
    df_lrn = ReadWriteDataset.read_dataset(FILE_LEARN0)

    df_lrn['credit_problems'] = df_lrn.NumberOfCreditProblems.astype(
        'category')
    dict_purposes = {p: i for i, p in enumerate(df_lrn.grp_purpose.unique())}
    df_lrn['num_purpose'] = df_lrn.grp_purpose.map(dict_purposes)

    # признаки с отметками пропущенных значений
    feats_isna = ['AnnualIncomeIsGood', 'CurrentLoanAmountIsGood',
                  'CreditScoreIsGood']
    # колонка с метками пропущенных значений
    isna_column = feats_isna[num_pos]

    # признаки с пропущенными значениями
    feats_cols = ['AnnualIncome', 'CurrentLoanAmount', 'CreditScore']
    # колонка для предсказаний
    target_column = feats_cols[num_pos]

    # очистка существующих значений в колонке с пропущенными значениями
    df_lrn.loc[df_lrn[isna_column] < 1, target_column] = np.NaN

    # колонки для группировки
    group_cols = ['HomeOwnership', 'grp_purpose', 'debt_group',
                  'rest_bal_group', 'credit_problems']

    prfx_dict = {0: '_ai_me', 1: '_cla_me', 2: '_cs_me'}

    features_gen = NewTargetFeature()
    df_lrn = features_gen.transform(df_lrn,
                                    exclude_cols=feats_isna + TargetEnc)
    df_lrn = memory_compression(df_lrn)

    # категорийные колонки
    learn_cats = features_gen.cat_features + group_cols
    # learn_cats = []  # убрать категории
    learn_cats.append('Purpose')
    # все колонки для обучения
    learn_cols = features_gen.learn_columns
    # уберем более колонки про кредит и метку обучения
    lrn_exclude = ['CreditDefault']
    # уберем более крупные группировки
    # lrn_exclude.extend(group_cols)  # убрать категории
    lrn_exclude.extend(features_gen.exclude_columns)
    lrn_exclude.append(target_column)
    print('Исключаем:', lrn_exclude)

    target_feats = [col for col in learn_cols if prfx_dict[num_pos] in col]
    print('target_feats', target_feats)

    # колонки для обучения
    mdl_columns = [col for col in learn_cols if col not in lrn_exclude and
                   prfx_dict[num_pos] not in col]

    mdl_columns.append(target_column)
    mdl_columns.append(isna_column)
    mdl_columns.extend(target_enc_feats)
    learn_cats = [col for col in learn_cats if col in mdl_columns]

    print('Обучаемся:', mdl_columns)
    print('Категории:', learn_cats)

    if len(es_groups):
        print(f'Генерация новых признаков по {es_groups}')
        start_time = time.time()

        agg_primitives = ['median', 'mode', 'num_unique',
                          'percent_true', 'count', 'std']

        features_gen.comment.append({'featuretools': (es_groups,
                                                      agg_primitives)})
        # creating and entity set 'es'
        es = ft.EntitySet(id='Credits')
        es_cat_cols = [col for col in learn_cats if col not in mdl_columns]
        # добавим колонку с индексом
        df_lrn.insert(0, 'ID', df_lrn.index)
        variable_types = {col: vtypes.Categorical for col in learn_cats if
                          col != 'NumberOfCreditProblems'}
        es_dataset_cols = es_cat_cols + mdl_columns
        # print(es_dataset_cols)
        # добавим колонки с target_encoding
        # es_dataset_cols.extend(processor_data.target_encoding_feats)
        # print(es_dataset_cols)
        es.entity_from_dataframe(entity_id='Clients',
                                 dataframe=df_lrn[es_dataset_cols],
                                 index='ID',
                                 variable_types=variable_types)

        for es_group in es_groups:
            es = es.normalize_entity(base_entity_id='Clients',
                                     new_entity_id=es_group, index=es_group)
            feats_matrix, feat_names = ft.dfs(entityset=es,
                                              target_entity='Clients',
                                              agg_primitives=agg_primitives,
                                              verbose=3)
            feats_matrix_enc, feats_enc = ft.encode_features(feats_matrix,
                                                             feat_names,
                                                             include_unknown=False)
        print_time(start_time)

        # по этим колонкам не нужны новые признаки
        drop_cols = [col for col in feats_matrix_enc.columns if
                     f'.{isna_column}' in col or
                     f'.{target_column}' in col or
                     f'.num_purpose' in col]
        feats_matrix_enc.drop(drop_cols, axis=1, inplace=True)
        print('Размер нового датасета:', feats_matrix_enc.shape)
        with open(os.path.join(PATH_EXPORT, 'matrix_cols.csv'), 'w') as fm:
            for col in feats_matrix_enc.columns:
                fm.write(f'{col}\n')

        # удаление признаков с высокой корреляцией
        # Threshold for removing correlated variables
        threshold = 0.8
        # Absolute value correlation matrix
        corr_matrix = feats_matrix_enc.corr().abs()
        corr_matrix_up = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                                   k=1).astype(np.bool))
        # Select columns with correlations above threshold
        collinear_features = [column for column in corr_matrix_up.columns if
                              any(corr_matrix_up[column] > threshold) and
                              any(True for elem in agg_primitives if
                                  elem.upper() in column)]
        # убрать колонки с сильной корреляцией
        # feats_matrix_enc.drop(collinear_features, axis=1, inplace=True)
        #
        mdl_columns = feats_matrix_enc.columns.to_list()
        df_lrn = feats_matrix_enc.copy(deep=True)
        for col in feats_matrix_enc.columns.to_list():
            if df_lrn[col].isna().sum() > 0 and col != target_column:
                print(f'Пропуски в {col} = {df_lrn[col].isna().sum()}')
                mdl_columns.remove(col)
                # print(dataset[col].describe())
        with open(os.path.join(PATH_EXPORT, 'model_cols.csv'), 'w') as mc:
            for col in mdl_columns:
                mc.write(f'{col}\n')

    # train_lrn = df_lrn[df_lrn.Learn == 1]
    train_lrn = df_lrn.copy(deep=True)
    # print(train_lrn.info())

    # обучающий датасет
    train_lrn = train_lrn[train_lrn[isna_column] > 0][mdl_columns]
    train_lrn.drop(isna_column, axis=1, inplace=True)
    # тестовый датасет
    test_lrn = df_lrn[df_lrn[isna_column] < 1][mdl_columns]
    test_lrn.drop([target_column, isna_column], axis=1, inplace=True)

    X_lrn = train_lrn.drop(target_column, axis=1)
    y_lrn = train_lrn[target_column].astype('float32')

    # добавил так, чтобы много не править
    X = X_lrn
    y = y_lrn

    txt = ('Размер ', ' пропусков ')
    print(f'{txt[0]}трейна: {X.shape}{txt[1]}{X.isna().sum().sum()}')
    print(f'{txt[0]}теста: {test_lrn.shape}'
          f'{txt[1]}{test_lrn.isna().sum().sum()}')
    # print(test_lrn.isna().sum())

    X_train, X_valid, \
    y_train, y_valid = train_test_split(X, y, test_size=0.3,
                                        shuffle=True,
                                        random_state=SEED,
                                        stratify=X.num_purpose
                                        )
    print()
    print(f'{txt[0]}X_train: {X_train.shape}'
          f'{txt[1]}{X_train.isna().sum().sum()}')
    print(f'{txt[0]}X_valid: {X_valid.shape}'
          f'{txt[1]}{X_valid.isna().sum().sum()}')

    # X_train, X_valid, y_train, y_valid = X, X, y, y

    # определение моделей
    # model = RandomForestRegressor(random_state=SEED, criterion='mse')
    # model = ExtraTreesRegressor(random_state=SEED, criterion='mse')
    # model = GradientBoostingRegressor(random_state=SEED, criterion='mse')
    # model = LGBMRegressor(random_state=SEED, num_leaves=63)
    model = CatBoostRegressor(random_state=SEED, loss_function='RMSE',
                              silent=True,
                              cat_features=learn_cats,
                              early_stopping_rounds=30, )

    # find_depth(RandomForestRegressor)
    # max_depth      11.000000
    # train_score     0.776896
    # valid_score     0.336233
    # find_depth(ExtraTreesRegressor)
    # max_depth      12.000000
    # train_score     0.771015
    # valid_score     0.344225
    # find_depth(GradientBoostingRegressor)
    # max_depth      4.000000
    # train_score    0.656405
    # valid_score    0.351794
    # find_depth(LGBMRegressor, 1)
    # max_depth      3.000000
    # train_score    0.506928
    # valid_score    0.348825
    # find_depth(CatBoostRegressor, 2)
    # max_depth      6.000000
    # train_score    0.720083
    # valid_score    0.352229

    # настройки для первого приближения: поиск глубины деревьев и
    # количества фолдов
    # f_params = {'max_depth': list(range(4, 15))}
    f_params = {'max_depth': list(range(4, 9))}
    # раскомментарить эту строку для расчета
    # process_model(model, params=f_params, folds_range=list(range(3, 5)))

    # models = []
    # for depth in range(3, 11):
    #     param = {'max_depth': [depth]}
    #     mdl = process_model(model, params=param,
    #                         folds_range=list(range(3, 6)))
    #     models.append(mdl[0][:4])
    # models.sort(key=lambda x: (-x[1], x[2]))
    # print()
    # for elem in models:
    #     print(elem)

    feat_imp_df_ = pd.DataFrame

    model = CatBoostRegressor(silent=True, random_state=SEED,
                              cat_features=learn_cats,
                              eval_metric='R2',
                              early_stopping_rounds=30, )
    f_params = {
        'max_depth': [4],
        'iterations': range(350, 451, 10),
        # 'learning_rate': [.05]
        # 'learning_rate': [.01, .025, .05, .075, 0.1]
        # 'min_samples_leaf': list(range(1, 9, 2)),
        # 'min_samples_split': list(range(2, 9, 2)),  # не меньше 2
        # 'min_samples_leaf': [3],
        # 'min_samples_split': [19]
    }
    # раскомментарить эту строку для расчета
    feat_imp_df_ = process_model(model, params=f_params, fold_single=5,
                                 verbose=1, build_model=True)
    print(feat_imp_df_)


if __name__ == "__main__":
    total_time = time.time()

    # перебор колонок с группировкой по метрикам
    enc_feats = [
        'grp_purpose_ai_median',
        'grp_purpose_ai_mean',
        'grp_purpose_ai_med_mean',
        # 'debt_group_ai_median',
        # 'debt_group_ai_mean',
        # 'debt_group_ai_med_mean',
        'rest_bal_group_ai_median',
        'rest_bal_group_ai_mean',
        'rest_bal_group_ai_med_mean',
        'NumberOfCreditProblems_ai_median',
        'NumberOfCreditProblems_ai_mean',
        'NumberOfCreditProblems_ai_med_mean',
        'HomeOwnership_ai_median',
        'HomeOwnership_ai_mean',
        'HomeOwnership_ai_med_mean'
    ]
    # for target_feat in enc_feats:
    #     target_feats = ['debt_group_ai_median', target_feat]
    #     # target_feats = [target_feat]
    #     make_model(target_enc_feats=target_feats)
    make_model(target_enc_feats=['debt_group_ai_median'])
    # best_params = {'depth': 4, 'iterations': 390}
    # make_model(target_enc_feats=['debt_group_ai_median',
    #                              'grp_purpose_ai_med_mean'])
    # best_params = {'depth': 4, 'iterations': 370}

    # колонки для группировки
    cat_groups = ['HomeOwnership', 'grp_purpose', 'debt_group',
                  'Purpose', 'rest_bal_group', 'NumberOfCreditProblems']
    # генерация новых фич с featuretools
    # for num_new_groups in range(1, len(cat_groups) + 1):
    #     for group_es in combinations(cat_groups, num_new_groups):
    #         make_model(es_groups=group_es)
    # make_model(es_groups=['grp_purpose'])

    # make_model()

    print_time(total_time)

    # prims = ft.list_primitives()
    # prims.name_export_to_excel = 'primitives'
    # # экспорт в эксель
    # export_to_excel(prims)
