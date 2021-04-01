import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
from matplotlib import rcParams

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score as r2
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering

import featuretools as ft
import featuretools.variable_types as vtypes
from itertools import combinations

warnings.filterwarnings("ignore")

SEED = 2021
USE_CORES = os.cpu_count() - 1
PATH_FILES = r'D:\python-txt\credit'
PATH_EXPORT = r'D:\python-txt\credit'
FILE_TRAIN = os.path.join(PATH_FILES, 'train.csv')
FILE_TEST = os.path.join(PATH_FILES, 'test.csv')
FILE_WITH_FEATURES = os.path.join(PATH_EXPORT, 'df_all.csv')
FILE_SAMPLE = os.path.join(PATH_EXPORT, 'sample_submission.csv')
FILE_SUBMIT = os.path.join(PATH_EXPORT, 'submission.csv')

rcParams.update({'font.size': 14})  # размер шрифта на графиках
pd.options.display.max_columns = 100
global_start_time = time.time()

TargetEnc = [
    'TargetEnc_grp_purpose',
    'TargetEnc_debt_group',
    'TargetEnc_rest_bal_group',
    'TargetEnc_NumberOfCreditProblems',
    'TargetEnc_HomeOwnership',
    'TargetEnc_purpose_term',
    'TargetEnc_grp_purpose_debt_group',
    'TargetEnc_grp_purpose_rest_bal_group',
    'TargetEnc_grp_purpose_NumberOfCreditProblems',
    'TargetEnc_grp_purpose_HomeOwnership',
    'TargetEnc_grp_purpose_purpose_term',
    'TargetEnc_debt_group_rest_bal_group',
    'TargetEnc_debt_group_NumberOfCreditProblems',
    'TargetEnc_debt_group_HomeOwnership',
    'TargetEnc_debt_group_purpose_term',
    'TargetEnc_rest_bal_group_NumberOfCreditProblems',
    'TargetEnc_rest_bal_group_HomeOwnership',
    'TargetEnc_rest_bal_group_purpose_term',
    'TargetEnc_NumberOfCreditProblems_HomeOwnership',
    'TargetEnc_NumberOfCreditProblems_purpose_term',
    'TargetEnc_HomeOwnership_purpose_term'
]


def process_model(use_model=CatBoostClassifier(random_state=SEED),
                  params={'max_depth': [5]}, folds_range=[], fold_single=5,
                  verbose=0, build_model=False):
    """
    Поиск лучшей модели
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
        :param cat: True - обучаем catBoost
        :return: f1_score_train, f1_score_valid, модель
        """
        skf = StratifiedKFold(n_splits=n_fold, random_state=SEED, shuffle=True)
        if cat == 'cb_':
            gscv = use_model.grid_search(params, X_train, y_train, cv=skf,
                                         stratified=True, refit=True)
            use_model.fit(X_train, y_train)
            y_train_pred = use_model.predict(X_train)
            y_valid_pred = use_model.predict(X_valid)
            best_ = gscv['params']
            gscv = use_model
        else:
            gscv = GridSearchCV(use_model, params, cv=skf, scoring='f1',
                                verbose=verb, n_jobs=USE_CORES)
            gscv.fit(X_train, y_train)
            best_ = gscv.best_params_
            best_tree_cv = gscv.best_estimator_
            y_train_pred = best_tree_cv.predict(X_train)
            y_valid_pred = best_tree_cv.predict(X_valid)
        f1_train = f1_score(y_train, y_train_pred)
        f1_valid = f1_score(y_valid, y_valid_pred)
        print(f'folds={n_fold:2d}, f1_score_train={f1_train:0.7f},'
              f' f1_score_valid={f1_valid:0.7f}'
              f' best_params={best_}')
        return f1_train, f1_valid, best_, gscv, y_train_pred, y_valid_pred

    submit_prefix = ''
    prefixes = {'import RandomForestClassifier': 'rf_',
                'import ExtraTreesClassifier': 'et_',
                'import GradientBoostingClassifier': 'gb_',
                'LightGBM classifier': 'lg_',
                'None': 'cb_'}
    for model_name, prefix in prefixes.items():
        if model_name in str(use_model.__doc__):
            submit_prefix = prefix
            break
    file_submit_csv = os.path.join(PATH_EXPORT, 'predictions',
                                   f'{submit_prefix}submit.csv')
    if folds_range:
        print('Поиск лучших параметров...')
    start_time_cv = time.time()
    best_folds = []
    for folds in folds_range:
        f1_trn, f1_vld, best_prms, search, _, _ = iter_folds(folds, verbose,
                                                             submit_prefix)
        best_folds.append([f1_trn, f1_vld, folds, best_prms, search])
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
        f1_trn, f1_vld, prms, search, trn_p, vld_p = iter_folds(fold_single,
                                                                verbose,
                                                                submit_prefix)
        print(prms)
        print()
        if submit_prefix == 'cb_':
            max_depth = prms['depth']
        else:
            max_depth = search.best_params_['max_depth']

        feat_imp = search.feature_importances_
        feat_imp_df = pd.DataFrame({'features': X.columns.values,
                                    'importances': feat_imp})
        feat_imp_df.sort_values('importances', ascending=False, inplace=True)
        # предсказание
        search.fit(X, y)
        submit = pd.read_csv(FILE_SAMPLE, index_col='Id')
        submit['Credit Default'] = search.predict(test_df)
        submit_proba = submit.copy(deep=True)
        submit_proba['p_value'] = search.predict_proba(test_df)[:, 1]
        submit_proba['new'] = submit_proba['p_value'].apply(
            lambda x: 1 if x > 0.4996 else 0)
        submit_proba['Credit Default'] = submit_proba['new']
        submit_proba.drop(['p_value', 'new'], axis=1, inplace=True)
        submit_proba.to_csv(file_submit_csv)
        date_now = datetime.now()
        time_stamp = date_now.strftime('%y%m%d%H%M%S')
        submit.to_csv(file_submit_csv.replace('.csv', f'_{time_stamp}.csv'))
        # сохранение результатов итерации в файл
        file_name = os.path.join(PATH_EXPORT, 'results.csv')
        if os.path.exists(file_name):
            file_df = pd.read_csv(file_name)
            file_df.time_stamp = pd.to_datetime(file_df.time_stamp,
                                                format='%y-%m-%d %H:%M:%S')
            file_df.time_stamp = file_df.time_stamp.dt.strftime(
                '%y-%m-%d %H:%M:%S')
            if 'delta' not in file_df.columns:
                file_df.insert(6, 'delta', 0)
            if 'comment' not in file_df.columns:
                file_df['comment'] = ''
            # удаление колонок после 'comment'
            list_columns = file_df.columns.to_list()
            file_df = file_df[list_columns[:list_columns.index('comment') + 1]]
        else:
            file_df = pd.DataFrame()
        time_stamp = date_now.strftime('%y-%m-%d %H:%M:%S')
        features_list = feat_imp_df.to_dict(orient='split')['data']
        temp_df = pd.DataFrame({'time_stamp': time_stamp,
                                'mdl': submit_prefix[:2].upper(),
                                'max_depth': max_depth,
                                'folds': fold_single,
                                'f1_train': f1_trn,
                                'f1_valid': f1_vld,
                                'delta': f1_trn - f1_vld,
                                'best_params': [prms],
                                'features': [features_list],
                                'column_dummies': [processor_data.dummy],
                                'model_columns': [model_columns],
                                'category_columns': [category_columns],
                                'learn_exclude': [learn_exclude],
                                'comment': [processor_data.comment]
                                })

        file_df = file_df.append(temp_df)
        file_df.f1_train = file_df.f1_train.round(7)
        file_df.f1_valid = file_df.f1_valid.round(7)
        file_df.delta = file_df.f1_train - file_df.f1_valid
        file_df.to_csv(file_name, index=False)
        file_df.name_export_to_excel = 'results'
        # экспорт в эксель
        export_to_excel(file_df)
        print_time(start_time_cv)
        return feat_imp_df
    else:
        return best_folds


def find_depth(use_model, max_depth_values=range(3, 11), not_sklearn=0,
               show_plot=True):
    print(use_model)
    # Подберем оптимальное значение глубины обучения дерева.
    scores = pd.DataFrame(columns=['max_depth', 'train_score', 'valid_score'])
    for max_depth in max_depth_values:
        print(f'max_depth = {max_depth}')
        if not_sklearn == 1:
            find_model = use_model(random_state=SEED, max_depth=max_depth,
                                   num_leaves=63)
        elif not_sklearn == 2:
            find_model = use_model(random_state=SEED, max_depth=max_depth,
                                   silent=True, early_stopping_rounds=20,
                                   cat_features=category_columns)
        else:
            find_model = use_model(random_state=SEED, max_depth=max_depth)

        find_model.fit(X_train, y_train)

        y_train_pred = find_model.predict(X_train)
        y_valid_pred = find_model.predict(X_valid)
        train_score = f1_score(y_train, y_train_pred)
        valid_score = f1_score(y_valid, y_valid_pred)

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
                     data=scores_data, markers=True)
        plt.show()
    print(scores.sort_values('valid_score', ascending=False))
    print()
    print('Наилучший результат с параметрами:')
    print(scores.loc[scores.valid_score.idxmax()])
    print()


def print_time(time_start):
    """
    Печать времени выполнения процесса
    :param time_start: время запуска в формате time.time()
    :return:
    """
    time_apply = time.time() - time_start
    hrs = time_apply // 3600
    mns = time_apply % 3600
    sec = mns % 60
    print(f'Время обработки: {hrs:.0f} час {mns // 60:.0f} мин {sec:.1f} сек')


def evaluate_preds(mdl, x_train, x_valid, ytrain, yvalid):
    y_train_pred = mdl.predict(x_train)
    y_valid_pred = mdl.predict(x_valid)
    show_classification_report(ytrain, y_train_pred, yvalid, y_valid_pred)


def show_classification_report(y_train_true, y_train_pred, y_valid_true,
                               y_valid_pred):
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('VALID\n\n' + classification_report(y_valid_true, y_valid_pred))
    print('CONFUSION MATRIX\n')
    print(pd.crosstab(y_valid_true, y_valid_pred))


def show_corr_matrix(df):
    plt.figure(figsize=(18, 12))
    sns.set(font_scale=1.4)
    corr_matrix = dataset[df.Learn == 1].drop('Learn', axis=1).corr()
    corr_matrix = np.round(corr_matrix, 2)
    corr_matrix[np.abs(corr_matrix) < 0.2] = 0
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
    plt.title('Корреляция признаков')
    plt.show()


def show_plot_feature(feature, hue_on):
    legends = []
    fig, ax = plt.subplots(figsize=(18, 7))
    for element in dataset[hue_on].unique():
        sns.kdeplot(dataset[(dataset[hue_on] == element)][feature], ax=ax)
        legends.append(f'Группа: {element}')
    ax.legend(legends)
    ax.set_xlim(dataset[feature].min(), dataset[feature].quantile(0.975))
    plt.title(f'Распределение признака {feature} в группах {hue_on}')
    plt.show()


def export_to_excel(data: pd.DataFrame) -> None:
    """
    # экспорт датафрема в эксель
    Convert the dataframe to an XlsxWriter Excel object.
    Note that we turn off default header and skip one row to allow us
    to insert a user defined header.
    :param data: dataframe
    :return: None
    """
    name_data = data.name_export_to_excel
    file_xls = os.path.join(PATH_EXPORT, f'{name_data}.xlsx')
    writer = pd.ExcelWriter(file_xls, engine='xlsxwriter')
    data.to_excel(writer, sheet_name=name_data, startrow=1,
                  header=False, index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[name_data]
    # Add a header format.
    font_name = 'Arial'
    header_format = workbook.add_format({
        'font_name': font_name,
        'font_size': 10,
        'bold': True,
        'text_wrap': True,
        'align': 'center',
        'valign': 'center',
        'border': 1})
    # Write the column headers with the defined format.
    worksheet.freeze_panes(1, 0)
    for col_num, value in enumerate(data.columns.values):
        worksheet.write(0, col_num, value, header_format)
    cell_format = workbook.add_format()
    cell_format.set_font_name(font_name)
    cell_format.set_font_size(12)
    for num, value in enumerate(data.columns.values):
        if value == 'time_stamp':
            width = 19
        elif value in ('mdl', 'folds'):
            width = 8
        elif value in ('max_depth', 'f1_train', 'f1_valid',
                       'r2_train', 'r2_valid', 'delta'):
            width = 14
        else:
            width = 32
        worksheet.set_column(num, num, width, cell_format)
    worksheet.autofilter(0, 0, len(data) - 1, len(data) - 1)
    writer.save()
    # End excel save


def memory_compression(df_in):
    """
    Изменение типов данных для экономии памяти
    :param df_in: исходный ДФ
    :return: сжатый ДФ
    """
    df = df_in.copy(deep=True)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        if str(df[col].dtype) not in ('object', 'category'):
            col_min = df[col].min()
            col_max = df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and \
                        col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and \
                        col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and \
                        col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and \
                        col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(df[col].dtype)[:5] == 'float':
                if col_min > np.finfo(np.float16).min and \
                        col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and \
                        col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Исходный размер датасета в памяти '
          f'равен {round(start_mem, 2)} мб.')
    print(f'Конечный размер датасета в памяти '
          f'равен {round(end_mem, 2)} мб.')
    print(f'Экономия памяти = {(1 - end_mem / start_mem):.1%}')
    return df


class ReadWriteDataset:
    """Класс для записи/чтения датасета"""

    @staticmethod
    def read_dataset(name_file=''):
        """
        Чтение датасета
        :param name_file: имя файла
        :return: ДФ c датасетом
        """
        if os.access(name_file, os.F_OK):
            file_pickle = name_file.replace('.csv', '.pkl')
            if os.access(file_pickle, os.F_OK):
                df = pd.read_pickle(file_pickle)
            else:
                df = pd.read_csv(name_file, sep=';')
            return df
        return pd.DataFrame()

    @staticmethod
    def write_dataset(df, name_file='df_all.csv'):
        """
        Сохранение датасета
        :param df: входной ДФ
        :param name_file: имя файла, куда записать
        :return: None
        """
        file_pickle = name_file.replace('.csv', '.pkl')
        df.to_csv(name_file, sep=';', index=False)
        df.to_pickle(file_pickle)


class NewTargetFeature:
    """Генерация новых фич для заполнения пропусков"""

    def __init__(self):
        """ Инициализация класса """
        self.indexes = None
        self.cat_features = []
        self.exclude_columns = ['Learn']
        self.learn_columns = []
        self.dummy = []

    def fit(self, indexes):
        self.indexes = indexes

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
        df = DataProcessing.new_features(df)

        # деление категорий по столбцам
        if self.dummy:
            df_dummy = pd.get_dummies(df[self.dummy], columns=self.dummy)
            df = pd.concat([df, df_dummy], axis=1)
            self.exclude_columns.extend(self.dummy)

        # колонки, которые нужно исключить из обучения
        self.exclude_columns.extend(exclude_cols)
        # колонки для обучения = собраны из ДФ - колонки с категориями в dummy
        self.learn_columns = [col_name for col_name in df.columns.values
                              if col_name not in self.exclude_columns]

        df.loc[df.Learn == 0, 'CreditDefault'] = np.NaN
        return df


class DataProcessing(ReadWriteDataset):
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.df_all = pd.DataFrame
        self.medians = None
        self.min_MonthlyDebt = None
        self.max_MonthsLastDel = None
        self.YearsInCurrentJob = None
        self.bin_years = {'< 1 year': 0, '1 year': 1, '2 years': 2,
                          '3 years': 3, '4 years': 4, '5 years': 5,
                          '6 years': 6, '7 years': 7, '8 years': 8,
                          '9 years': 9, '10+ years': 10}
        self.cat_groups = ['grp_purpose', 'debt_group', 'rest_bal_group',
                           'NumberOfCreditProblems', 'HomeOwnership',
                           'purpose_term']
        self.features_good = ['AnnualIncomeIsGood', 'CurrentLoanAmountIsGood',
                              'CreditScoreIsGood', 'calc_monthsIsGood',
                              'MonthsSinceLastDelinquentIsGood']
        self.target_encoding_feats = []
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.num_debt_bins = 7
        self.debt_bins = None
        self.debt_labels = None
        self.num_history_bins = 7
        self.history_bins = None
        self.history_labels = None
        self.med_history_job = None
        self.med_debt_income = None
        self.med_problems_score = None
        self.dummy = []
        self.cat_features = []
        self.exclude_columns = ['Learn', 'debt_group']
        self.learn_columns = []
        self.comment = []
        self.tsne_columns = []

    @staticmethod
    def concat_df(df_train, df_test):
        """Объединение датафреймов"""
        df_train['learn'] = 1
        df_test['learn'] = 0
        df = pd.concat([df_train, df_test])
        df.columns = [col_name.title().replace(' ', '') for col_name in
                      df.columns.values]
        return df

    def preprocess_df(self, df_in):
        """
        Небольшая предобработка данных для дальнейшей обработки
        :param df_in: ДФ
        :return: предобработанный ДФ
        """
        df = df_in.copy(deep=True)

        # для групировки по целям кредита
        df['grp_purpose'] = df.Purpose

        # # вот это добавил новое
        # # MonthlyDebt на трейне удалим нулевые значения
        # # если удалить - рейтинг хуже
        # df = df[~((df.Learn == 1) & (df.MonthlyDebt < 0.1))]
        # # нулевые значения заменим на минимальные для этой колонки
        df.loc[df.MonthlyDebt < 0.1, 'MonthlyDebt'] = self.min_MonthlyDebt

        # # MonthsSinceLastDelinquent заполнение выбросов - с этим хуже:
        # df.loc[df.MonthsSinceLastDelinquent > self.max_MonthsLastDel,
        #        'MonthsSinceLastDelinquent'] = self.max_MonthsLastDel

        # заменим выбросы на NaN
        df.loc[df.CurrentLoanAmount >= 99999999, 'CurrentLoanAmount'] = np.NaN

        # посчитаем количество месяцев кредита
        df['calc_months'] = df.CurrentLoanAmount / df.MonthlyDebt
        # выбросы заменим на NaN
        max_months = df.calc_months.quantile(0.995)
        df.loc[df.calc_months > max_months, 'calc_months'] = np.NaN

        df.Term = df.Term.map({'Short Term': 0, 'Long Term': 1})
        df.Term = df.Term.astype('int')

        # YearsInCurrentJob преобразуем в числовые значения
        df.YearsInCurrentJob = df.YearsInCurrentJob.map(self.bin_years)

        # новый признак цель кредита + срок
        df['purpose_term'] = df.grp_purpose + ' ' + df.Term.astype('str')

        # TaxLiens уменьшим кол-во групп
        df.loc[df.TaxLiens > 4, 'TaxLiens'] = 5
        df.TaxLiens = df.TaxLiens.astype('int')

        # NumberOfCreditProblems уменьшим кол-во групп
        df.loc[df.NumberOfCreditProblems > 4, 'NumberOfCreditProblems'] = 5
        df.NumberOfCreditProblems = df.NumberOfCreditProblems.astype('int')

        # CreditScore значения выше 1000 поделим на 10
        cond = df.CreditScore > 999
        df.loc[cond, 'CreditScore'] = df[cond].CreditScore / 10

        # добавление групп долга
        df['debt_group'] = pd.cut(df.MonthlyDebt, bins=self.debt_bins,
                                  labels=self.debt_labels).astype('str')

        # добавление групп кредитной истории
        df['history_group'] = pd.cut(df.YearsOfCreditHistory,
                                     bins=self.history_bins,
                                     labels=self.history_labels).astype('str')

        # добавление разницы между балансом и долгом
        df['rest_bal_debt'] = df.CurrentCreditBalance - df.MonthlyDebt
        # возвращается кортеж:
        _, rest_bins = pd.qcut(df.rest_bal_debt, q=self.num_debt_bins,
                               precision=0, retbins=True)
        rest_bins[0] = -np.inf
        rest_bins[-1] = np.inf
        if rest_bins[0] < 0 < rest_bins[1]:
            rest_bins[1] = 0
        df['rest_bal_group'] = pd.cut(df.rest_bal_debt, bins=rest_bins,
                                      labels=self.debt_labels).astype('str')
        return df

    @staticmethod
    def new_features(df_in):
        """
        Добавление новых признаков:
        :param df_in: ДФ
        :return: ДФ с новыми признаками
        """
        df = df_in.copy(deep=True)

        # попробовать эти три строки не делать
        df.loc[df.Bankruptcies > 0, 'Bankruptcies'] = 1
        df.loc[df.TaxLiens > 0, 'TaxLiens'] = 1
        # df.loc[df.NumberOfCreditProblems > 0, 'NumberOfCreditProblems'] = 1

        # сгруппируем цели: три большие группы в одну и остальные в другую
        df['grp_purpose'] = df.Purpose
        purpose = ['debt consolidation', 'other', 'home improvements']
        df.loc[df.Purpose.isin(purpose), 'Purpose'] = 0
        df.loc[df.Purpose != 0, 'Purpose'] = 1
        df.Purpose = df.Purpose.astype(int)
        return df

    def make_clusters(self, df, n_clusters):
        """
        Разбиение датасета на кластеры
        :param df:
        :param n_clusters:
        :param all_data: кластеризация на всем датасете
        :return: ДФ с кластерами
        """
        print(f'Кластеризация')
        cls_time = time.time()
        # # исходные колонки
        col_clusters = self.tsne_columns[:15]
        # # колонки после препроцессинга
        # col_clusters = self.tsne_columns

        col_clusters = [col for col in col_clusters if
                        str(df[col].dtype) not in ('object', 'category')]

        print('self.tsne_columns', self.tsne_columns)
        print('col_clusters', col_clusters)

        scaler = RobustScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(df[col_clusters]),
                                    columns=col_clusters, index=df.index)

        aggl = AgglomerativeClustering(n_clusters=n_clusters)
        labels = aggl.fit_predict(train_scaled)
        labels = pd.DataFrame(data=labels, columns=['cluster'], index=df.index)

        self.dummy.extend(['cluster'])
        print_time(cls_time)
        return labels

    def fit(self, df_in, num_debt_bins=7, num_history_bins=7):
        """Сохранение статистик"""
        df = df_in.copy(deep=True)

        self.min_MonthlyDebt = df[df.MonthlyDebt > 0.1].MonthlyDebt.min()
        self.max_MonthsLastDel = df.MonthsSinceLastDelinquent.quantile(0.995)

        # колонки для кластеризации
        self.tsne_columns = [col for col in df.columns if
                             col not in self.exclude_columns + [
                                 'CreditDefault']]

        if num_debt_bins > 3:
            self.num_debt_bins = num_debt_bins
        # возвращается кортеж:
        _, self.debt_bins = pd.qcut(df.MonthlyDebt, q=self.num_debt_bins,
                                    precision=0, retbins=True)
        self.debt_bins[0] = -np.inf
        self.debt_bins[-1] = np.inf
        self.debt_labels = [self.alphabet[i] for i in
                            range(self.num_debt_bins)]

        if num_history_bins > 2:
            self.num_history_bins = num_history_bins
        # возвращается кортеж:
        # _, self.history_bins = pd.qcut(df.YearsOfCreditHistory,
        #                                q=self.num_history_bins,
        #                                precision=0, retbins=True)
        _, self.history_bins = pd.cut(df.YearsOfCreditHistory,
                                      bins=self.num_history_bins,
                                      precision=0, retbins=True)
        self.history_bins[0] = -np.inf
        self.history_bins[-1] = np.inf
        self.history_labels = [self.alphabet[i] for i in
                               range(self.num_history_bins)]

        # небольшой препроцессинг, который нужен в двух методах fit и transform
        df = self.preprocess_df(df)

        self.YearsInCurrentJob = int(df.YearsInCurrentJob.mode()[0])

        # медианный YearsInCurrentJob по группам history_group
        self.med_history_job = df.groupby('history_group').agg(
            {'YearsInCurrentJob': 'median'})
        self.med_history_job.fillna(self.YearsInCurrentJob, inplace=True)
        self.med_history_job = self.med_history_job.astype('int').to_dict()[
            'YearsInCurrentJob']
        # print(self.YearsInCurrentJob, self.med_history_job)

        # Расчет медиан
        self.medians = df.median()

        # медианный AnnualIncome по группам debt_group
        self.med_debt_income = df.groupby('debt_group').agg(
            {'AnnualIncome': 'median'}).round(0).to_dict()['AnnualIncome']
        # print(self.med_debt_income)

        # медианный CreditScore по группам NumberOfCreditProblems
        self.med_problems_score = df.groupby('NumberOfCreditProblems').agg(
            {'CreditScore': 'median'}).astype('int').to_dict()['CreditScore']
        self.med_problems_score[3] -= 10
        # print(self.med_problems_score)

        feats_isna = {'AnnualIncome': 'ai_',
                      'CurrentLoanAmount': 'cla_',
                      'CreditScore': 'cs_',
                      'calc_months': 'cm_',
                      'MonthsSinceLastDelinquent': 'mld_'
                      }
        # file = open(os.path.join(PATH_EXPORT, 'groups.txt'), 'w')
        for name_feature, feat in feats_isna.items():
            cond = ~df[name_feature].isna()
            # print(name_feature)
            # группировка признака по Purpose, debt_group, NumberOfCrdProblems
            for col_grpby in self.cat_groups:
                grp_feat = df[cond].groupby(col_grpby, as_index=False).agg(
                    {name_feature: ['median', 'mean']})
                n_cols = [col_grpby, f'{feat}median', f'{feat}mean']
                grp_feat.columns = n_cols
                if name_feature == 'CreditScore' and \
                        col_grpby == 'NumberOfCreditProblems':
                    grp_feat.loc[3, 'cs_median'] -= 10
                grp_feat[f'{feat}med_mean'] = grp_feat[n_cols[1:]].mean(axis=1)
                grp_feat = grp_feat.round(0)
                if name_feature == 'CreditScore':
                    cols = grp_feat.columns.values[1:]
                    grp_feat.loc[:, cols] = grp_feat.loc[:, cols].astype('int')
                    # print(grp_feat)
                grp_feat = grp_feat.set_index(col_grpby)
                for atr in (f'{feat}median', f'{feat}mean', f'{feat}med_mean'):
                    name_atr = f'{col_grpby}_{atr}'
                    name_grp = grp_feat[atr]
                    name_grp.columns = [name_feature]
                    name_grp = name_grp.to_dict()
                    setattr(DataProcessing, name_atr, name_grp)
                    # if name_feature == 'calc_months':
                    #     print(name_atr)
                    #     print(name_grp)
                    # file.write(f'{name_atr}\n')
        # file.close()

        # группировки для Target encoding
        cond = (df.Learn == 1)
        all_groups = [''] + self.cat_groups
        for idx_one, grp_one in enumerate(all_groups):
            for grp_two in all_groups[idx_one + 1:]:
                if grp_one:
                    col_grpby = [grp_one] + [grp_two]
                else:
                    col_grpby = [grp_two]
                name_one = f'TargetEnc_{grp_one}'
                name_atr = '_'.join(['TargetEnc'] + col_grpby)
                # print('имена', col_grpby, name_atr)
                grp_feat = df[cond].groupby(col_grpby, as_index=False).agg(
                    {'CreditDefault': 'mean'}).rename(
                    columns={'CreditDefault': name_atr})
                if len(col_grpby) > 1:
                    # заполнение нулей в группировке по двум полям из
                    # вышестоящей группировки, т.е. по первому полю
                    grp_feat = grp_feat.merge(
                        getattr(DataProcessing, name_one), on=grp_one,
                        how='left')
                    grp_feat.loc[grp_feat[name_atr] > 0.001, name_one] = 0
                    grp_feat[name_atr] = grp_feat[[name_atr, name_one]].sum(
                        axis=1)
                    grp_feat.drop(name_one, axis=1, inplace=True)
                setattr(DataProcessing, name_atr, grp_feat)

    def make_target_encoding(self, df_in):
        """
        добавление колонок по Target encoding
        :param df_in: входной ДФ
        :return: выходной ДФ
        """
        df = df_in.copy(deep=True)

        all_groups = [''] + self.cat_groups
        for idx_one, grp_one in enumerate(all_groups):
            for grp_two in all_groups[idx_one + 1:]:
                if grp_one:
                    col_grpby = [grp_one] + [grp_two]
                else:
                    col_grpby = [grp_two]
                name_one = f'TargetEnc_{grp_one}'
                name_atr = '_'.join(['TargetEnc'] + col_grpby)
                # print('имена', col_grpby, name_atr)
                grp_feat = getattr(DataProcessing, name_atr)
                self.target_encoding_feats.append(name_atr)
                df = df.merge(grp_feat, on=col_grpby, how='left')
                if len(col_grpby) > 1 and df[name_atr].isna().sum() > 0:
                    # заполнение пропусков в новой фиче по двум полям из
                    # вышестоящей группировки, т.е. по первому полю
                    df.loc[df[name_atr].isna(), name_atr] = df.loc[
                        df[name_atr].isna(), name_one]
        # print('новые фичи', self.target_encoding_feats)
        # print('пропуски', df.TargetEnc_grp_purpose_debt_group.isna().sum())
        # print(df[df.TargetEnc_grp_purpose_debt_group.isna()])
        return df

    @staticmethod
    def fill_ai_cla_cs(df_in, num_pos):
        """
        Заполнение 'AnnualIncome', 'CurrentLoanAmount', 'CreditScore'
        на основе предсказаний модели
        :param df_in: входной ДФ
        :param num_pos: порядковый номер колонки для предсказаний
        :return: ДФ
        """
        df_lrn = df_in.copy(deep=True)

        target_enc_feats = {0: ['debt_group_ai_median'],
                            1: ['NumberOfCreditProblems_cla_median',
                                'grp_purpose_cla_median'],
                            2: ['grp_purpose_cs_median']}

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
                      'rest_bal_group']

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
        mdl_columns.extend(target_enc_feats[num_pos])
        learn_cats = [col for col in learn_cats if col in mdl_columns]

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

        models = {0: CatBoostRegressor(max_depth=4, random_state=SEED,
                                       cat_features=learn_cats,
                                       iterations=390, eval_metric='R2',
                                       early_stopping_rounds=30, ),
                  1: CatBoostRegressor(max_depth=3, random_state=SEED,
                                       cat_features=learn_cats,
                                       iterations=270, eval_metric='R2',
                                       early_stopping_rounds=30, ),
                  2: CatBoostRegressor(max_depth=4, random_state=SEED,
                                       cat_features=learn_cats,
                                       iterations=750, eval_metric='R2',
                                       early_stopping_rounds=30, )}
        target_model = models[num_pos]
        target_model.fit(X_lrn, y_lrn)
        target_pred = target_model.predict(test_lrn)
        print(f'best_score: {target_model.best_score_}')
        return target_pred

    def get_names(self, prefix, idx_group, idx_attribute):
        name_group_idx = self.cat_groups[idx_group]
        name_attributes = [f'{name_group_idx}_{prefix}_median',
                           f'{name_group_idx}_{prefix}_mean',
                           f'{name_group_idx}_{prefix}_med_mean']
        return name_group_idx, name_attributes[idx_attribute]

    def transform(self, df_in, clusters=0, dummy_cols=[], exclude_cols=[],
                  idx_grp=0, idx_attr=0):
        """
        Трансформация данных
        :type df_in: входной ДФ
        :param clusters: делить данные на количество кластеров
        :param dummy_cols: категорийные колонки для преобразования в признаки
        :param exclude_cols: колонки не участвующие в обработке
        :param idx_grp: индекс признака для группировки
        :param idx_attr: индекс атрибута для группировки
        :return: ДФ
        """
        if idx_grp not in range(len(self.cat_groups)):
            idx_grp = 0
        if idx_attr not in range(3):
            idx_attr = 0
        self.dummy = dummy_cols
        df = df_in.copy(deep=True)

        # небольшой препроцессинг, который нужен в двух методах fit и transform
        df = self.preprocess_df(df)

        # Target encoding
        # с ним стало хуже
        df = self.make_target_encoding(df)

        cond = df.YearsInCurrentJob.isna()
        # # YearsInCurrentJob заполняем модой
        # df.loc[cond, 'YearsInCurrentJob'] = self.YearsInCurrentJob
        # self.comment.append(f'YearsInCurrentJob=moda, '
        #                     f'bins={self.num_history_bins}')
        # YearsInCurrentJob заполняем медианой по группам кредитной истории
        df.loc[cond, 'YearsInCurrentJob'] = df[cond]['history_group'].map(
            self.med_history_job)
        self.comment.append(f'YearsInCurrentJob=med_history_job, '
                            f'bins={self.num_history_bins}')

        # Bankruptcies заполняем нулями
        df.loc[df.Bankruptcies.isna(), 'Bankruptcies'] = 0

        # отметка, что эти поля изначально были заполнены
        for column in self.features_good:
            df[column] = 1

        # посмотреть на каком усреднении будет выше F1_score
        # ['Purpose', 'debt_group', 'rest_bal_group', 'NumberOfCreditProblems']
        # [median, mean, med_mean]
        # подбор этих параметров

        # # MonthsSinceLastDelinquent заполняем нулями
        cond = df.MonthsSinceLastDelinquent.isna()
        # df.loc[cond, 'MonthsSinceLastDelinquent'] = 0
        # self.comment.append('MonthsSinceLastDelinquent = 0')
        # # Попробовать заполнить min, max, mean, median, mode - хуже "0"
        # df.loc[cond, 'MonthsSinceLastDelinquent'] =
        # df.MonthsSinceLastDelinquent.mode()
        # self.comment.append('MonthsSinceLastDelinquent = mode')
        # # MonthsSinceLastDelinquent - заполним медианным значением по целям
        # # и сроку кредита
        # name_group, name_attribute = self.get_names('mld', idx_grp, idx_attr)
        name_group, name_attribute = self.get_names('mld', 2, 0)
        df.loc[cond, 'MonthsSinceLastDelinquent'] = df[cond][name_group].map(
            getattr(DataProcessing, name_attribute))
        self.comment.append(name_attribute)
        # # если не получилось заполнить по группировке - заполним медианой
        df.MonthsSinceLastDelinquent.fillna(
            df.MonthsSinceLastDelinquent.median(), inplace=True)

        # calc_months - заполним медианным значением по целям и сроку кредита
        # по calc_months посчитать AnnualIncome = calc_months * MonthlyDebt
        cond = df.calc_months.isna()
        df.loc[cond, 'calc_monthsIsGood'] = 0
        # name_group, name_attribute = self.get_names('cm', idx_grp, idx_attr)
        name_group, name_attribute = self.get_names('cm', 0, 0)
        df.loc[cond, 'calc_months'] = df[cond][name_group].map(
            getattr(DataProcessing, name_attribute))
        self.comment.append(name_attribute)
        # если не получилось заполнить по группировке - заполним медианой
        df.calc_months.fillna(df.calc_months.median(), inplace=True)

        # AnnualIncome - заполним медианным доходом по группам долга
        cond = df.AnnualIncome.isna()
        df.loc[cond, 'AnnualIncomeIsGood'] = 0
        # df.loc[cond, 'AnnualIncome'] = df[cond].debt_group.map(
        #     self.med_debt_income)
        # name_group, name_attribute = self.get_names('ai', idx_grp, idx_attr)
        name_group, name_attribute = self.get_names('ai', 1, 0)
        df.loc[cond, 'AnnualIncome'] = df[cond][name_group].map(
            getattr(DataProcessing, name_attribute))
        self.comment.append(name_attribute)
        # по calc_months посчитать AnnualIncome = calc_months * MonthlyDebt
        # df.loc[cond, 'AnnualIncome'] = df[cond].calc_months * df[
        #     cond].MonthlyDebt
        # self.comment.append('ai = cm * MonthlyDebt')
        # если не получилось заполнить по группировке - заполним медианой
        df.AnnualIncome.fillna(df.AnnualIncome.median(), inplace=True)

        # CurrentLoanAmount замена значений 99999999 - теперь они NaN
        cond = df.CurrentLoanAmount.isna()
        df.loc[cond, 'CurrentLoanAmountIsGood'] = 0
        # name_group, name_attribute = self.get_names('cla', idx_grp, idx_attr)
        name_group, name_attribute = self.get_names('cla', 0, 2)
        df.loc[cond, 'CurrentLoanAmount'] = df[cond][name_group].map(
            getattr(DataProcessing, name_attribute))
        self.comment.append(name_attribute)
        # если не получилось заполнить по группировке - заполним медианой
        df.CurrentLoanAmount.fillna(df.CurrentLoanAmount.median(),
                                    inplace=True)

        # как-то нужно заполнить пропуски в CreditScore
        # простой вариант - максимальный рейтинг
        # df.CreditScore.fillna(999, inplace=True)
        # второй вариант - медиана
        # df.CreditScore.fillna(df.CreditScore.median(), inplace=True)
        # третий вариант - мадиана по группам NumberOfCreditProblems
        cond = df.CreditScore.isna()
        df.loc[cond, 'CreditScoreIsGood'] = 0
        # df.loc[cond, 'CreditScore'] = df[cond].NumberOfCreditProblems.map(
        #     self.med_problems_score)
        # name_group, name_attribute = self.get_names('cs', idx_grp, idx_attr)
        name_group, name_attribute = self.get_names('cs', 3, 0)
        df.loc[cond, 'CreditScore'] = df[cond][name_group].map(
            getattr(DataProcessing, name_attribute))
        self.comment.append(name_attribute)
        # если не получилось заполнить по группировке - заполним медианой
        df.CreditScore.fillna(df.CreditScore.median(), inplace=True)

        # сохранение ДФ для отдельного обучения моделек на заполнение пропусков
        self.write_dataset(df, os.path.join(PATH_EXPORT, 'to_learn_all.csv'))

        # # заполнение таргет-енкодингд для моделек
        # num_pos = 2
        # pref = ['ai', 'cla', 'cs'][num_pos]
        # for g_id in range(len(self.cat_groups)):
        #     for a_id in range(3):
        #         name_group, name_attribute = self.get_names(pref, g_id, a_id)
        #         df[name_attribute] = df[name_group].map(
        #             getattr(DataProcessing, name_attribute))
        #         self.exclude_columns.append(name_attribute)
        #         cond = df[name_attribute].isna()
        #         df.loc[cond, name_attribute] = df.loc[cond, 'AnnualIncome']
        # #
        # # датасет для моделек по AnnualIncome, CurrentLoanAmount, CreditScore
        # self.write_dataset(df, os.path.join(PATH_EXPORT, 'to_learn_all.csv'))
        # #
        # # колонка с метками пропущенных значений
        # isna_column = self.features_good[num_pos]
        # cond = df[isna_column] < 1
        # #
        # # предсказание на модели дает худший результат, чем группировка
        # df.loc[cond, 'AnnualIncome'] = self.fill_ai_cla_cs(df, num_pos)
        # # предсказание на модели дает худший результат, чем группировка
        # df.loc[cond, 'CurrentLoanAmount'] = self.fill_ai_cla_cs(df, num_pos)
        # # предсказание на модели дает худший результат, чем группировка
        # df.loc[cond, 'CreditScore'] = self.fill_ai_cla_cs(df, num_pos)
        # df.CreditScore = df.CreditScore.astype(int)

        # на всякий случай, если что-то пошло не так
        df.fillna(self.medians, inplace=True)

        # Добавление новых признаков
        df = self.new_features(df)

        if clusters:
            cluster_labels = self.make_clusters(df, clusters)
            df = pd.concat([df, cluster_labels], axis=1)

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

        df.loc[df.Learn == 0, 'CreditDefault'] = np.NaN
        return df


def make_model(idx_grp, idx_attr, target_enc_feat='', multy_feature='',
               clusters=0, es_groups=[], num_history_bins=7):
    """
    Построение модели
    :param num_history_bins: делить кредитную историю на кол-во частей
    :param es_groups: список признаков для добавления новых фич
    :param idx_grp: индекс группы ['grp_purpose', 'debt_group',
           'rest_bal_group', 'NumberOfCreditProblems', 'HomeOwnership']
    :param idx_attr: индекс метрики [median, mean, med_mean]
    :param clusters: делить данные на количество кластеров
    :param target_enc_feat: колонка с таргетенкодингом
    :param multy_feature: колонка на которую умножается target_enc_feature
    :return: None
    """
    global processor_data, dataset, model_columns, test_df, X, y
    global X_train, X_valid, y_train, y_valid, category_columns, learn_exclude
    model_columns = category_columns = learn_exclude = []

    # обучающая выборка
    train = pd.read_csv(FILE_TRAIN)
    # тестовая выборка
    test = pd.read_csv(FILE_TEST)

    processor_data = DataProcessing()
    # удобнее работать с одним ДФ: отображать, сохранять, читать
    dataset = processor_data.concat_df(train, test)

    # for col in dataset.select_dtypes(include='float64').columns:
    #     print(f'Признак: {col}')
    #     delta = 0.001
    #     min_quant = dataset[col].quantile(delta)
    #     max_quant = dataset[col].quantile(1 - delta)
    #     print(f'< квантиля {delta:0.1%} кол-во значений: '
    #           f'{len(dataset[dataset[col] < min_quant])}, '
    #           f'> квантиля {1 - delta:0.1%} кол-во значений: '
    #           f'{len(dataset[dataset[col] > max_quant])}')

    print(f'Обработка данных')
    start_time = time.time()
    # статистики по всему датасету
    processor_data.fit(dataset, num_debt_bins=7,
                       num_history_bins=num_history_bins)
    # # статистики только на трейне
    # processor_data.fit(dataset[dataset.Learn == 1], num_debt_bins=7,
    #                    num_history_bins=num_history_bins)

    exclude_columns = ['HomeOwnership', 'grp_purpose', 'debt_group',
                       'rest_bal_debt', 'rest_bal_group',
                       'history_group'
                       ]

    # exclude_columns.extend(['calc_months', 'purpose_term'])
    exclude_columns.extend(['purpose_term'])
    # exclude_columns.extend(['calc_months'])

    dataset = processor_data.transform(dataset, clusters=clusters,
                                       exclude_cols=exclude_columns,
                                       dummy_cols=[],
                                       idx_grp=idx_grp,
                                       idx_attr=idx_attr)
    processor_data.write_dataset(dataset, FILE_WITH_FEATURES)
    print_time(start_time)
    print('Пропуски заполнены по группировкам: ', processor_data.comment)
    dataset = memory_compression(dataset)

    # если есть пустые значения - выведем на экран
    if dataset.drop(['CreditDefault'], axis=1).isna().sum().sum() > 0:
        print(dataset.drop(['CreditDefault'], axis=1).isna().sum())

    cat_features = processor_data.cat_features
    # cat_features.append('history_group')
    # все колонки для обучения
    learn_columns = processor_data.learn_columns

    print('CatGroups:', processor_data.cat_groups)
    print('TargetEnc:', processor_data.target_encoding_feats)

    # эти колонки исключаем из обучения
    # add_exclude = []
    add_exclude = TargetEnc
    # колонки с отметками о заполненности полей
    add_exclude.extend(processor_data.features_good)

    # уберем более крупные группировки
    learn_exclude = []
    learn_exclude.extend(processor_data.exclude_columns)
    learn_exclude.extend(add_exclude)
    # если это убрать результат становится хуже
    # learn_exclude.append('MonthsSinceLastDelinquent')
    print('Исключаем:', learn_exclude)

    # колонки для обучения
    model_columns = [col for col in learn_columns if
                     col not in learn_exclude]
    category_columns = [col for col in cat_features if
                        col in model_columns]

    if target_enc_feat in learn_columns:
        if multy_feature:
            new_feat = f'{target_enc_feat}_mult_CreditScore'
            dataset[new_feat] = dataset[target_enc_feat] * dataset.CreditScore
        else:
            new_feat = target_enc_feat
        model_columns.insert(0, new_feat)
        print('Добавляем:', new_feat)
    elif target_enc_feat == 'AllTargetEnc':
        model_columns.extend(processor_data.target_encoding_feats)

    print('Обучаемся:', model_columns)
    print('Категории:', category_columns)

    # print(dataset.info())

    if len(es_groups):
        print(f'Генерация новых признаков по {es_groups}')
        start_time = time.time()

        agg_primitives = ['median', 'mode', 'num_unique', 'mean', 'sum',
                          'percent_true', 'count', 'std']

        processor_data.comment.append({'featuretools': (es_groups,
                                                        agg_primitives)})

        # creating and entity set 'es'
        all_cat_groups = processor_data.cat_groups
        es = ft.EntitySet(id='Credits')
        es_cat_cols = [col for col in all_cat_groups if
                       col not in model_columns]
        # добавим колонку с индексом
        dataset.insert(0, 'ID', dataset.index)
        variable_types = {col: vtypes.Categorical for col in all_cat_groups if
                          col != 'NumberOfCreditProblems'}
        es_dataset_cols = ['Learn'] + es_cat_cols + model_columns
        # print(es_dataset_cols)
        # добавим колонки с target_encoding
        # es_dataset_cols.extend(processor_data.target_encoding_feats)
        # print(es_dataset_cols)
        es.entity_from_dataframe(entity_id='Clients',
                                 dataframe=dataset[es_dataset_cols],
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
                     '.Learn' in col or '.CreditDefault' in col]
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
        model_columns = feats_matrix_enc.columns.to_list()[1:]
        dataset = feats_matrix_enc.copy(deep=True)
        for col in feats_matrix_enc.columns.to_list()[1:]:
            if dataset[col].isna().sum() > 0 and col != 'CreditDefault':
                print(f'Пропуски в {col} = {dataset[col].isna().sum()}')
                model_columns.remove(col)
                # print(dataset[col].describe())
        with open(os.path.join(PATH_EXPORT, 'model_cols.csv'), 'w') as mc:
            for col in model_columns:
                mc.write(f'{col}\n')

    # обучающий датасет
    train_df = dataset[dataset.Learn == 1][model_columns]
    # тестовый датасет
    test_df = dataset[dataset.Learn == 0][model_columns]
    test_df.drop('CreditDefault', axis=1, inplace=True)

    X = train_df.drop('CreditDefault', axis=1)
    y = train_df['CreditDefault'].astype('int')

    txt = ('Размер ', ' пропусков ')
    print(f'{txt[0]}трейна: {X.shape}{txt[1]}{X.isna().sum().sum()}')
    print(f'{txt[0]}теста: {test_df.shape}{txt[1]}'
          f'{test_df.isna().sum().sum()}')

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.3,
                                                          shuffle=True,
                                                          random_state=SEED,
                                                          stratify=y
                                                          )
    print()
    print(f'{txt[0]}X_train: {X_train.shape}{txt[1]}'
          f'{X_train.isna().sum().sum()}')
    print(f'{txt[0]}X_valid: {X_valid.shape}{txt[1]}'
          f'{X_valid.isna().sum().sum()}')

    # X_train, X_valid, y_train, y_valid = X, X, y, y

    imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f'Дисбаланс классов = {imbalance}')
    print(f'Общий дисбаланс = {y.value_counts()[0] / y.value_counts()[1]}')

    # немного потюним и результат грузим на Kaggle
    feat_imp_df_ = pd.DataFrame
    params = {
        # 'iterations': range(50, 901, 50),
        'iterations': range(10, 201, 10),
        'max_depth': range(5, 7),
        # 'learning_rate': [.005, .01, .025, .05]
    }
    # поставил общий дисбаланс попробовать это на Каггле - результат хуже
    # imbalance = y.value_counts()[0] / y.value_counts()[1]

    model = CatBoostClassifier(silent=True, random_state=SEED,
                               class_weights=[1, imbalance],
                               cat_features=category_columns,
                               eval_metric='F1',
                               early_stopping_rounds=30,
                               )

    feat_imp_df_ = process_model(model, params=params, fold_single=5,
                                 verbose=1, build_model=True)
    print(feat_imp_df_)


if __name__ == "__main__":

    total_time = time.time()

    tmp_class = DataProcessing()
    tmp_group = ['median', 'mean', 'med_mean']
    # ['grp_purpose', 'debt_group', 'rest_bal_group',
    #  'NumberOfCreditProblems', 'HomeOwnership', 'purpose_term']
    range_grp_idx = range(len(tmp_class.cat_groups))
    # range_grp_idx = [0, 6]
    # [median, mean, med_mean]
    range_attr_idx = range(3)
    # range_attr_idx = [0]
    # перебор колонок с группировкой по метрикам - подобрал нужные метрики
    # for grp_idx in range_grp_idx:
    #     for attr_idx in range_attr_idx:
    #         print(f'Группировка по {tmp_class.cat_groups[grp_idx]} '
    #               f'метрика {tmp_group[attr_idx]}')
    #         make_model(grp_idx, attr_idx)

    # # перебор колонок с таргетенкодингом - не помогло
    # for target_enc in TargetEnc:
    #     # добавление колонки с таргет_енкодингом
    #     make_model(0, 0, target_enc)
    #     # добавление колонки с таргет_енкодингом * CreditScore
    #     make_model(0, 0, target_enc, 'CreditScore')

    # # перебор колонок с кластеризацией - не помогло
    # for cluster in range(2, 33):
    #     make_model(0, 0, clusters=cluster)

    # # генерация новых фич с featuretools
    # for num_new_groups in range(1, len(tmp_class.cat_groups) + 1):
    #     for group_es in combinations(tmp_class.cat_groups, num_new_groups):
    #         make_model(0, 0, es_groups=group_es)

    # подбор на сколько частей делить кредитную историю для группировки = 7
    # for num in range(3, 24):
    #     make_model(0, 0, num_history_bins=num)

    make_model(0, 0)

    # cb_submit_210331201626.csv Private / Public : 0.66025 / 0.65404
    # cb_submit_210331155924.csv Private / Public : 0.65473 / 0.62702
    # cb_submit_210331160809.csv Private / Public : 0.65239 / 0.64625

    print_time(total_time)
