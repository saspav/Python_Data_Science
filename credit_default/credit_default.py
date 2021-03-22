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
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import mannwhitneyu

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


def process_model(use_model=RandomForestClassifier(random_state=SEED),
                  params={'max_depth': [7]}, folds_range=[], fold_single=5,
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
            gscv = model.grid_search(params, X_train, y_train, cv=skf,
                                     stratified=True, refit=True)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_valid_pred = model.predict(X_valid)
            best_ = gscv['params']
            f1_train = np.array(gscv['cv_results']['train-Logloss-mean']).max()
            f1_valid = np.array(gscv['cv_results']['test-Logloss-mean']).max()
            gscv = model
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
        date_now = datetime.now()
        time_stamp = date_now.strftime('%y%m%d%H%M')
        submit.to_csv(file_submit_csv.replace('.csv', f'_{time_stamp}.csv'))
        # сохранение результатов итерации в файл
        file_name = os.path.join(PATH_EXPORT, 'results.csv')
        if os.path.exists(file_name):
            file_df = pd.read_csv(file_name)
            file_df.time_stamp = pd.to_datetime(file_df.time_stamp,
                                                format='%y-%m-%d %H:%M:%S')
            file_df.time_stamp = file_df.time_stamp.dt.strftime(
                '%y-%m-%d %H:%M:%S')
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
                                'best_params': [prms],
                                'features': [features_list],
                                'column_dummies': [processor_data.dummy],
                                'model_columns': [model_columns],
                                'category_columns': [category_columns],
                                'learn_exclude': [learn_exclude]
                                })

        file_df = file_df.append(temp_df)
        file_df.f1_train = file_df.f1_train.round(7)
        file_df.f1_valid = file_df.f1_valid.round(7)
        file_df.to_csv(file_name, index=False)
        file_df.name = 'results'
        # экспорт в эксель
        export_to_excel(file_df)
        print_time(start_time_cv)
        return feat_imp_df
    else:
        return best_folds


def find_depth(use_model, max_depth_values=range(3, 11), not_sklearn=False,
               show_plot=True):
    print(use_model)
    # Подберем оптимальное значение глубины обучения дерева.
    scores = pd.DataFrame(columns=['max_depth', 'train_score', 'valid_score'])
    for max_depth in max_depth_values:
        print(f'max_depth = {max_depth}')
        if not_sklearn:
            find_model = use_model(random_state=SEED, max_depth=max_depth,
                                   num_leaves=63)
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
                     data=scores_data)
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
    min = time_apply % 3600
    sec = min % 60
    print(f'Время обработки: {hrs:.0f} час {min // 60:.0f} мин {sec:.1f} сек')


def evaluate_preds(mdl, X_train, X_test, y_train, y_test):
    y_train_pred = mdl.predict(X_train)
    y_test_pred = mdl.predict(X_test)
    show_classification_report(y_train, y_train_pred, y_test, y_test_pred)


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
    name_data = data.name
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
                       'r2_train', 'r2_valid'):
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
        df.loc[df.TaxLiens > 0, 'TaxLiens'] = 1
        df.loc[df.NumberOfCreditProblems > 0, 'NumberOfCreditProblems'] = 1

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
        self.YearsInCurrentJob = None
        self.bin_years = {'< 1 year': 0, '1 year': 1, '2 years': 2,
                          '3 years': 3, '4 years': 4, '5 years': 5,
                          '6 years': 6, '7 years': 7, '8 years': 8,
                          '9 years': 9, '10+ years': 10}
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.num_debt_bins = 7
        self.debt_bins = None
        self.debt_labels = None
        self.med_debt_income = None
        self.med_problems_score = None
        self.dummy = []
        self.cat_features = []
        self.exclude_columns = ['Learn', 'debt_group']
        self.learn_columns = []

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
        df = df_in.copy(deep=True)

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

        # # добавление разницы между балансом и долгом
        # df['rest_bal_debt'] = df.CurrentCreditBalance - df.MonthlyDebt
        # # возвращается кортеж:
        # _, rest_bins = pd.qcut(df.rest_bal_debt, q=self.num_debt_bins,
        #                        precision=0, retbins=True)
        # rest_bins[0] = -np.inf
        # rest_bins[-1] = np.inf
        # df['rest_bal_group'] = pd.cut(df.rest_bal_debt, bins=rest_bins,
        #                               labels=self.debt_labels).astype('str')
        return df

    @staticmethod
    def new_features(df_in):
        df = df_in.copy(deep=True)

        # Добавление новых признаков:
        df.Term = df.Term.map({'Short Term': 0, 'Long Term': 1})
        df.loc[df.Bankruptcies > 0, 'Bankruptcies'] = 1
        # сгруппируем цели: три большие группы в одну и остальные в другую
        df['grp_purpose'] = df.Purpose
        purpose = ['debt consolidation', 'other', 'home improvements']
        df.loc[df.Purpose.isin(purpose), 'Purpose'] = 0
        df.loc[df.Purpose != 0, 'Purpose'] = 1
        df.Purpose = df.Purpose.astype(int)
        return df

    def fit(self, df_in, num_debt_bins=7):
        """Сохранение статистик"""
        df = df_in.copy(deep=True)

        if num_debt_bins > 3:
            self.num_debt_bins = num_debt_bins
        # возвращается кортеж:
        _, self.debt_bins = pd.qcut(df.MonthlyDebt, q=self.num_debt_bins,
                                    precision=0, retbins=True)
        self.debt_bins[0] = -np.inf
        self.debt_bins[-1] = np.inf
        self.debt_labels = [self.alphabet[i] for i in range(num_debt_bins)]

        # небольшой препроцессинг, который нужен в двух методах fit и transform
        df = self.preprocess_df(df)

        # Расчет медиан
        self.medians = df.median()

        # медианный AnnualIncome по группам debt_group
        self.med_debt_income = df.groupby('debt_group').agg(
            {'AnnualIncome': 'median'}).to_dict()['AnnualIncome']
        # print(self.med_debt_income)

        # медианный CreditScore по группам NumberOfCreditProblems
        self.med_problems_score = df.groupby('NumberOfCreditProblems').agg(
            {'CreditScore': 'median'}).astype('int').to_dict()['CreditScore']
        self.med_problems_score[3] -= 10
        # print(self.med_problems_score)

        self.YearsInCurrentJob = df.YearsInCurrentJob.mode()[0]

        features_isna = {'AnnualIncome': ~df.AnnualIncome.isna(),
                         'CurrentLoanAmount': df.CurrentLoanAmount < 99999999,
                         'CreditScore': ~df.CreditScore.isna()}
        for name_feature, cond in features_isna.items():
            print(name_feature)
            # группировка признака по группам Purpose и debt_group
            for col_groupby in ('Purpose', 'debt_group',
                                'NumberOfCreditProblems'):
                grp_feat = df[cond].groupby(col_groupby, as_index=False).agg(
                    {name_feature: ['median', 'mean']})
                grp_feat.columns = [col_groupby, 'feat_median', 'feat_mean']
                grp_feat['feat_med_mean'] = grp_feat.mean(axis=1)
                grp_feat = grp_feat.round(0)
                grp_feat = grp_feat.set_index(col_groupby)
                for attr in ('feat_median', 'feat_mean', 'feat_med_mean'):
                    name_atr = f'{col_groupby}_{attr}'
                    name_grp = grp_feat[attr]
                    name_grp.columns = [name_feature]
                    name_grp = name_grp.to_dict()
                    setattr(DataProcessing, name_atr, name_grp)
                    print(name_atr)
                    print(name_grp)

    @staticmethod
    def fill_ai_cla_cs(df_in, num_pos, indexes_isna):
        """
        Заполнение 'AnnualIncome', 'CurrentLoanAmount', 'CreditScore'
        на основе предсказаний модели
        :param df_in: входной ДФ
        :param num_pos: порядковый номер колонки для предсказаний
        :param indexes_isna: индексы пропущенных значений
        :return: ДФ
        """
        df_lrn = df_in.copy(deep=True)

        # эти колонки исключаем из обучения
        excld_cols = ['AnnualIncome', 'CurrentLoanAmount', 'CreditScore']
        group_cols = ['HomeOwnership', 'grp_purpose', 'debt_group']
        # колонка для предсказаний
        target_column = excld_cols[num_pos]
        dummy_dict = {0: ['HomeOwnership', 'grp_purpose'],
                      1: ['grp_purpose'],
                      2: ['HomeOwnership', 'grp_purpose', 'debt_group']}
        models = {0: LGBMRegressor(learning_rate=0.075, max_depth=3,
                                   n_estimators=140, num_leaves=63,
                                   random_state=SEED),
                  1: LGBMRegressor(learning_rate=0.05, max_depth=3,
                                   n_estimators=180, num_leaves=63,
                                   random_state=SEED),
                  2: LGBMRegressor(learning_rate=0.1, max_depth=3,
                                   n_estimators=140, num_leaves=63,
                                   random_state=SEED)}
        if num_pos < 2:
            df_lrn.loc[df_lrn.CurrentLoanAmount >= 99999999,
                       'CurrentLoanAmount'] = np.NaN

        features_gen = NewTargetFeature()
        features_gen.fit(indexes_isna)
        df_lrn = features_gen.transform(df_lrn, exclude_cols=group_cols,
                                        dummy_cols=dummy_dict[num_pos])
        df_lrn = memory_compression(df_lrn)

        # все колонки для обучения
        learn_cols = features_gen.learn_columns
        # уберем более крупные группировки
        lrn_exclude = []
        lrn_exclude.extend(features_gen.exclude_columns)
        lrn_exclude.extend(excld_cols[num_pos:])
        # колонки для обучения
        mdl_columns = [col for col in learn_cols if col not in lrn_exclude]
        mdl_columns.append(target_column)

        train_lrn = df_lrn[df_lrn.Learn == 1]

        # обучающий датасет
        train_lrn = train_lrn[~train_lrn[target_column].isna()][mdl_columns]
        # тестовый датасет
        test_lrn = df_lrn[indexes_isna][mdl_columns]
        test_lrn.drop(target_column, axis=1, inplace=True)

        X_lrn = train_lrn.drop(target_column, axis=1)
        y_lrn = train_lrn[target_column]

        target_model = models[num_pos]
        target_model.fit(X_lrn, y_lrn)
        target_pred = target_model.predict(test_lrn)
        return target_pred

    def transform(self, df_in, dummy_cols=[], exclude_cols=[]):
        """
        Трансформация данных
        :type df_in: входной ДФ
        :param dummy_cols: категорийные колонки для преобразования в признаки
        :param exclude_cols: колонки не участвующие в обработке
        :return: ДФ
        """
        self.dummy = dummy_cols
        df = df_in.copy(deep=True)

        # небольшой препроцессинг, который нужен в двух методах fit и transform
        df = self.preprocess_df(df)

        # YearsInCurrentJob заполняем модой
        df.loc[df.YearsInCurrentJob.isna(),
               'YearsInCurrentJob'] = self.YearsInCurrentJob
        # YearsInCurrentJob преобразуем в числовые значения
        df.YearsInCurrentJob = df.YearsInCurrentJob.map(self.bin_years)

        # Bankruptcies заполняем нулями
        df.loc[df.Bankruptcies.isna(), 'Bankruptcies'] = 0

        # MonthsSinceLastDelinquent заполняем нулями
        df.loc[df.MonthsSinceLastDelinquent.isna(),
               'MonthsSinceLastDelinquent'] = 0

        # датасет для моделей по AnnualIncome, CurrentLoanAmount, CreditScore
        self.write_dataset(df, os.path.join(PATH_EXPORT, 'to_learn_ai.csv'))

        # AnnualIncome - заполним медианным доходом по группам долга
        cond = df.AnnualIncome.isna()
        df.loc[cond, 'AnnualIncome'] = df[cond].debt_group.map(
            self.med_debt_income)
        # предсказание на модели дает худший результат, чем группировка
        # df.loc[cond, 'AnnualIncome'] = self.fill_ai_cla_cs(df, 0, cond)
        self.write_dataset(df, os.path.join(PATH_EXPORT, 'to_learn_cla.csv'))

        # CurrentLoanAmount замена значений 99999999
        # посмотреть на каком усреднении будет выше F1_score
        cond = df.CurrentLoanAmount >= 99999999
        # CurrentLoanAmount замена значений 99999999 на медиану по Purpose
        # name_atr = 'Purpose_feat_median'
        # name_atr = 'Purpose_feat_mean'
        name_atr = 'Purpose_feat_med_mean'
        # CurrentLoanAmount замена значений 99999999 на медиану по debt_group
        # name_atr = 'debt_group_feat_median'
        # name_atr = 'debt_group_feat_mean'
        # name_atr = 'debt_group_feat_med_mean'
        df.loc[cond, 'CurrentLoanAmount'] = df[cond].Purpose.map(
            getattr(DataProcessing, name_atr))
        # print(df[cond][['Purpose', 'CurrentLoanAmount']])
        # print(df[cond][['debt_group', 'CurrentLoanAmount']])
        # если не получилось заполнить по группировке - заполним медианой
        df.loc[df.CurrentLoanAmount >= 99999999, 'CurrentLoanAmount'] = np.NaN
        df.CurrentLoanAmount.fillna(df.CurrentLoanAmount.median(),
                                    inplace=True)
        # предсказание на модели дает худший результат, чем группировка
        # df.loc[cond, 'CurrentLoanAmount'] = self.fill_ai_cla_cs(df, 1, cond)
        self.write_dataset(df, os.path.join(PATH_EXPORT, 'to_learn_cs.csv'))

        # как-то нужно заполнить пропуски в CreditScore
        # простой вариант - максимальный рейтинг
        # df.CreditScore.fillna(999, inplace=True)
        # второй вариант - медиана
        # df.CreditScore.fillna(df.CreditScore.median(), inplace=True)
        # третий вариант - мадиана по группам NumberOfCreditProblems
        cond = df.CreditScore.isna()
        df.loc[cond, 'CreditScore'] = df[cond].NumberOfCreditProblems.map(
            self.med_problems_score)
        # предсказание на модели дает худший результат, чем группировка
        # df.loc[cond, 'CreditScore'] = self.fill_ai_cla_cs(df, 2, cond)
        df.CreditScore = df.CreditScore.astype(int)

        # на всякий случай, если что-то пошло не так
        df.fillna(self.medians, inplace=True)

        # Добавление новых признаков
        df = self.new_features(df)
        df.loc[df.TaxLiens > 0, 'TaxLiens'] = 1
        df.loc[df.NumberOfCreditProblems > 0, 'NumberOfCreditProblems'] = 1

        # деление категорий по столбцам
        if self.dummy:
            df_dummy = pd.get_dummies(df[self.dummy], columns=self.dummy)
            df = pd.concat([df, df_dummy], axis=1)
            # self.cat_features.extend(df_dummy.columns.values)
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


if __name__ == "__main__":
    # обучающая выборка
    train = pd.read_csv(FILE_TRAIN)
    # тестовая выборка
    test = pd.read_csv(FILE_TEST)

    processor_data = DataProcessing()
    dataset = processor_data.concat_df(train, test)

    # NUM_FEATURE_NAMES = dataset.select_dtypes(
    #     include='float64').columns.values.tolist() + dataset.select_dtypes(
    #     include='int32').columns.values.tolist()
    # NUM_FEATURE_NAMES.remove('CreditDefault')
    # print(NUM_FEATURE_NAMES)
    # for col in NUM_FEATURE_NAMES:
    #     plt.figure(figsize=(8, 4))
    #     sns.kdeplot(data=dataset, x=col, shade=True, hue='Learn',
    #                 hue_order=[1, 0])
    #     print(col)
    #     print(mannwhitneyu(dataset[dataset.Learn == 1][col],
    #                        dataset[dataset.Learn == 0][col]))
    #     plt.title(col)
    #     plt.show()

    # CAT_FEATURE_NAMES = dataset.select_dtypes(
    #     include='object').columns.values.tolist()
    # print(CAT_FEATURE_NAMES)
    # num_feature = 'CurrentCreditBalance'
    # for col in CAT_FEATURE_NAMES:
    #     sns.set(font_scale=1.1)
    #     fig, ax = plt.subplots(figsize=(16, 6))
    #     sns.pointplot(data=dataset, x=col, y=num_feature, capsize=.1,
    #                   shade=True, hue='Learn', hue_order=[1, 0])
    #     plt.title(col)
    #     plt.setp(ax.get_xticklabels(), rotation=90)
    #     plt.tight_layout()
    #     plt.show()

    print(f'Обработка данных')
    start_time = time.time()
    processor_data.fit(dataset, num_debt_bins=7)
    exclude_columns = ['HomeOwnership', 'grp_purpose']
    dataset = processor_data.transform(dataset,
                                       exclude_cols=exclude_columns)
    processor_data.write_dataset(dataset, FILE_WITH_FEATURES)
    print_time(start_time)

    dataset = memory_compression(dataset)

    # если есть пустые значения - выведем на экран
    if dataset.drop(['CreditDefault'], axis=1).isna().sum().sum() > 0:
        print(dataset.drop(['CreditDefault'], axis=1).isna().sum())

    cat_features = processor_data.cat_features
    # все колонки для обучения
    learn_columns = processor_data.learn_columns

    # эти колонки исключаем из обучения
    add_exclude = []

    # уберем более крупные группировки
    learn_exclude = []
    learn_exclude.extend(processor_data.exclude_columns)
    learn_exclude.extend(add_exclude)
    print(learn_exclude)

    # колонки для обучения
    model_columns = [col for col in learn_columns if col not in learn_exclude]
    category_columns = [col for col in cat_features if col in model_columns]
    print(model_columns)
    print(category_columns)

    print(dataset.info())

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

    # было test_size=0.2
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.2,
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

    # определение моделей
    model = RandomForestClassifier(random_state=SEED)
    # mdl = ExtraTreesClassifier(random_state=SEED)
    # mdl = GradientBoostingClassifier(random_state=SEED, criterion='mse')
    # mdl = LGBMClassifier(random_state=SEED, num_leaves=63)
    # mdl = CatBoostClassifier(random_state=SEED, loss_function='RMSE',
    #                           silent=True, cat_features=category_columns)

    # find_depth(RandomForestClassifier)
    # find_depth(ExtraTreesClassifier)
    # find_depth(GradientBoostingClassifier)
    # find_depth(LGBMClassifier, True)

    # настройки для первого приближения: поиск глубины деревьев и количества фолдов
    # f_params = {'max_depth': list(range(4, 15))}
    f_params = {'max_depth': list(range(4, 9))}
    # раскомментарить эту строку для расчета
    # process_model(mdl, params=f_params, folds_range=list(range(3, 8)))

    # models = []
    # for depth in range(3, 11):
    #     param = {'max_depth': [depth]}
    #     mdl = process_model(mdl, params=param, folds_range=list(range(3, 6)))
    #     models.append(mdl[0][:4])
    # models.sort(key=lambda x: (-x[1], x[2]))
    # print()
    # for elem in models:
    #     print(elem)

    # Зададим параметры при max_depth = 18 для подбора параметров
    # и отдыхаем несколько часов
    f_params = {'n_estimators': list(range(100, 701, 100)),
                'max_depth': [5],
                # 'min_samples_leaf': list(range(1, 11, 1)),
                # 'min_samples_split': list(range(2, 22, 1)),  # не меньше 2
                }
    # раскомментарить эту строку для расчета
    # process_model(mdl, params=f_params, folds_range=[7], verbose=1)

    # немного потюним и результат грузим на Kaggle
    feat_imp_df_ = pd.DataFrame

    # # GradientBoostingRegressor
    # f_params = {
    #     # 'n_estimators': [1100],
    #     'n_estimators': list(range(50, 151, 50)),
    #     'max_depth': [4],
    #     # 'learning_rate': [.05]
    #     'learning_rate': [.005, .01, .025, .05]
    #     # 'min_samples_leaf': list(range(1, 9, 2)),
    #     # 'min_samples_split': list(range(2, 9, 2)),  # не меньше 2
    #     # 'min_samples_leaf': [3],
    #     # 'min_samples_split': [19]
    # }

    # f_params = {
    #     'boosting_type': ['dart'],
    #     # 'boosting_type': ['dart', 'gbdt'],
    #     # 'n_estimators': [1000],
    #     'n_estimators': list(range(900, 1301, 50)),
    #     # 'max_bin': [512],
    #     'max_depth': [5],
    #     # 'learning_rate': [.05]
    #     'learning_rate': [.005, .01, .025, .05, 0.1]
    #     # 'min_samples_leaf': list(range(1, 9, 2)),
    #     # 'min_samples_split': list(range(2, 9, 2)),  # не меньше 2
    #     # 'min_samples_leaf': [3],
    #     # 'min_samples_split': [19]
    # }
    # раскомментарить эту строку для расчета
    # _, feat_imp_df_ = process_model(mdl, params=f_params, fold_single=5,
    #                                 verbose=1, build_model=True)
    # print(feat_imp_df_)

    # model = CatBoostClassifier(silent=True, random_state=SEED,
    #                            class_weights=[1, imbalance],
    #                            eval_metric='F1')
    # model.fit(X_train, y_train)
    # evaluate_preds(model, X_train, X_valid, y_train, y_valid)

    params = {
        # 'iterations': [5, 7, 10, 20, 30, 50, 100],
        # 'max_depth': [3, 5, 7, 10],
        'max_depth': range(5, 6),
        'iterations': range(10, 151, 10),
        # 'learning_rate': [.005, .01, .025, .05]
    }
    # поставил общий дисбаланс попробовать это грузануть
    # imbalance = y.value_counts()[0] / y.value_counts()[1]

    model = CatBoostClassifier(silent=True, random_state=SEED,
                               class_weights=[1, imbalance],
                               cat_features=category_columns,
                               eval_metric='F1',
                               early_stopping_rounds=50, )

    feat_imp_df_ = process_model(model, params=params, fold_single=5,
                                 verbose=1, build_model=True)
    print(feat_imp_df_)

    # skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    # search_cv = model.grid_search(params, X_train, y_train, cv=skf,
    #                               stratified=True, refit=True)
    # for key, value in model.get_all_params().items():
    #     print(f'{key} : {value}'.format(key, value))
    #
    # a = model.get_all_params()['iterations']
    # b = model.get_all_params()['depth']
