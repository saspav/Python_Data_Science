import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score as r2
import warnings

warnings.filterwarnings("ignore")

SEED = 2021
PATH_FILES = r'D:\python-txt\real_estate'
PATH_EXPORT = r'D:\python-txt\real_estate'
FILE_TRAIN = os.path.join(PATH_FILES, 'train.csv')
FILE_TEST = os.path.join(PATH_FILES, 'test.csv')
FILE_YEARS = os.path.join(PATH_EXPORT, 'changed_years.csv')
FILE_HOUSES = os.path.join(PATH_EXPORT, 'moscow_houses.csv')
FILE_PREPROCESS = os.path.join(PATH_EXPORT, 'df_preprocess.csv')
FILE_WITH_FEATURES = os.path.join(PATH_EXPORT, 'df_all.csv')
FILE_SAMPLE = os.path.join(PATH_EXPORT, 'sample_submission.csv')
FILE_SUBMIT = os.path.join(PATH_EXPORT, 'predictions', 'submission.csv')

show_columns = ['DistrictId', 'Rooms', 'Square', 'LifeSquare', 'KitchenSquare',
                'other_sq', 'Floor', 'HouseFloor', 'HouseYear', 'Price',
                'learn', 'kitchen_good', 'kt_ratio', 'rm_ratio']


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
        elif value in ('max_depth', 'r2_train', 'r2_valid'):
            width = 14
        else:
            width = 32
        worksheet.set_column(num, num, width, cell_format)
    worksheet.autofilter(0, 0, len(data) - 1, len(data) - 1)
    writer.save()
    # End excel save


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
                df = pd.read_csv(name_file, sep=';', index_col='Id')
                df.Floor = df.Floor.astype(int)
                df.HouseFloor = df.HouseFloor.astype(int)
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
        df.to_csv(name_file, sep=';')
        df.to_pickle(file_pickle)


class DataPreprocessing(ReadWriteDataset):
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.number_iterations = 2
        self.dfg = pd.DataFrame
        self.msk = pd.DataFrame
        self.df_all = pd.DataFrame
        self.rooms_square_quantile = dict()
        self.medians = None
        self.dfg_HouseYear_min = None
        self.file_houses = 'moscow_houses.csv'
        self.years_floors = ([1957, 14], [1960, 12], [1961, 12], [1977, 22])
        self.new_columns = ['rooms_good', 'square_good', 'livesq_good',
                            'orig_both_sq', 'livesq_nonan', 'kitchen_good',
                            'other_sq_good', 'rm_ratio_good', 'house_fl_good',
                            'house_fl_nonan', 'house_hy_good']

    @staticmethod
    def concat_df(df_train, df_test):
        """Объединение датафреймов"""
        df_train['learn'] = 1
        df_test['learn'] = 0
        df = pd.concat([train, test])
        df.DistrictId = df.DistrictId.astype(int)
        df.Rooms = df.Rooms.astype(int)
        df.HouseFloor = df.HouseFloor.astype(int)
        df.KitchenSquare = df.KitchenSquare.astype(int)
        return df

    def fit(self, df, name_file=''):
        """Сохранение статистик"""
        # Расчет медиан
        self.medians = df.median()
        self.dfg = df.groupby('HouseYear',
                              as_index=False).HouseFloor.max().astype(int)
        self.dfg['DataKind'] = 'Датасет'

        # подготовка данных по домам
        if not name_file:
            name_file = self.file_houses
        # распределение этажности домов по годам с сайта МинЖКХ
        # https://dom.mingkh.ru/moskva/moskva/
        if os.access(name_file, os.F_OK):
            houses = pd.read_csv(name_file, sep=';', index_col='pos')
            houses['area'] = pd.to_numeric(houses['area'], errors='coerce')
            houses['HouseYear'] = pd.to_numeric(houses['HouseYear'],
                                                errors='coerce')
            houses['HouseFloor'] = pd.to_numeric(houses['HouseFloor'],
                                                 errors='coerce')
            houses.dropna(inplace=True)
            houses.HouseYear = houses.HouseYear.astype(int)
            houses.HouseFloor = houses.HouseFloor.astype(int)
            houses.HouseYear = houses.HouseYear - 1
            msk = houses[houses.HouseYear >= self.dfg.HouseYear.min()]
            msk = msk.groupby('HouseYear',
                              as_index=False).HouseFloor.max().astype(int)
        else:
            msk = self.dfg.copy(deep=True)
            for year, floor in self.years_floors:
                msk.loc[msk.HouseYear == year, 'HouseFloor'] = floor
        self.msk = msk.sort_values('HouseYear')
        self.msk['DataKind'] = 'Cайт ЖКХ'

    @property
    def group_year_floor(self):
        """
        Группировка данных для отображения максимальной этажности домов по году
        :return: объединенные: ДФ по строкам, ДФ по столбцам
        """
        df_concat = pd.concat([self.dfg, self.msk], axis=0)
        df_merged = self.dfg[['HouseYear', 'HouseFloor']].merge(
            self.msk[['HouseYear', 'HouseFloor']], on='HouseYear')
        df_merged.columns = ['HouseYear', 'HouseFloor', 'HouseFloor_MSK']
        return df_concat, df_merged

    def fill_square(self, row, name_field, good_field):
        """
        Заполнение Square и KitchenSquare
        :param row: строка датафрейма
        :param name_field: имя поля, которое заполняем
        :param good_field: имя поля для отбора записей, какие корректируем
        :return: площадь
        """
        # если это правильная запись - просто выход и возврат её значения
        if row[good_field]:
            return row[name_field]

        koefic = 0.1
        square = row['Square']
        min_sq = square * (1 - koefic)
        max_sq = square * (1 + koefic)

        conditions = [(self.df_all['DistrictId'] == row['DistrictId']),
                      (self.df_all['HouseYear'] == row['HouseYear']),
                      (self.df_all['Square'] > min_sq),
                      (self.df_all['Square'] < max_sq),
                      True]
        if name_field == 'Square':
            idx_end = 4
            for idx_c in range(idx_end - 1, len(conditions)):
                conditions[idx_c] = True
        else:
            idx_end = len(conditions)
        tmp = pd.DataFrame()
        for idx_c in range(idx_end):
            if idx_c != 2:
                tmp = self.df_all.loc[(self.df_all[good_field] > 0) &
                                      (self.df_all['Rooms'] == row['Rooms']) &
                                      conditions[0] &
                                      conditions[1] &
                                      conditions[2] &
                                      conditions[3]]
                if len(tmp):
                    break
            conditions[idx_c] = True
        if len(tmp):
            if name_field == 'KitchenSquare':
                return int(round(tmp[name_field].median() + 0.1, 0))
            else:
                sq_mean = tmp[name_field].mean()
                if name_field == 'LifeSquare' and sq_mean > square:
                    sq_mean = square
                return sq_mean
        return row[name_field]

    @staticmethod
    def recalc_square(df_in):
        """
        расчет дополнительных фич
        :return: ДФ
        """
        df = df_in.copy(deep=True)
        df['kt_ratio'] = df.Square / df.KitchenSquare
        df['rm_ratio'] = df.LifeSquare / df.Rooms
        df['other_sq'] = df.Square - df.KitchenSquare - df.LifeSquare
        df['int_square'] = df.Square.round(0).astype(int)
        return df

    def transform_rooms(self, df_in):
        """
        Заполнение кол-ва комнат
        :param df_in: входной ДФ
        :return: измененный ДФ
        """
        df = df_in.copy(deep=True)
        # Rooms
        # заполним отсутствующее число комнат по минимальной разнице площади
        # квартиры и медианного значения площади в этом районе
        df.loc[(df.Rooms == 0) | (df.Rooms > 6), 'rooms_good'] = 0
        tmp = df[df.rooms_good == 0]
        list_index = tmp.index
        districts = tmp.DistrictId.unique()
        tmp = df[df.DistrictId.isin(districts) & ~df.index.isin(list_index)]
        grooms = tmp.groupby(['DistrictId', 'Rooms'],
                             as_index=False).agg({'Square': 'median'})
        grooms.columns = ['DistrictId', 'Rooms', 'sq_median']
        for idx in list_index:
            grooms['delta'] = abs(grooms.sq_median - df.loc[idx, 'Square'])
            id_dst = df.loc[idx, 'DistrictId']
            idx_min = grooms[grooms.DistrictId == id_dst].delta.idxmin()
            df.loc[idx, 'Rooms'] = int(grooms.loc[idx_min]['Rooms'])
        # при сопоставлении кол-ва комнат, площади и цены были выявлены выбросы
        tmp = df[df.Rooms == 5]
        # площадь соответствует количеству комнат = 1 с медианной площадью 40.3
        for idx in tmp[tmp.Price < tmp.Price.median() / 2].index:
            df.loc[idx, 'Rooms'] = 1
            df.loc[idx, 'rooms_good'] = 0
        return self.recalc_square(df)

    def transform_rooms_add(self, df_in):
        """
        Изменние кол-ва комнат - тонкая настройка
        :param df_in: входной ДФ
        :return: измененный ДФ
        """
        df = df_in.copy(deep=True)
        # исправление данных по шестикомнатной квартире на двухкомнатную
        room_idx = df.query('Rooms == 6 and Square < 60').index
        df.loc[room_idx, 'Rooms'] = 2
        # исправление данных по шестикомнатной квартире на пятикомнатную
        idxes = df.query('Rooms == 6 and Square < 120').index
        room_idx.append(idxes)
        df.loc[idxes, 'Rooms'] = 5
        df.loc[room_idx, 'KitchenSquare'] = 10
        df.loc[room_idx, 'kitchen_good'] = 0
        # Оставшиеся шестикомнатные квартиры преобразуем в пятикомнатные
        idxes = df.query('Rooms == 6').index
        df.loc[idxes, 'Rooms'] = 5
        room_idx.append(idxes)
        # Двух-трехкомнатные квартиры в общей площадью < 32
        # преобразуем в однокомнатные
        idxes = df.query('Rooms in (2, 3) and Square < 32').index
        df.loc[idxes, 'Rooms'] = 1
        room_idx.append(idxes)
        # Трехкомнатные квартиры в общей площадью < 40
        # преобразуем в двухкомнатные
        idxes = df.query('Rooms == 3 and Square < 40').index
        df.loc[idxes, 'Rooms'] = 2
        room_idx.append(idxes)
        # 4х комнатные, с общей площадью < 50 преобразуем в двухкомнатные
        idxes = df.query('Rooms == 4 and Square < 50').index
        df.loc[idxes, 'Rooms'] = 2
        room_idx.append(idxes)
        # 5ти комнатные, с общей площадью < 60 преобразуем в трехкомнатные
        idxes = df.query('Rooms == 5 and Square < 60').index
        df.loc[idxes, 'Rooms'] = 3
        room_idx.append(idxes)
        df.loc[room_idx, 'rooms_good'] = 0
        return self.recalc_square(df)

    def transform_life_square_square(self, df_in):
        """
        Заполнение общей и жилой площади
        :param df_in: входной ДФ
        :return: измененный ДФ
        """
        df = df_in.copy(deep=True)
        # LifeSquare, Square
        # присвоим значения LifeSquare = Square, если LifeSquare > Square * 2
        # или если LifeSquare > Square * 1.2 и LifeSquare > 120
        term1 = (df.LifeSquare > df.Square * 2)
        term2 = ((df.LifeSquare > df.Square * 2) & (
                df.LifeSquare > 120))
        idxes = df[term1 | term2].index
        df.loc[idxes, 'LifeSquare'] = df.loc[idxes, 'Square'] - 0.1
        df.loc[idxes, 'livesq_good'] = 0
        # очистим LifeSquare если LifeSquare > 280
        df.loc[df.LifeSquare > 280, 'LifeSquare'] = 0
        df.loc[df.LifeSquare > 280, 'livesq_good'] = 0
        # поменяем значения LifeSquare <-> Square, если LifeSquare > Square
        idxes = df[df.LifeSquare > df.Square].index
        df.loc[idxes, ['Square', 'LifeSquare']] = df.loc[
            idxes, ['LifeSquare', 'Square']].values
        df.loc[idxes, 'orig_both_sq'] = 0
        # удаление выбросов Square
        df.loc[(df.Square < 13) | (df.Square > 280), 'square_good'] = 0
        self.df_all = df
        df['new_sq'] = df.apply(
            lambda rowx: self.fill_square(rowx, 'Square', 'square_good'),
            axis=1)
        df['Square'] = df['new_sq']
        df.drop(['new_sq'], axis=1, inplace=True)
        return self.recalc_square(df)

    def fill_nan_life_square(self, df_in):
        """
        Заполнение отсутствущей жилой площади
        :param df_in: входной ДФ
        :return: измененный ДФ
        """
        df = df_in.copy(deep=True)
        # отсутствующую жилую площадь будем заполнять средней площадью квартир
        # из этого района, количества комнат и дома одного года постройки.
        # Если не нашли дома по году, проделаем тоже самое без года
        # удаление пропусков LifeSquare
        df.loc[pd.isna(df.LifeSquare), 'livesq_nonan'] = 0
        self.df_all = df
        df['new_ls1'] = df.apply(lambda rowx: self.fill_square(rowx,
                                                               'LifeSquare',
                                                               'livesq_nonan'),
                                 axis=1)
        # второй этап
        tmp = df[df.livesq_nonan > 0]
        grooms = tmp.groupby(['DistrictId', 'Rooms'],
                             as_index=False).aggregate({'Square': 'mean',
                                                        'LifeSquare': 'mean'})
        grooms.columns = ['DistrictId', 'Rooms', 'sq_mean', 'lq_mean']
        # заполним отсутствующие жилую площадь по минимальной разнице площади
        # квартиры и медианного значения площади в этом районе
        df['new_ls2'] = df.LifeSquare
        for idx in df[df.livesq_nonan < 1].index:
            id_dst = df.loc[idx, 'DistrictId']
            grooms['delta'] = abs(grooms.sq_mean - df.loc[idx, 'Square'])
            idx_tmp = grooms[grooms.DistrictId == id_dst]
            if not len(idx_tmp):
                idx_tmp = grooms
            idx_min = idx_tmp.delta.idxmin()
            df.loc[idx, 'new_ls2'] = grooms.loc[idx_min]['lq_mean']
            min_ls = df.loc[idx, ['new_ls1', 'new_ls2']].min()
            max_ls = df.loc[idx, ['new_ls1', 'new_ls2']].max()
            if max_ls <= df.loc[idx, 'Square'] * 0.8:
                df.loc[idx, 'LifeSquare'] = max_ls
            else:
                df.loc[idx, 'LifeSquare'] = min_ls
        df.drop(['new_ls1', 'new_ls2'], axis=1, inplace=True)
        return self.recalc_square(df)

    def fill_kitchen_square(self, df_in):
        """
        Заполнение площади кухни
        :param df_in: входной ДФ
        :return: измененный ДФ
        """
        df = df_in.copy(deep=True)
        #  выбросами будем считать площадь кухни в квартирах с комнатами:
        #  1к < 3м, 2к < 4м, остальные < 5м.
        #  Так же выбросами будем считать если площадь кухни меньше
        #  оставшейся площади = Square - LifeSquare
        df['kt_ratio'] = df.Square / df.KitchenSquare
        rooms_cond = [(df.Rooms == 1), (df.Rooms == 2), (df.Rooms >= 3)]
        ksq, k_rat = (3, 4, 5), (2.3, 2.4, 3.2)
        # отметка выбросов по площади кухни
        for idx, cnd in enumerate(rooms_cond):
            df.loc[cnd & (df.KitchenSquare < ksq[idx]), 'kitchen_good'] = 0
            df.loc[cnd & (df.kt_ratio < k_rat[idx]), 'kitchen_good'] = 0
        # площадь кухни будем заполнять по медианной площади кухни у квартир
        # из этого района, количества комнат и дома одного года постройки.
        # Если не нашли дома по году, проделаем тоже самое без года
        # удаление выбросов KitchenSquare
        self.df_all = df
        df['new_ks'] = df.apply(lambda rowx: self.fill_square(rowx,
                                                              'KitchenSquare',
                                                              'kitchen_good'),
                                axis=1)
        df['KitchenSquare'] = df['new_ks']
        df.drop(['new_ks'], axis=1, inplace=True)
        df = self.recalc_square(df)
        # исправление косяков с площадями: пересчитаем площадь кухни
        cond_other = (df.Square < 26) & (df.other_sq < 0)
        df.loc[cond_other, 'KitchenSquare'] = df.loc[cond_other, 'Square'] - \
                                              df.loc[cond_other, 'LifeSquare']
        # если площадь кухни стала = 0 - поставим медианное значение
        ks_null = (df.Square < 26) & (df.KitchenSquare < 1)
        med_ks = df[(df.Square < 26) &
                    (df.KitchenSquare > 0)].KitchenSquare.median()
        df.loc[ks_null, 'KitchenSquare'] = med_ks
        df.loc[ks_null, 'LifeSquare'] = df.loc[ks_null, 'LifeSquare'] - med_ks
        df.KitchenSquare = df.KitchenSquare.astype(int)
        df = self.recalc_square(df)

        # для проставления адекватной площади кухни сгруппируем данные
        # по кол-ву комнат и целой площади
        cond_grp = (df.other_sq > 0) | (df.kitchen_good > 0)
        inv_cond = ((df.other_sq <= 0) & (df.kitchen_good < 1))
        group_rooms = self.make_grp_df(df, cond_grp, inv_cond)
        # установка KitchenSquare
        # для (Square > 26) & (other_sq <= 0) & (kitchen_good < 1)
        cond_ks = (df.Square > 26) & inv_cond
        for idx in df[cond_ks].index:
            intsq = df.loc[idx, 'rooms_intsq']
            df.loc[idx, 'KitchenSquare'] = group_rooms.loc[intsq, 'ks_median']
        return self.recalc_square(df)

    def transform_life_square(self, df_in):
        """
        Заполняем LifeSquare, где средняя площадь меньше порога в 9.8 кв.м
        :return: ДФ
        """
        df = df_in.copy(deep=True)
        # отметка, что средняя площадь комнаты менее допустимого порога
        df.loc[(df.Square > 26) & (df.rm_ratio < 9.8), 'rm_ratio_good'] = 0
        # Заполняем LifeSquare
        cond_grp = ((df.other_sq > 0) & (df.rm_ratio > 9.8))
        inv_cond = ((df.other_sq <= 0) | (df.rm_ratio < 9.8))
        group_rooms = self.make_grp_df(df, cond_grp, inv_cond)
        cond_ks = (df.Square > 26) & inv_cond
        for idx in df[cond_ks].index:
            intsq = df.loc[idx, 'rooms_intsq']
            df.loc[idx, 'LifeSquare'] = group_rooms.loc[intsq, 'lq_median']
        df = self.recalc_square(df)
        # посчитаем LifeSquare = Square - KitchenSquare - other_sq
        for k, cond_sq in enumerate(
                [((df.Square > 58) & (df.Square < 75)),
                 ((df.Square > 110) & (df.Square < 130))]):
            for idx in df[(df.other_sq <= 0) & cond_sq].index:
                sq = df.loc[idx, 'Square']
                ks = df.loc[idx, 'KitchenSquare']
                mos = df[cond_sq]['other_sq'].mean()
                rms = df.loc[idx, 'Rooms']
                df.loc[idx, 'LifeSquare'] = sq - ks - mos / (rms - k)
        return self.recalc_square(df)

    def transform_floors(self, df_in):
        """
        Заполняем выбросы по этажам, отсутствующие и перепутанные этажи
        :return: ДФ
        """
        df = df_in.copy(deep=True)

        # В этой квартире ошиблись номером этажа, вместо 18, написали 78
        df.loc[df.Floor == 78, 'Floor'] = 18
        # исправленяем этажи в домах более 48 этажей
        grp_field = 'HouseFloor'
        inv_cond = (df[grp_field] > 48) & (df.HouseYear == 1977)
        idx_cond = df[inv_cond].index
        id_rooms = df.loc[idx_cond, 'Rooms'].unique()
        cond_grp = (df[grp_field] > 0) & (df[grp_field] <= 48) & \
                   (df.HouseYear == 1977) & df.Rooms.isin(id_rooms) & \
                   ~df.index.isin(idx_cond)
        group_rooms = self.make_grp_df(df, cond_grp, inv_cond)
        for idx in idx_cond:
            intsq = df.loc[idx, 'rooms_intsq']
            df.loc[idx, grp_field] = group_rooms.loc[intsq, 'hf_median']
            df.loc[idx, 'house_fl_good'] = 0

        # пометим нулевую этажность домов
        df.loc[df.HouseFloor < 1, 'house_fl_nonan'] = 0
        # Для записей HouseFloor == 0
        # заполним этажность медианным значением у аналогичных квартир
        grp_field = 'HouseFloor'
        inv_cond = (df[grp_field] < 1)
        idx_cond = df[inv_cond].index
        cond_grp = (df[grp_field] > 0) & ~df.index.isin(idx_cond)
        group_rooms = self.make_grp_df(df, cond_grp, inv_cond)
        for idx in idx_cond:
            intsq = df.loc[idx, 'rooms_intsq']
            df.loc[idx, grp_field] = group_rooms.loc[intsq, 'hf_median']
            # если вдруг найденная этажность дома меньше нужной запишем часто
            # используемое значение в этом районе
            if df.loc[idx, grp_field] < df.loc[idx, 'Floor']:
                district = df.loc[idx, 'DistrictId']
                square = df.loc[idx, 'Square'] * 0.6
                hfloor = df.loc[idx, grp_field]
                hf_median = df.loc[(df.Square > square) &
                                   (df.HouseFloor >= hfloor) &
                                   (df.DistrictId == district),
                                   'HouseFloor'].value_counts().index[0]
                df.loc[idx, grp_field] = hf_median
        # поменяем значения HouseFloor <-> Floor, если HouseFloor < Floor
        idxes = df[df.HouseFloor < df.Floor].index
        df.loc[idxes, ['Floor', 'HouseFloor']] = df.loc[
            idxes, ['HouseFloor', 'Floor']].values
        # пометим кривую этажность домов
        df.loc[idxes, 'house_fl_good'] = 0

        # Группировка данных c максимальной этажностью домов по годам
        tmp_concat, tmp_merged = self.group_year_floor
        # отбор домов за период (1920, 2000) где разница больше трех этажей
        tmp_merged = tmp_merged.query('HouseFloor - HouseFloor_MSK > 3 and '
                                      'HouseYear > 1920 and HouseYear < 2000')
        years_floors = tmp_merged[['HouseYear',
                                   'HouseFloor_MSK']].to_dict('split')['data']
        # исправляем год дома, если этажность больше построенных домов
        grp_field = 'HouseYear'
        for year, floors in years_floors:
            inv_cond = (df[grp_field] == year) & (df.HouseFloor - floors > 1)
            set_floors = sorted(df[inv_cond].HouseFloor.unique())
            for floor in set_floors:
                flr_cond = (df.HouseFloor == floor)
                inv_cond = (df[grp_field] == year) & flr_cond
                idx_cond = df[inv_cond].index
                cond_grp = (df[grp_field] != year) & flr_cond & \
                           ~df.index.isin(idx_cond)
                cond = (df[grp_field] != year) & ~df.index.isin(
                    idx_cond)
                group_rooms = self.make_grp_df(df, cond_grp, inv_cond, cond,
                                               floor)
                for idx in idx_cond:
                    intsq = df.loc[idx, 'rooms_intsq']
                    df.loc[idx, grp_field] = group_rooms.loc[intsq, 'hy_mode']
                    df.loc[idx, 'house_hy_good'] = 0
        return self.recalc_square(df)

    def transform(self, df_in):
        """
        Трансформация данных
        :return: ДФ
        """
        # если есть файл с заполненными HouseYear после первой итерации
        # - достаточно одной итерации
        if os.access(FILE_YEARS, os.F_OK):
            self.number_iterations = 1

        for n_iter in range(self.number_iterations):
            if self.number_iterations == 2:
                print(f"{['Первый', 'Второй'][n_iter]} этап обработки данных")

            df = df_in.copy(deep=True)

            # исправим 2 ошибочных года
            df.loc[df.HouseYear == 4968, 'HouseYear'] = 1968
            df.loc[df.HouseYear == 20052011, 'HouseYear'] = 2011
            current_year = datetime.now().year
            df.loc[df.HouseYear > current_year, 'HouseYear'] = current_year

            # есть файл с заполненными HouseYear после первой итерации - читаем
            # т.к. на основании года будут заполняться отсутствующие площади
            if os.access(FILE_YEARS, os.F_OK):
                changed_years = pd.read_csv(FILE_YEARS, sep=';',
                                            index_col='Id')
                idx_years = changed_years.index
                columns_years = ['Floor', 'HouseFloor', 'HouseYear',
                                 'house_fl_good', 'house_fl_nonan',
                                 'house_hy_good']
                df.loc[idx_years, columns_years] = changed_years

            self.df_all = df

            # удалим Healthcare_1
            if 'Healthcare_1' in df.columns:
                df.drop('Healthcare_1', axis=1, inplace=True)

            # пропишем, что изначально эти признаки в датасете хорошие
            for column in self.new_columns:
                df[column] = 1
            df = self.recalc_square(df)
            df.loc[df.other_sq <= 0, 'other_sq_good'] = 0

            # Rooms
            # заполним отсутствующее число комнат по мин.разнице площади
            # квартиры и медианного значения площади в этом районе
            df = self.transform_rooms(df)

            # LifeSquare, Square
            # значения LifeSquare = Square, если LifeSquare > Square * 2
            # или если LifeSquare > Square * 1.2 и LifeSquare > 120
            # поменяем значения LifeSquare <-> Square, если LifeSquare > Square
            # удаление выбросов Square
            df = self.transform_life_square_square(df)

            # Rooms
            # доп. преобразование после этапа transform_life_square_square
            df = self.transform_rooms_add(df)

            # добавление составного поля из двух колонок для индекса
            df['rooms_intsq'] = df.Rooms.astype(str) + '_' + \
                                df.int_square.astype(str)
            df['rooms_intsq'] = df['rooms_intsq'].astype('category')

            # Nan LifeSquare
            # Nan жилую площадь будем заполнять средней площадью квартир
            # из этого района, количества комнат и дома одного года постройки.
            # Если не нашли дома по году, проделаем тоже самое без года
            # удаление пропусков LifeSquare
            df = self.fill_nan_life_square(df)

            # KitchenSquare
            # заполнение отсутствующей площади кухни
            # и исправление ошибок в данных
            df = self.fill_kitchen_square(df)

            # LifeSquare
            # Заполняем LifeSquare, где средняя площадь меньше порога 9.8 кв.м
            df = self.transform_life_square(df)

            # Floor, HouseFloor
            # Заполняем выбросы по этажам, отсутствующие и перепутанные этажи
            df = self.transform_floors(df)

            # если нет файла с заполненными HouseYear после первой итерации -
            # создаем его
            if not os.access(FILE_YEARS, os.F_OK):
                # сохранение измененных годов,
                # чтобы прочитать на следующей итерации
                columns_years = ['Floor', 'HouseFloor', 'HouseYear',
                                 'house_fl_good', 'house_fl_nonan',
                                 'house_hy_good']
                changed_years = df[(df['house_hy_good'] < 1)][columns_years]
                changed_years.to_csv(FILE_YEARS, sep=';')
        # end iterations

        df.fillna(self.medians, inplace=True)
        df.loc[df.learn == 0, 'Price'] = np.NaN

        # добавим колонку с ценой 1кв.метра
        df['price_sqm'] = df.Price / df.Square

        # подготовка словаря: кол-во комнат, площадь по квантилям
        for room in range(1, df.Rooms.max() + 1):
            quarters = []
            # # делим на 4 части диапазон площадей
            # for q in np.linspace(0, 1, 5):
            # на 3 части по принципу 80/20%
            for q in (0, .2, .8, 1.):
                quarters.append(df[df.Rooms == room]['int_square'].quantile(q))
            quarters[0] -= 1
            quarters.insert(0, 0)
            self.rooms_square_quantile[room] = [int(q) for q in quarters]

        # заполнение в какой четверти находится площадь квартиры
        df['quart_sq'] = df.apply(lambda row: self.quarters(row), axis=1)
        df['roomsq'] = df['Rooms'] * 10 + df['quart_sq']

        # поделим квартиры на категории в зависимости от года постройки дома
        # по эпохам строительства
        # https://strelkamag.com/ru/article/moscow-housing-map
        years_bins = [0, 1917, 1924, 1953, 1964, 1982, 1991, 2010,
                      df['HouseYear'].max()]
        # заполнение категории года дома
        df['year_cat'] = pd.cut(df.HouseYear, bins=years_bins, labels=False)

        return df

    def quarters(self, rowx):
        bins = self.rooms_square_quantile[rowx.Rooms]
        return pd.cut([rowx.int_square], bins=bins, labels=False)[0]

    @staticmethod
    def make_grp_df(df, cond_grp, inv_cond, add_cond=True, chk_floor=None,
                    test_index=None):
        """
        для проставления адекватной площади кухни сгруппируем данные
        по кол-ву комнат и целой площади
        :param df:  входной ДФ
        :param cond_grp: условие для группировки
        :param inv_cond: условие для "плохих записей", которые нужно править
        :param add_cond: добавочное условие для группировки
        :param chk_floor: условие для проверки этажности
        :param test_index: индекс для тестирования метода
        :return: ДФ с группированными данными
        """

        def calc_min_max(square, koefic=None, check_floor=None):
            if not koefic:
                if square <= 50:
                    koefic = 0.1
                elif square < 150:
                    koefic = 0.05
                else:
                    koefic = 0.2
            min_sq = square * (1 - koefic)
            max_sq = square * (1 + koefic)

            if check_floor:
                add_floor = (df.HouseFloor == check_floor)
            else:
                add_floor = True

            return df[(df['Rooms'] == rooms) &
                      (df['Square'] > min_sq) &
                      (df['Square'] < max_sq) & add_cond & add_floor]

        grpr = df[cond_grp].groupby(['Rooms', 'int_square'],
                                    as_index=False).agg(
            {'LifeSquare': 'median',
             'KitchenSquare': 'median',
             'HouseFloor': 'median',
             'HouseYear': lambda x: x.value_counts().index[0]})
        grpr.columns = ['Rooms', 'int_square', 'lq_median', 'ks_median',
                        'hf_median', 'hy_mode']
        grpr['rooms_intsq'] = grpr.Rooms.astype(
            str) + '_' + grpr.int_square.astype(str)
        grpr['rooms_intsq'] = grpr['rooms_intsq'].astype('category')

        # проверим все ли квартиры найдутся в группировке
        if test_index:
            no_found = df[df.index == test_index]
        else:
            no_found = df[inv_cond & ~df.rooms_intsq.isin(
                grpr.rooms_intsq)].sort_values(['Rooms', 'Square'])
        # Добавим в группировку строки из этой таблицы для отсутствующих
        # значений int_square с группировкой по диапазону Square +/- 10%.
        for idx in no_found.index:
            rooms = no_found.loc[idx, 'Rooms']
            rsquare = no_found.loc[idx, 'Square']
            int_square = no_found.loc[idx, 'int_square']
            for koef in (None, 0.2, 0.25, 0.3):
                tmp = calc_min_max(rsquare, koef, chk_floor)
                if len(tmp):
                    break
            if not len(tmp):
                for koef in (None, 0.2, 0.25, 0.3):
                    tmp = calc_min_max(rsquare, koef, chk_floor - 1)
                    if len(tmp):
                        break
            # Rooms	int_sq sq_median ks_median hf_median hhy_mode rooms_intsq
            idx_rooms_intsq = f'{rooms}_{int_square}'
            if idx_rooms_intsq not in grpr.rooms_intsq.values:
                # print(f'{idx_rooms_intsq} '
                #       f'{tmp.HouseYear.value_counts().index}')
                grpr.loc[len(grpr)] = [rooms, int_square,
                                       tmp.LifeSquare.median(),
                                       tmp.KitchenSquare.median(),
                                       tmp.HouseFloor.median(),
                                       tmp.HouseYear.value_counts().index[0],
                                       idx_rooms_intsq]
        grpr.ks_median = (grpr.ks_median + 0.1).round(0).astype(int)
        grpr.hf_median = (grpr.hf_median + 0.1).round(0).astype(int)
        grpr.hy_mode = (grpr.hy_mode + 0.1).round(0).astype(int)
        # установка нового индекса
        grpr = grpr.set_index('rooms_intsq')
        return grpr


class FeatureGenerator(ReadWriteDataset):
    """Генерация новых фич"""

    def __init__(self):
        """ Инициализация класса """
        self.binary_to_numbers = None
        self.district_size = None
        self.cat_features = ['DistrictId', 'Ecology_2', 'Ecology_3', 'Shops_2',
                             'rooms_intsq', 'roomsq']
        self.columns_to_group = ['year_cat', 'DistrictId',
                                 'Social_1', 'Social_2', 'Social_3', 'Shops_1']
        self.tsne_columns = []
        self.med_rooms_price_quart = None
        self.med_year_floor_Rooms_price = None
        self.med_year_floor_roomsq_price = None
        self.med_district_rooms_price = None
        self.med_district_year_Rooms_price = None
        self.med_district_year_roomsq_price = None
        self.med_social_rooms_price = None
        self.med_social_year_Rooms_price = None
        self.med_social_year_roomsq_price = None
        self.exclude_columns = ['Id', 'learn', 'price_sqm']
        self.learn_columns = []
        self.dummy = []
        self.dif = 1.6

    def fit(self, df_in):
        """
        Подготовка для добавления новых признаков
        :param df_in: входной ДФ
        :return: None
        """
        df = df_in.copy(deep=True)

        # колонки для кластеризации
        self.tsne_columns = [col for col in df.columns if
                             col not in self.exclude_columns + ['Price']]

        # Binary features
        self.binary_to_numbers = {'A': 0, 'B': 1}

        # DistrictID
        self.district_size = df[
            'DistrictId'].value_counts().reset_index().rename(
            columns={'index': 'DistrictId', 'DistrictId': 'district_size'})

        # группировка по кол-ву комнат с медианной ценой квартиры и кв. метра
        self.med_rooms_price_quart = df[df.learn == 1].groupby(['roomsq']).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_rooms_square',
                     'Price': 'med_rooms_price',
                     'price_sqm': 'med_rooms_psqm'
                     })

        for name_col in self.columns_to_group:
            name_sqr = f'med_{name_col}_square'
            name_atr = f'med_{name_col}_price'
            name_sqm = f'med_{name_col}_psqm'
            grp = df[df.learn == 1].groupby(
                [name_col, 'Rooms'], as_index=False).agg(
                {'Square': 'median', 'Price': 'median',
                 'price_sqm': 'median'}).rename(
                columns={'Square': name_sqr, 'Price': name_atr,
                         'price_sqm': name_sqm})
            grp['feat_rooms'] = grp[name_col] * 10 + grp.Rooms
            grp = grp.set_index('feat_rooms')
            setattr(FeatureGenerator, name_atr, grp)
            # более детальная группировка
            name_sqr = f'med_{name_col}_rooms_square'
            name_atr = f'med_{name_col}_rooms_price'
            name_sqm = f'med_{name_col}_rooms_psqm'
            grp = df[df.learn == 1].groupby(
                [name_col, 'Rooms', 'quart_sq'], as_index=False).agg(
                {'Square': 'median', 'Price': 'median',
                 'price_sqm': 'median'}).rename(
                columns={'Square': name_sqr, 'Price': name_atr,
                         'price_sqm': name_sqm})
            setattr(FeatureGenerator, name_atr, grp)

        # группировка по району и кол-ву комнат с медианной ценой квартиры
        # и медионной ценой квадратного метра
        self.med_district_rooms_price = df[df.learn == 1].groupby(
            ['DistrictId', 'Rooms'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_district_rooms_square',
                     'Price': 'med_district_rooms_price',
                     'price_sqm': 'med_district_rooms_psqm'})
        self.med_district_rooms_price[
            'dist_rooms'] = self.med_district_rooms_price.DistrictId * 10 + \
                            self.med_district_rooms_price.Rooms
        self.med_district_rooms_price = \
            self.med_district_rooms_price.set_index('dist_rooms')

        # группировка по району, категории года и комнатам с медианной ценой
        # и медионной ценой квадратного метра
        self.med_district_year_Rooms_price = df[df.learn == 1].groupby(
            ['DistrictId', 'year_cat', 'Rooms'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_district_year_rooms_square',
                     'Price': 'med_district_year_rooms_price',
                     'price_sqm': 'med_district_year_rooms_psqm'})

        # группировка по району, категории года и комнатам поделенным на
        # категории с медианной ценой и медионной ценой квадратного метра
        # и медионной ценой квадратного метра
        self.med_district_year_roomsq_price = df[df.learn == 1].groupby(
            ['DistrictId', 'year_cat', 'roomsq'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_district_year_rooms_square',
                     'Price': 'med_district_year_rooms_price',
                     'price_sqm': 'med_district_year_rooms_psqm'})

        # группировка по категории года, этажу и комнатам с медианной ценой
        # и медионной ценой квадратного метра
        self.med_year_floor_Rooms_price = df[df.learn == 1].groupby(
            ['year_cat', 'Floor', 'Rooms'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_year_floor_rooms_square',
                     'Price': 'med_year_floor_rooms_price',
                     'price_sqm': 'med_year_floor_rooms_psqm'})

        # группировка по категории года, этажу и комнатам поделенным на
        # категории, с медианной ценой и медионной ценой квадратного метра
        self.med_year_floor_roomsq_price = df[df.learn == 1].groupby(
            ['year_cat', 'Floor', 'roomsq'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_year_floor_rooms_square',
                     'Price': 'med_year_floor_rooms_price',
                     'price_sqm': 'med_year_floor_rooms_psqm'})

        # группировка по Social_1 и кол-ву комнат с медианной ценой квартиры
        # и медионной ценой квадратного метра
        self.med_social_rooms_price = df[df.learn == 1].groupby(
            ['Social_1', 'Rooms'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_social_rooms_square',
                     'Price': 'med_social_rooms_price',
                     'price_sqm': 'med_social_rooms_psqm'})
        self.med_social_rooms_price[
            'soc_rooms'] = self.med_social_rooms_price.Social_1 * 10 + \
                           self.med_social_rooms_price.Rooms
        self.med_social_rooms_price = \
            self.med_social_rooms_price.set_index('soc_rooms')

        # группировка по Social_1, категории года и комнатам с медианной ценой
        # и медионной ценой квадратного метра
        self.med_social_year_Rooms_price = df[df.learn == 1].groupby(
            ['Social_1', 'year_cat', 'Rooms'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_social_year_rooms_square',
                     'Price': 'med_social_year_rooms_price',
                     'price_sqm': 'med_social_year_rooms_psqm'})

        # группировка по по району, категории года и комнатам поделенным на
        # категории с медианной ценой и медионной ценой квадратного метра
        # и медионной ценой квадратного метра
        self.med_social_year_roomsq_price = df[df.learn == 1].groupby(
            ['Social_1', 'year_cat', 'roomsq'], as_index=False).agg(
            {'Square': 'median', 'Price': 'median',
             'price_sqm': 'median'}).rename(
            columns={'Square': 'med_social_year_rooms_square',
                     'Price': 'med_social_year_rooms_price',
                     'price_sqm': 'med_social_year_rooms_psqm'})

    def find_min(self, rowx):
        """
        Исправление разницы между med_year_floor_rooms_price и
        med_district_year_rooms_price больше, чем в полтора раза
        :param rowx: строка ДФ
        :return: более подходящее значение med_year_floor_rooms_price
        """
        fields = ['med_year_cat_rooms_price',
                  'med_DistrictId_rooms_price',
                  'med_Social_1_rooms_price',
                  'med_Social_2_rooms_price',
                  'med_Social_3_rooms_price',
                  'med_Shops_1_rooms_price']
        yfr = rowx.med_year_floor_rooms_price
        dyr = rowx.med_district_year_rooms_price
        dvd = yfr / dyr

        if 1 / self.dif <= dvd <= self.dif:
            return yfr

        values = [rowx[col] for col in fields if abs(rowx[col] - dyr) > 1 and
                  (1 / self.dif <= rowx[col] / dyr <= self.dif)]
        if dvd < 1:
            values_less = [rowx[col] for col in fields if
                           abs(rowx[col] - dyr) > 1 and rowx[col] < dyr and
                           (rowx[col] / dyr > 1 / self.dif or rowx[col] > yfr)]
            if values_less:
                return max(values_less)
            return yfr if not values else max(values)
        else:
            values_over = [rowx[col] for col in fields if
                           abs(rowx[col] - dyr) > 1 and rowx[col] > dyr and
                           (rowx[col] / dyr <= self.dif or rowx[col] < yfr)]
            if values_over:
                return min(values_over)
        return yfr if not values else max(values)

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
        col_clusters = self.tsne_columns[:17]
        # # колонки после препроцессинга
        # col_clusters = self.tsne_columns

        scaler = RobustScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(df[col_clusters]),
                                    columns=col_clusters, index=df.index)

        aggl = AgglomerativeClustering(n_clusters=n_clusters)
        labels = aggl.fit_predict(train_scaled)
        labels = pd.DataFrame(data=labels, columns=['cluster'], index=df.index)

        self.dummy.extend(['cluster'])
        print_time(cls_time)
        return labels

    def transform(self, df_in, clusters=0, dummy_cols=[], exclude_cols=[]):
        """
        Преобразование данных
        :param df_in: входной ДФ
        :param clusters: делить данные на количество кластеров
        :param dummy_cols: категорийные колонки для преобразования в признаки
        :param exclude_cols: колонки не участвующие в обработке
        :return: ДС с новыми признаками
        """
        self.dummy = dummy_cols
        df = df_in.copy(deep=True)
        df['Id'] = df.index

        # кодировка категорийных признаков
        df['Ecology_2'] = df['Ecology_2'].map(self.binary_to_numbers)
        df['Ecology_3'] = df['Ecology_3'].map(self.binary_to_numbers)
        df['Shops_2'] = df['Shops_2'].map(self.binary_to_numbers)

        # заполнение размера района
        df = df.merge(self.district_size, on='DistrictId', how='left')
        df['district_size'].fillna(1, inplace=True)

        # заполнение медианной цены квартиры по атрибутам
        for name_col in self.columns_to_group:
            name_sqr = f'med_{name_col}_rooms_square'
            name_atr = f'med_{name_col}_rooms_price'
            name_sqm = f'med_{name_col}_rooms_psqm'
            df = df.merge(getattr(FeatureGenerator, name_atr),
                          on=[name_col, 'Rooms', 'quart_sq'], how='left')
            # отсутствующиие значения заполним из верхнеуровневой группировки
            for idx in df[pd.isna(df[name_atr])].index:
                grp_sqr = f'med_{name_col}_square'
                grp_atr = f'med_{name_col}_price'
                grp_sqm = f'med_{name_col}_psqm'
                grp = getattr(FeatureGenerator, grp_atr)
                dsr = df.loc[idx, name_col] * 10 + df.loc[idx, 'Rooms']
                if dsr in grp.index:
                    sqr = grp.loc[dsr, grp_sqr]
                    prs = grp.loc[dsr, grp_atr]
                    sqm = grp.loc[dsr, grp_sqm]
                else:
                    # отсутствующиие значения заполним из группировки:
                    # район/кол.комнат, если такие значения не найдены -
                    # заполним из группировки: кол-во комнат
                    dsr = df.loc[idx, 'DistrictId'] * 10 + df.loc[idx, 'Rooms']
                    if dsr in self.med_district_rooms_price.index:
                        sqr = self.med_district_rooms_price.loc[
                            dsr, 'med_district_rooms_square']
                        prs = self.med_district_rooms_price.loc[
                            dsr, 'med_district_rooms_price']
                        sqm = self.med_district_rooms_price.loc[
                            dsr, 'med_district_rooms_psqm']
                    else:
                        roomsq = df.loc[idx, 'roomsq']
                        sqr = self.med_rooms_price_quart.loc[
                            roomsq, 'med_rooms_square']
                        prs = self.med_rooms_price_quart.loc[
                            roomsq, 'med_rooms_price']
                        sqm = self.med_rooms_price_quart.loc[
                            roomsq, 'med_rooms_psqm']
                df.loc[idx, name_sqr] = sqr
                df.loc[idx, name_atr] = prs
                df.loc[idx, name_sqm] = sqm

        # заполнение медианной цены по категории года, этажу, кол-ву комнат
        if self.med_year_floor_roomsq_price is not None:
            df = df.merge(self.med_year_floor_roomsq_price,
                          on=['year_cat', 'Floor', 'roomsq'], how='left')
            # заполнение пропущенных значений с уровня группировки  -->
            # заполнение медианной цены по категории года, этажу, кол-ву комнат
            idx_nan = df[pd.isna(df.med_year_floor_rooms_price)].index
            rooms = df.loc[idx_nan].drop(['med_year_floor_rooms_square',
                                          'med_year_floor_rooms_price',
                                          'med_year_floor_rooms_psqm'], axis=1)
            rooms['idx'] = rooms.index
            rooms = rooms.merge(self.med_year_floor_Rooms_price,
                                on=['year_cat', 'Floor', 'Rooms'], how='left')
            rooms = rooms.set_index('idx')
            df.loc[idx_nan, 'med_year_floor_rooms_square'] = rooms.loc[
                idx_nan, 'med_year_floor_rooms_square']
            df.loc[idx_nan, 'med_year_floor_rooms_price'] = rooms.loc[
                idx_nan, 'med_year_floor_rooms_price']
            df.loc[idx_nan, 'med_year_floor_rooms_psqm'] = rooms.loc[
                idx_nan, 'med_year_floor_rooms_psqm']
            # заполнение пропущенных значений с уровня группировки  -->
            # заполнение медианной цены по категории года, кол-ву комнат
            idx_nan = df[pd.isna(df.med_year_floor_rooms_price)].index
            # отметка что для поля "med_year_floor_rooms_price" не нашли данных
            # в группировке
            df['myfrp_good'] = 1
            df.loc[idx_nan, 'myfrp_good'] = 0
            df.loc[idx_nan, 'med_year_floor_rooms_square'] = df.loc[
                idx_nan, 'med_year_cat_rooms_square']
            df.loc[idx_nan, 'med_year_floor_rooms_price'] = df.loc[
                idx_nan, 'med_year_cat_rooms_price']
            df.loc[idx_nan, 'med_year_floor_rooms_psqm'] = df.loc[
                idx_nan, 'med_year_cat_rooms_psqm']

        # заполнение медианной цены по району и категории года
        if self.med_district_year_roomsq_price is not None:
            df = df.merge(self.med_district_year_roomsq_price,
                          on=['DistrictId', 'year_cat', 'roomsq'], how='left')
            # заполнение пропущенных значений с уровня группировки -->
            # заполнение медианной цены квартиры по району, году, комнатам
            idx_nan = df[pd.isna(df.med_district_year_rooms_price)].index
            rooms = df.loc[idx_nan].drop(['med_district_year_rooms_square',
                                          'med_district_year_rooms_price',
                                          'med_district_year_rooms_psqm'],
                                         axis=1)
            rooms['idx'] = rooms.index
            rooms = rooms.merge(self.med_district_year_Rooms_price,
                                on=['DistrictId', 'year_cat', 'Rooms'],
                                how='left')
            rooms = rooms.set_index('idx')
            df.loc[idx_nan, 'med_district_year_rooms_square'] = rooms.loc[
                idx_nan, 'med_district_year_rooms_square']
            df.loc[idx_nan, 'med_district_year_rooms_price'] = rooms.loc[
                idx_nan, 'med_district_year_rooms_price']
            df.loc[idx_nan, 'med_district_year_rooms_psqm'] = rooms.loc[
                idx_nan, 'med_district_year_rooms_psqm']
            # заполнение пропущенных значений с уровня группировки -->
            # заполнение медианной цены квартиры по району/году, среднее из них
            idx_nan = df[pd.isna(df.med_district_year_rooms_price)].index
            # отметка что для поля "med_district_year_rooms_price" не нашли
            # данных в группировке
            df['mdyrp_good'] = 1
            df.loc[idx_nan, 'mdyrp_good'] = 0
            sqr = (df.loc[idx_nan, 'med_DistrictId_rooms_square'] +
                   df.loc[idx_nan, 'med_year_cat_rooms_square']) / 2
            prs = (df.loc[idx_nan, 'med_DistrictId_rooms_price'] +
                   df.loc[idx_nan, 'med_year_cat_rooms_price']) / 2
            sqm = (df.loc[idx_nan, 'med_DistrictId_rooms_psqm'] +
                   df.loc[idx_nan, 'med_year_cat_rooms_psqm']) / 2
            df.loc[idx_nan, 'med_district_year_rooms_square'] = sqr
            df.loc[idx_nan, 'med_district_year_rooms_price'] = prs
            df.loc[idx_nan, 'med_district_year_rooms_psqm'] = sqm

            df['new_yfr'] = df.apply(self.find_min, axis=1)
            df['div_yfrdyr'] = df.med_year_floor_rooms_price / \
                               df.med_district_year_rooms_price
            df['new_div'] = df.new_yfr / df.med_district_year_rooms_price
            cnd = (df.div_yfrdyr < 1 / self.dif) | (df.div_yfrdyr > self.dif)
            df.loc[cnd, 'med_year_floor_rooms_price'] = df.loc[cnd, 'new_yfr']
            df.drop(['new_yfr', 'div_yfrdyr', 'new_div'], axis=1, inplace=True)

        # заполнение медианной цены по Social_1 и категории года
        if self.med_social_year_roomsq_price is not None:
            df = df.merge(self.med_social_year_roomsq_price,
                          on=['Social_1', 'year_cat', 'roomsq'], how='left')
            # заполнение пропущенных значений с уровня группировки -->
            # заполнение медианной цены квартиры по Social_1, году, комнатам
            idx_nan = df[pd.isna(df.med_social_year_rooms_price)].index
            rooms = df.loc[idx_nan].drop(['med_social_year_rooms_square',
                                          'med_social_year_rooms_price',
                                          'med_social_year_rooms_psqm'],
                                         axis=1)
            rooms['idx'] = rooms.index
            rooms = rooms.merge(self.med_social_year_Rooms_price,
                                on=['Social_1', 'year_cat', 'Rooms'],
                                how='left')
            rooms = rooms.set_index('idx')
            df.loc[idx_nan, 'med_social_year_rooms_square'] = rooms.loc[
                idx_nan, 'med_social_year_rooms_square']
            df.loc[idx_nan, 'med_social_year_rooms_price'] = rooms.loc[
                idx_nan, 'med_social_year_rooms_price']
            df.loc[idx_nan, 'med_social_year_rooms_psqm'] = rooms.loc[
                idx_nan, 'med_social_year_rooms_psqm']
            # заполнение пропущенных значений с уровня группировки -->
            # заполнение медианной цены квартиры по району/году, среднее из них
            idx_nan = df[pd.isna(df.med_social_year_rooms_price)].index
            # отметка что для поля "med_social_year_rooms_price" не нашли
            # данных в группировке
            df['msyrp_good'] = 1
            df.loc[idx_nan, 'msyrp_good'] = 0
            sqr = (df.loc[idx_nan, 'med_Social_1_rooms_square'] +
                   df.loc[idx_nan, 'med_district_year_rooms_square']) / 2
            prs = (df.loc[idx_nan, 'med_Social_1_rooms_price'] +
                   df.loc[idx_nan, 'med_district_year_rooms_price']) / 2
            sqm = (df.loc[idx_nan, 'med_Social_1_rooms_psqm'] +
                   df.loc[idx_nan, 'med_district_year_rooms_psqm']) / 2
            df.loc[idx_nan, 'med_social_year_rooms_square'] = sqr
            df.loc[idx_nan, 'med_social_year_rooms_price'] = prs
            df.loc[idx_nan, 'med_social_year_rooms_psqm'] = sqm

        # пометим первый и последние этажи в доме
        df['floor_first'] = 0
        df['floor_last'] = 0
        df.loc[(df.Floor == 1) & (df.HouseFloor > 1), 'floor_first'] = 1
        df.loc[(df.Floor == df.HouseFloor) & (df.HouseFloor > 1),
               'floor_last'] = 1

        if clusters:
            cluster_labels = self.make_clusters(df, clusters)
            df = pd.concat([df, cluster_labels], axis=1)

        # деление категорий по столбцам
        if self.dummy:
            # почему-то не срабатывает pd.get_dummies с drop_first=False
            df_dummy = pd.get_dummies(df[self.dummy], columns=self.dummy)
            df = df.merge(df_dummy, left_index=True, right_index=True)
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
        if 'Id' in df.columns:
            df = df.set_index('Id')

        return df


if __name__ == "__main__":
    # обучающая выборка
    train = pd.read_csv(FILE_TRAIN, index_col='Id')
    # тестовая выборка
    test = pd.read_csv(FILE_TEST, index_col='Id')

    preprocessor = DataPreprocessing()

    dataset = preprocessor.concat_df(train, test)

    print(f'Предобработка данных')
    start_time = time.time()
    preprocessor.fit(dataset)
    dataset = preprocessor.transform(dataset)
    preprocessor.write_dataset(dataset, FILE_PREPROCESS)
    print_time(start_time)

    dataset = ReadWriteDataset.read_dataset(FILE_PREPROCESS)

    print(f'Генерация новых признаков')
    start_time = time.time()
    features_gen = FeatureGenerator()
    features_gen.fit(dataset)
    # dataset = features_gen.transform(dataset, exclude_cols=['rooms_intsq'])
    dataset = features_gen.transform(dataset)
    print_time(start_time)

    dataset = memory_compression(dataset)

    features_gen.write_dataset(dataset, FILE_WITH_FEATURES)

    # если есть пустые значения - выведем на экран
    if dataset.drop(['Price', 'price_sqm'], axis=1).isna().sum().sum() > 0:
        print(dataset.drop(['Price', 'price_sqm'], axis=1).isna().sum())
