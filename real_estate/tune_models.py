from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor
from lightgbm import LGBMRegressor, Dataset
# from catboost import CatBoostRegressor

from process_data import *

warnings.filterwarnings("ignore")
USE_CORES = os.cpu_count() - 1


def process_model(use_model=RandomForestRegressor(random_state=SEED),
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

    def iter_folds(n_fold, verb=0):
        """
        Итерация для поиска лучшей модели для заданного количества флодов
        :param n_fold: количество фолдов
        :param verb: = 1 - отображать процесс
        :return: r2_score_train, r2_score_valid, модель
        """
        skf = KFold(n_splits=n_fold, random_state=SEED, shuffle=True)
        search_cv = GridSearchCV(use_model, params, cv=skf, scoring='r2',
                                 verbose=verb, n_jobs=USE_CORES)
        search_cv.fit(X_train, y_train)
        best_tree_cv = search_cv.best_estimator_
        y_train_pred = best_tree_cv.predict(X_train)
        y_valid_pred = best_tree_cv.predict(X_valid)
        r2_score_train = r2(y_train, y_train_pred)
        r2_score_valid = r2(y_valid, y_valid_pred)
        print(f'folds={n_fold:2d}, r2_train={r2_score_train:0.7f},'
              f' r2_valid={r2_score_valid:0.7f}'
              f' best_params={search_cv.best_params_}')
        return r2_score_train, r2_score_valid, search_cv, y_train_pred, y_valid_pred

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
    file_submit_csv = os.path.join(PATH_EXPORT, 'predictions',
                                   f'{submit_prefix}submit.csv')

    if folds_range:
        print('Поиск лучших параметров...')
    start_time_cv = time.time()
    best_folds = []
    for folds in folds_range:
        r2_trn, r2_vld, search, _, _ = iter_folds(folds, verbose)
        best_folds.append([r2_trn, r2_vld, folds, search.best_params_, search])
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
        r2_trn, r2_vld, search, trn_p, vld_p = iter_folds(fold_single, verbose)
        best_tree = search.best_estimator_
        print(best_tree)
        print()
        evaluate_preds(y_train, trn_p, y_valid, vld_p)

        feat_imp = best_tree.feature_importances_
        feat_imp_df = pd.DataFrame({'features': X.columns.values,
                                    'importances': feat_imp})
        feat_imp_df.sort_values('importances', ascending=False, inplace=True)
        # предсказание
        submit = pd.read_csv(FILE_SAMPLE, index_col='Id')
        submit['Price'] = best_tree.predict(test_df)
        date_now = datetime.now()
        time_stamp = date_now.strftime('%y%m%d%H%M')
        submit.to_csv(file_submit_csv.replace('.csv', f'_{time_stamp}.csv'))
        # сохранение результатов итерации в файл
        file_name = os.path.join(PATH_EXPORT, 'results.csv')
        if os.path.exists(file_name):
            file_df = pd.read_csv(file_name)
            if 'mdl' not in file_df.columns:
                file_df.insert(1, 'mdl', '')
            if 'category_columns' not in file_df.columns:
                file_df.insert(10, 'category_columns', 'unknown list')
            file_df.time_stamp = pd.to_datetime(file_df.time_stamp,
                                                format='%y-%m-%d %H:%M:%S')
            d1 = datetime(2021, 2, 28, 13, 40)
            d2 = datetime(2021, 2, 28, 22, 00)
            cnd = pd.isna(file_df.mdl)
            file_df.loc[cnd, 'mdl'] = file_df[cnd].time_stamp.apply(
                lambda x: 'RF' if x < d1 else 'LG' if x > d2 else 'ET')
            file_df.time_stamp = file_df.time_stamp.dt.strftime(
                '%y-%m-%d %H:%M:%S')
            if 'r2_score' in file_df.columns:
                file_df = file_df.rename(columns={'r2_score': 'r2_train'})
                file_df.insert(5, 'r2_valid', np.NaN)
            # print(file_df[['time_stamp', 'mdl', 'r2_score']])
            # print(file_df.info())
        else:
            file_df = pd.DataFrame()
        time_stamp = date_now.strftime('%y-%m-%d %H:%M:%S')
        features_list = feat_imp_df.to_dict(orient='split')['data']
        temp_df = pd.DataFrame({'time_stamp': time_stamp,
                                'mdl': submit_prefix[:2].upper(),
                                'max_depth': search.best_params_['max_depth'],
                                'folds': fold_single,
                                'r2_train': r2_trn,
                                'r2_valid': r2_vld,
                                'best_params': [search.best_params_],
                                'features': [features_list],
                                'column_dummies': [features_gen.dummy],
                                'model_columns': [model_columns],
                                'category_columns': [category_columns],
                                'learn_exclude': [learn_exclude]
                                })

        file_df = file_df.append(temp_df)
        file_df.r2_train = file_df.r2_train.round(7)
        file_df.r2_valid = file_df.r2_valid.round(7)
        file_df.to_csv(file_name, index=False)
        file_df.name = 'results'
        # экспорт в эксель
        export_to_excel(file_df)
        print_time(start_time_cv)
        return best_tree, feat_imp_df
    else:
        return best_folds


def find_depth(use_model, not_sklearn=False, show_plot=True):
    print(use_model)
    # Подберем оптимальное значение глубины обучения дерева.
    scores = pd.DataFrame(columns=['max_depth', 'train_score', 'valid_score'])
    max_depth_values = range(3, 15)
    for max_depth in max_depth_values:
        print(f'max_depth = {max_depth}')
        if not_sklearn:
            find_model = use_model(random_state=SEED, max_depth=max_depth,
                                   num_leaves=63)
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


dataset = pd.DataFrame()

# чтение предобработанного датасета
dataset = ReadWriteDataset.read_dataset(FILE_PREPROCESS)

print(f'Генерация новых признаков')
gen_time = time.time()
features_gen = FeatureGenerator()
features_gen.fit(dataset)
# эти колонки уберем из датасета
exclude_columns = ['rooms_intsq']
# dataset = features_gen.transform(dataset)
dataset = features_gen.transform(dataset, clusters=0,
                                 exclude_cols=exclude_columns)
cat_features = features_gen.cat_features
print_time(gen_time)

dataset = memory_compression(dataset)

# print(dataset.info())

# все колонки для обучения
learn_columns = features_gen.learn_columns

# эти колонки исключаем из обучения
add_exclude = ['floor_first', 'year_cat', ]

add_exclude = ['rm_ratio_good',
               'other_sq_good',
               'house_fl_good',
               'house_fl_nonan',
               'house_hy_good',
               'rooms_good',
               'square_good',
               'livesq_good',
               # 'orig_both_sq',
               # 'livesq_nonan',
               'kitchen_good',
               'floor_first',  # этот был
               # 'floor_last',
               'year_cat',  # этот был
               'quart_sq',  # этот был
               ]

# уберем более крупные группировки
learn_exclude = ['myfrp_good', 'mdyrp_good', 'msyrp_good',
                 'med_year_cat_rooms_square',
                 'med_year_cat_rooms_price',
                 'med_year_cat_rooms_psqm',
                 # 'med_DistrictId_rooms_square',
                 # 'med_DistrictId_rooms_price',
                 # 'med_DistrictId_rooms_psqm',

                 'med_Social_1_rooms_square',
                 'med_Social_1_rooms_price',
                 'med_Social_1_rooms_psqm',
                 'med_Social_2_rooms_square',
                 'med_Social_2_rooms_price',
                 'med_Social_2_rooms_psqm',
                 'med_Social_3_rooms_square',
                 'med_Social_3_rooms_price',
                 'med_Social_3_rooms_psqm',
                 'med_Shops_1_rooms_square',
                 'med_Shops_1_rooms_price',
                 'med_Shops_1_rooms_psqm',

                 'med_year_floor_rooms_square',
                 'med_year_floor_rooms_price',
                 'med_year_floor_rooms_psqm',
                 'med_district_year_rooms_square',
                 'med_district_year_rooms_price',
                 'med_district_year_rooms_psqm',
                 'med_social_year_rooms_square',
                 'med_social_year_rooms_price',
                 'med_social_year_rooms_psqm',
                 ]

# add_exclude = []

learn_exclude.extend(add_exclude)
print(learn_exclude)
# колонки для обучения
model_columns = [col for col in learn_columns if col not in learn_exclude]
category_columns = [col for col in cat_features if col in model_columns]
print(model_columns)
print(category_columns)

# обучающий датасет
train_df = dataset[dataset.learn == 1][model_columns]
# тестовый датасет
test_df = dataset[dataset.learn == 0][model_columns]
test_df.drop('Price', axis=1, inplace=True)

X = train_df.drop('Price', axis=1)
y = train_df['Price']

txt = ('Размер ', ' пропусков ')
print(f'{txt[0]}трейна: {X.shape}{txt[1]}{X.isna().sum().sum()}')
print(f'{txt[0]}теста: {test_df.shape}{txt[1]}{test_df.isna().sum().sum()}')

# Добавление еще одного признака: кластера
# отмасштабируем данные - c ними результат на каггле немного различается
# scaled=0.69650, без маштабирования 0.69622 при 'boosting_type': ['gbdt']
# scaled=0.70706, без маштабирования 0.70832 при 'boosting_type': ['dart']
# c кластером для 'boosting_type': ['gbdt'] = 0.69799
# c кластером для 'boosting_type': ['dart'] = 0.70318
#

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2,
                                                      shuffle=True,
                                                      random_state=SEED,
                                                      stratify=X.Social_1
                                                      )
print()
print(f'{txt[0]}X_train: {X_train.shape}{txt[1]}{X_train.isna().sum().sum()}')
print(f'{txt[0]}X_valid: {X_valid.shape}{txt[1]}{X_valid.isna().sum().sum()}')

# X_train, X_valid, y_train, y_valid = X, X, y, y

# определение моделей
# model = RandomForestRegressor(random_state=SEED, criterion='mse')
# model = ExtraTreesRegressor(random_state=SEED, criterion='mse')
# model = GradientBoostingRegressor(random_state=SEED, criterion='mse')
model = LGBMRegressor(random_state=SEED, num_leaves=63)
# model = CatBoostRegressor(random_state=SEED, loss_function='RMSE',
#                           silent=True, cat_features=category_columns)

# find_depth(RandomForestRegressor)
# find_depth(ExtraTreesRegressor)
# find_depth(GradientBoostingRegressor)
# find_depth(LGBMRegressor, True)

# настройки для первого приближения: поиск глубины деревьев и количества фолдов
# f_params = {'max_depth': list(range(4, 15))}
f_params = {'max_depth': list(range(4, 9))}
# раскомментарить эту строку для расчета
# process_model(model, params=f_params, folds_range=list(range(3, 8)))

# [0.9179397450812147, 0.8652495194653406, 3, {'max_depth': 5}]
# [0.9179397450812147, 0.8652495194653406, 4, {'max_depth': 5}]
# [0.9179397450812147, 0.8652495194653406, 5, {'max_depth': 5}]
# [0.9179397450812147, 0.8652495194653406, 6, {'max_depth': 5}]
# [0.9179397450812147, 0.8652495194653406, 7, {'max_depth': 5}]
# Время обработки: 0 час 0 мин 34.6 сек
# [0.8940403331205953, 0.871717598194723, 3, {'max_depth': 4}]
# [0.9114084031895705, 0.871530636776139, 3, {'max_depth': 5}]
# [0.9258820537598811, 0.87135670974162,  3, {'max_depth': 6}]
# [0.9354736819991437, 0.866309550205158, 3, {'max_depth': 7}]
# GradientBoostingRegressor
# [0.9075486325532687, 0.8763868168315355, 5, {'max_depth': 4}]
# [0.9290640150483941, 0.8718722360723892, 5, {'max_depth': 5}]
# [0.9515302994536885, 0.8684971259076733, 5, {'max_depth': 6}]
# [0.9696113285145207, 0.8606174529015522, 5, {'max_depth': 7}]
# [0.9819079241087404, 0.8574458345253455, 5, {'max_depth': 8}]


# models = []
# for depth in range(3, 11):
#     param = {'max_depth': [depth]}
#     mdl = process_model(model, params=param, folds_range=list(range(3, 6)))
#     models.append(mdl[0][:4])
# models.sort(key=lambda x: (-x[1], x[2]))
# print()
# for elem in models:
#     print(elem)

# [0.9002480120931018, 0.8608351509623726, 3, {'max_depth': 4}]
# [0.8842216766772993, 0.8602982079128827, 3, {'max_depth': 3}]
# [0.9151301808739283, 0.8590609167827505, 3, {'max_depth': 5}]
# [0.9305347446789762, 0.8581118658582922, 3, {'max_depth': 6}]
# [0.9493536751237575, 0.8569095712643164, 3, {'max_depth': 9}]
# [0.9451563642347535, 0.855631587235926, 3, {'max_depth': 8}]
# [0.9412367443481793, 0.8542161618366814, 3, {'max_depth': 7}]

# Зададим параметры при max_depth = 18 для подбора параметров
# и отдыхаем несколько часов
f_params = {'n_estimators': list(range(100, 701, 100)),
            'max_depth': [5],
            # 'min_samples_leaf': list(range(1, 11, 1)),
            # 'min_samples_split': list(range(2, 22, 1)),  # не меньше 2
            }
# раскомментарить эту строку для расчета
# process_model(model, params=f_params, folds_range=[7], verbose=1)

# [0.9179397450812147, 0.8652495194653406, 3,
# {'max_depth': 5, 'n_estimators': 100}]
# Время обработки: 0 час 0 мин 9.8 сек
# [0.8940403331205953, 0.871717598194723, 3,
# {'max_depth': 4, 'n_estimators': 100}]
# Время обработки: 0 час 0 мин 14.2 сек


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

f_params = {
    'boosting_type': ['dart'],
    # 'boosting_type': ['dart', 'gbdt'],
    # 'n_estimators': [1000],
    'n_estimators': list(range(900, 1301, 50)),
    # 'max_bin': [512],
    'max_depth': [5],
    # 'learning_rate': [.05]
    'learning_rate': [.005, .01, .025, .05, 0.1]
    # 'min_samples_leaf': list(range(1, 9, 2)),
    # 'min_samples_split': list(range(2, 9, 2)),  # не меньше 2
    # 'min_samples_leaf': [3],
    # 'min_samples_split': [19]
}
# раскомментарить эту строку для расчета
# _, feat_imp_df_ = process_model(model, params=f_params, fold_single=5,
#                                 verbose=1, build_model=True)
print(feat_imp_df_)
