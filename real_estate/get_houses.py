import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup
import time

save_path = r'D:\python-txt\real_estate\htmls'
base_url = 'https://dom.mingkh.ru/moskva/moskva/houses'

# start_time = time.time()
# url = base_url
# for page in range(1, 323):
#     if page > 1:
#         url = f'{base_url}?page={page}'
#     print(f'Скачиваю страницу: {url}')
#     html_text = requests.get(url).text
#     with open(os.path.join(save_path, f'houses_page{page}.html'), 'w') as file:
#         file.write(html_text)
#     # file_df = pd.read_html(html_text)[0]
#     # print(file_df)
#     # time.sleep(2)
# time_apply = time.time() - start_time
# print(f'Время обработки: {time_apply:.1f} сек')

name_columns = ['pos', 'city', 'address', 'area', 'HouseYear', 'HouseFloor']
df_houses = pd.DataFrame(columns=name_columns)

# for fileHTML in os.listdir(save_path):
#     print(fileHTML)
#     with open(os.path.join(save_path, fileHTML)) as file:
#         html_text = file.read()
#         file_df = pd.read_html(html_text)[0]
#         file_df.columns = name_columns
#         df_houses = pd.concat([df_houses, file_df], axis=0)
# df_houses.sort_values('pos', inplace=True)
# df_houses = df_houses.set_index('pos')
# print(df_houses)
# df_houses.to_csv(r'D:\python-txt\real_estate\moscow_houses.csv', sep=';')

# посмотрим на распределение этажности домов по годам с сайта
# https://dom.mingkh.ru/moskva/moskva/
df_houses = pd.read_csv(r'D:\python-txt\real_estate\moscow_houses.csv',
                        sep=';', index_col='pos')
df_houses['area'] = pd.to_numeric(df_houses['area'], errors='coerce')
df_houses['HouseYear'] = pd.to_numeric(df_houses['HouseYear'], errors='coerce')
df_houses['HouseFloor'] = pd.to_numeric(df_houses['HouseFloor'],
                                        errors='coerce')
df_houses.dropna(inplace=True)
df_houses.HouseYear = df_houses.HouseYear.astype(int)
df_houses.HouseFloor = df_houses.HouseFloor.astype(int)
df_houses.HouseYear = df_houses.HouseYear - 1
print(df_houses.info())
grp_hy = df_houses[df_houses.HouseYear >= 1908].groupby('HouseYear',
                                                        as_index=False).HouseFloor.max()
grp_hy.sort_values('HouseYear')
div_years = grp_hy.loc[len(grp_hy) // 2 + 1, 'HouseYear']
for num_part in (0, 1):
    year_query = f'HouseYear {["<", ">="][num_part]} {div_years}'
    y_min = grp_hy.query(year_query).HouseFloor.min()
    y_max = grp_hy.query(year_query).HouseFloor.max() + 1
    if y_min > 1:
        y_min -= 1
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.barplot(x='HouseYear', y='HouseFloor', data=grp_hy.query(year_query))
    ax.set_ylim(y_min, y_max)
    plt.yticks([y for y in range(y_min, y_max)])
    ax.grid()
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.show()
# tmp = df_houses.query('HouseYear in (1952, 1953, 1954) and HouseFloor > 16')
tmp = df_houses.query('HouseYear in (1967,) and HouseFloor > 22')
tmp.to_csv('tmp.csv', sep=';', encoding='cp1251')
