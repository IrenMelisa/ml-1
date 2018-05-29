import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def get_dataframe():
    data_dir = 'data'
    files = [os.path.join(data_dir, file) for file in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, file)) and file.startswith('SaleML_')]
    result = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    result = result.drop(['Дата накладной', 'Цена без НДС'], axis=1)

    # Преобразование категориальных полей
    mapping = {
        'День недели': {'Пн': 1, 'Вт': 2, 'Ср': 3, 'Чт': 4, 'Пт': 5, 'Сб': 6, 'Вс': 7},
        'Месяц': {'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6, 'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12},
        'Рабочий день': {'Да': 1, 'Нет': 0},
        'День перед праздником': {'Да': 1, 'Нет': 0},
        'День после праздника': {'Да': 1, 'Нет': 0},
        'День перед длинными выходными': {'Да': 1, 'Нет': 0},
        'День после длинных выходных': {'Да': 1, 'Нет': 0},
        'Наличие собственного склада': {'да': 1, 'нет': 0, 'Требует уточнения': -1},
        'Закупочный товарооборот':   {
            'Требует уточнения': 1,
            'Бюджет': 2,
            'Прочие продажи': 3,
            'Разовые сделки': 4,
            'до 50 тыс': 5,
            '50 тыс - 100 тыс': 6,
            '100 тыс - 200 тыс': 7,
            '200 тыс - 500 тыс': 8,
            '500 тыс - 1 млн': 9,
            '1-2 млн': 10,
            '2-3 млн': 11,
            '3-5 млн': 12,
            '5-8 млн': 13,
            '8-10 млн': 14,
            '10-15 млн': 15,
            '15-20 млн': 16,
            '20-25 млн': 17,
            '25-30 млн': 18,
            '30-35 млн': 19,
            '35-40 млн': 20,
            '40-50 млн': 21,
            '50-60 млн': 22,
            '60-80 млн': 23,
            '80-100 млн': 24,
            'более 100 млн': 25
        },
        'Доля БаДМ в закупках': {
            'Требует уточнения': 1,
            'Разовые сделки': 2,
            '<10%': 3,
            '10-20%': 4,
            '20-25%': 5,
            '25-30%': 6,
            '30-35%': 7,
            '35-40%': 8,
            '40-45%': 9,
            '45-50%': 10,
            '50-55%': 11,
            '55-60%': 12,
            '60-65%': 13,
            '65-70%': 14,
            '>70%': 15
        }
    }
    result = result.replace(mapping)
    series = ['Задержка доставки', 'Регулярность во времени', 'Регулярность по количеству', 'Регулярность по сумме']
    result[series] = result[series].fillna(value=0)

    return result


def main():
    df = get_dataframe()

    y = df['Количество товара']
    df = df.drop(['Количество товара', 'Сумма без НДС'], axis=1)

    # Для определения точности используем 20%; начального набора данных
    x_train, x_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.2, random_state=17)

    # DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=11, random_state=17)
    tree.fit(x_train, y_train)
    x_pred = tree.predict(x_holdout)
    print('DecisionTreeClassifier accuracy score: ' + str(accuracy_score(y_holdout, x_pred)))

    # RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100, max_depth=11, n_jobs=-1, random_state=17)
    forest.fit(x_train, y_train)
    x_pred = forest.predict(x_holdout)
    print('RandomForestClassifier accuracy score: ' + str(accuracy_score(y_holdout, x_pred)))


if __name__ == "__main__":
    main()
