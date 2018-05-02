import pandas as pd
import numpy as np
import re


def get_age_range(age):
    if np.isnan(age):
        return 'NaN'
    elif age < 15:
        return '< 15'
    elif age >= 15 and age < 30:
        return '>= 15 and < 30'
    elif age >= 30 and age < 45:
        return '>= 30 and < 45'
    elif age >= 45 and age < 60:
        return '>= 45 and < 60'
    elif age >= 60 and age < 75:
        return '>= 60 and < 75'
    elif age >= 75 and age < 90:
        return '>= 75 and < 90'
    else:
        return '>= 90'

def get_first_name(name):
    search_result = re.search('\s\((\w+)\s?', name)
    if search_result:
        return search_result.group(1)
    else:
        search_result = re.search('\.\s(\w+)\s?', name)
        if search_result:
            return search_result.group(1)
        else:
            return ''


titanic_df = pd.read_csv('data/titanic.csv')
titanic_df['AgeRange'] = titanic_df['Age'].apply(get_age_range)
titanic_df['FirstName'] = titanic_df['Name'].apply(get_first_name)

print('Sex')
print(titanic_df.groupby(['Sex', 'Survived'])['PassengerId'].count())
print()
print('Survived percent')
print(round(titanic_df[titanic_df['Survived'] == 1].groupby('Sex')['PassengerId'].count() / titanic_df.groupby('Sex')['PassengerId'].count() * 100, 2))
#print(round(titanic_df.groupby(['Sex', 'Survived'])['PassengerId'].count() / titanic_df.groupby(['Sex'])['PassengerId'].count() * 100, 2))
print()

print('AgeRange')
print(titanic_df.groupby(['AgeRange', 'Survived'])['PassengerId'].count())
print()
print('Survived percent')
print(round(titanic_df[titanic_df['Survived'] == 1].groupby('AgeRange')['PassengerId'].count() / titanic_df.groupby('AgeRange')['PassengerId'].count() * 100, 2))
#print(round(titanic_df.groupby(['AgeRange', 'Survived'])['PassengerId'].count() / titanic_df.groupby(['AgeRange'])['PassengerId'].count() * 100, 2))
print()

print('Pclass')
print(titanic_df.groupby(['Pclass', 'Survived'])['PassengerId'].count())
print()
print('Survived percent')
print(round(titanic_df[titanic_df['Survived'] == 1].groupby('Pclass')['PassengerId'].count() / titanic_df.groupby('Pclass')['PassengerId'].count() * 100, 2))
#print(round(titanic_df.groupby(['Pclass', 'Survived'])['PassengerId'].count() / titanic_df.groupby(['Pclass'])['PassengerId'].count() * 100, 2))
print()

print('Pivot table')
print(pd.pivot_table(titanic_df, values = 'PassengerId', index = ['Pclass', 'Sex'], aggfunc = len))
print()

print('FirstName stat')
print(titanic_df.groupby(['Sex', 'FirstName'])['PassengerId'].count().groupby(['Sex']).nlargest(3))
print()