import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FuncFormatter, EngFormatter
from time import localtime, strftime, time

pd.set_option('display.max_columns', None)
pd.set_option('max_columns', None)


def first_commit():
    pass


def print_time_line_sep(msg):
    """
    :param msg: massage to print
    print len sep with current date time.
    """
    dt_fmt = "%d/%m/%Y %H:%M:%S"
    line_msg = " " * 41 + msg
    line_sep = "-" * 35 + " " + strftime(dt_fmt, localtime()) + " " + "-" * 35 + '\n'
    print(line_msg)
    print(line_sep)


def read_data():
    data = 'data.csv'
    df = pd.read_csv(data)
    return df


def missing_values():
    """ Missing values """
    print_time_line_sep("Missing values")
    print(df.isnull().sum())
    print(df.isnull().mean().round(4))  # It returns percentage of missing values in each column in the dataframe.

    # assert that there are no missing values in the dataframe
    # assert pd.notnull(df).all().all(), "missing values exits!"


def fix_value_spaces_and_names(df):
    df = df.replace(to_replace=[" <=50K.", " <=50K"], value="<=50K")
    df = df.replace(to_replace=[" >50K", " >50K."], value=">50K")
    df = df.replace(to_replace=[" ?", "?"], value=np.nan)
    # df = df.replace(to_replace=[np.nan], value=None)
    print(df.columns)
    cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    df[cols] = df[cols].apply(lambda x: x.str.strip())
    # df[cols].apply(lambda x: print(pd.unique(df['workclass']).tolist()))
    # print(df.isin([' ? ']).any())
    # column_names = df.columns.tolist()
    # print(type(df['income'].iloc[0:1]))
    # for i in range(len(column_names)):
    #     print("%%%%", i, type(df[column_names[i]].iloc[0:1].dtypes))
    #     df[df.columns] = df.apply(lambda x: x.str.strip())
    #     # if type(df[column_names[i]].iloc[0:1]) is object:
    #     #     print(i, ": ", df[column_names[i]])
    #     #     df[column_names[i]] = df[column_names[i]].strip()
    #     # else:
    #     #     print(i, df[column_names[i]].iloc[0:1])

    return df



def describe_df(df):
    print_time_line_sep("describe data")
    print(df.describe(include=['object']).T)
    print(df.describe().T)


def corr_matrices(df):
    corr = df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Spearman")
    plt.show()
    corr = df.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Pearson")
    plt.show()


def check_columns():
    column_names = df.columns.tolist()
    for i in range(len(column_names)):
        print(column_names[i])
        print(df[column_names[i]].isnull().sum())
        print(df[column_names[i]].nunique())
        print(df[column_names[i]].value_counts())
        if type(df[column_names[i]]) is str:
            print(df[column_names[i]].str.startswith(' '), '\n')


def income_general_distribution():
    print_time_line_sep("Descriptive Statistics")
    # # visualize frequency distribution of income variable
    f = plt.subplots(1, 1, figsize=(10, 8))
    # ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
    df['income'].value_counts().plot(kind='pie',   autopct='%1.1f%%', colors=["red", "green"],
            startangle=90, shadow=False,  legend=True, fontsize=19, labels=["<=50K$",">50K$"])
    plt.title('Income distribution', fontsize=22, fontweight="bold")
    plt.legend(fontsize='x-large')
    plt.ylabel('', fontsize=20)
    plt.show()


# def distribution_per_variable(df, df_names):
#     print_time_line_sep("Descriptive Statistics")
#     for i in df_names:
#         # visualize frequency distribution of variable
#         f = plt.subplots(1, 1, figsize=(10, 8))
#         # ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
#         a = sns.displot(x=df[i], data=df, palette="Set1")
#         plt.title(f'{i} distribution', fontsize=22, fontweight="bold")
#         # plt.legend(fontsize='x-large')
#         # plt.ylabel('', fontsize=20)
#         plt.show()


def age_distribution():
    g = sns.displot(df, x=df['age'],hue="income", bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    g.ax.set_title('Age distribution', fontsize=14, fontweight="bold")
    plt.show()


def workclass_distribution():
    workclass_names = df['workclass'].unique().tolist()
    ax = sns.histplot(data=df['workclass'], shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(8), xlabels_new)
    plt.tight_layout()
    plt.title('Work class distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def education_distribution():
    workclass_names = df['education'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df['education'], shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(16), xlabels_new, rotation=60)
    plt.tight_layout()
    plt.title('Education distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()
    plt.show()


def education_num_distribution():
    """
    TODO: switch the numbers on the X axis with corresponding education (or find a way to sort education manually)
    :return:
    """

    education = df['education'].unique().tolist()
    education_num = df['education.num'].unique().tolist()
    education_sorted = [x for _, x in sorted(zip(education_num, education))]
    xlabels_new = [label.replace('-', '-\n') for label in education_sorted]
    xlabels_new = [""] + xlabels_new
    plt.xticks(range(17), xlabels_new, rotation=90)

    ax = sns.histplot(data=df['education.num'], discrete=True)
    plt.tight_layout()
    plt.xticks(np.arange(0, 17, 1))
    plt.title('Education num distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    # ax.set_xticklabels(education_sorted, minor=False, rotation=90)
    plt.show()


def marital_status_distribution():
    workclass_names = df['marital.status'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df['marital.status'], shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(7), xlabels_new, rotation=0)
    plt.tight_layout()
    plt.title('Marital status distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def relationship_distribution():
    workclass_names = df['relationship'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df['relationship'], shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(6), xlabels_new, rotation=0)
    plt.tight_layout()
    plt.title('Relationship distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def occupation_distribution():
    workclass_names = df['occupation'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df['occupation'], shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(14), xlabels_new, rotation=90)
    plt.tight_layout()
    plt.title('Occupation distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def race_distribution():
    workclass_names = df['race'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df['race'], shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(5), xlabels_new, rotation=0)
    plt.tight_layout()
    plt.title('Race distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def sex_distribution():
    ax = sns.histplot(data=df['sex'], shrink=.9)
    plt.tight_layout()
    plt.title('Gender distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def hours_per_week_distribution():
    g = sns.displot(df, x=df['hours.per.week'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Hours per week distribution', fontsize=14, fontweight="bold")
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def fnlwgt_distribution():
    g = sns.displot(df, x=df['fnlwgt'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('fnlwgt distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    plt.xticks(np.arange(0, 1_200_000, 250_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def capital_gain_distribution():
    g = sns.displot(df, x=df['capital.gain'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital gain distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    plt.xticks(np.arange(0, 100_000, 20_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def capital_loss_distribution():
    g = sns.displot(df, x=df['capital.loss'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital loss distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    # plt.xticks(np.arange(0, 1_200_000, 250_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def sex_vs_income():
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.countplot(x="income", hue="sex", data=df, palette=['#0000FF',"#ff0000"])
    ax.set_title("Distribution of Income vs Gender", fontsize=22, fontweight="bold")
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.xticks(fontsize=14)
    plt.show()


#TODO make it look beter (display only us, mexico)
def native_country_distribution():
    def autopct(pct):  # only show the label when it's > 10%
        return ('%.2f' % pct) if pct > 10 else ''

    my_labels = df['native.country'].unique().tolist()

    ax = df['native.country'].value_counts().plot(kind='pie', figsize=(28, 12), autopct=autopct)
    ax.axes.get_yaxis().set_visible(False)

    # ax = df['native.country'].value_counts().plot(kind='pie',   autopct='%1.1f%%',
    #         startangle=90, shadow=False)
    # plt.title('Native country distribution', fontsize=22, fontweight="bold")
    # plt.ylabel('', fontsize=20)
    plt.show()


def plot_descriptive_statistics():
    age_distribution()
    workclass_distribution()
    fnlwgt_distribution()
    education_distribution()
    education_num_distribution()
    occupation_distribution()
    relationship_distribution()
    marital_status_distribution()
    race_distribution()
    sex_distribution()
    capital_gain_distribution()
    capital_loss_distribution()
    hours_per_week_distribution()
    native_country_distribution()
    income_general_distribution()
    # # Visualize income wrt race # uncomment to plot separately
    #  f, ax = plt.subplots(figsize=(10, 8))
    # ax[0] = sns.countplot(x="income", hue="race", data=df, palette="Set1") #
    # ax[0].set_title("Frequency distribution of income variable wrt race")
    # # # ax[1] = sns.countplot(x="income", hue="race", data=df, palette="Set1") #
    # # # ax[1].set_title("Frequency distribution of income variable wrt race")
    # plt.show()


def handle_missing_values():
    df['workclass'].fillna(df['workclass'].mode()[0], inplace=True)
    df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
    df['native.country'].fillna(df['native.country'].mode()[0], inplace=True)


if __name__ == '__main__':
    df = read_data()
    names = df.columns.tolist()
    df = fix_value_spaces_and_names(df)
    df.sort_values(by=['education.num'])
    # check_columns()
    print("############", df['workclass'].mode())
    # missing_values()
    # workclass_distribution()
    # occupation_distribution()
    print(df.isnull().values.any(axis=1))
    handle_missing_values()
    print("############", df['workclass'].mode())
    missing_values()
    # missing_values()















# if __name__ == '__main__':
#     df = read_data()
#     names = df.columns.tolist()
#     print('The shape of the dataset : ', df.shape)
#     # df.info()
#     # describe_df(df)
#     df = fix_value_spaces_and_names(df)
#     df.sort_values(by=['education.num'])
#     missing_values(df)
#     # handle_missing_values()
#     missing_values(df)
#     # sex_vs_income()
#
#     # education_num_distribution()
#     # print(df.describe(include='all').T)
#     # target value
#     # check_columns()
#     # print(df['income'].isnull().sum())
#     # print(df['income'].nunique())
#     # print(df['income'].value_counts())
#     # print(df["native.country"].str.startswith(' '))
#     # corr_matrices(df)
#     # stat, p, dof, expected = chi2_contingency(df['occupation'], df['workclass'])
#
#
#     # plt.matshow(df.corr())
#     # plt.show()
#
#     # plot_descriptive_statistics(df)
#     # age_distribution(df)
#
#     # workclass_distribution()
#     # distribution_per_variable(df,names)
#
#     # X, y = df.iloc[:,:14], df.iloc[:, 14:15]
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     # tune_pipeline(pipeline, hyper_parameters_grid, X, y, n_folds=5, n_jobs=1)