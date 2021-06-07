import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
#from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FuncFormatter, EngFormatter
from time import localtime, strftime, time

pd.set_option('display.max_columns', None)
pd.set_option('max_columns', None)


def check_commit():
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


def missing_values(df):
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
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Spearman", fontsize=22, fontweight="bold")
    plt.subplots_adjust(bottom=.26)
    plt.show()
    # corr = df.corr(method="pearson")
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.title("Pearson")
    # plt.show()


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


def distribution_per_variable(df, df_names):
    print_time_line_sep("Descriptive Statistics")
    for i in df_names:
        # visualize frequency distribution of variable
        f = plt.subplots(1, 1, figsize=(10, 8))
        # ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
        a = sns.displot(x=df[i], data=df, palette="Set1")
        plt.title(f'{i} distribution', fontsize=22, fontweight="bold")
        # plt.legend(fontsize='x-large')
        # plt.ylabel('', fontsize=20)
        plt.show()


def age_distribution():
    g = sns.displot(df, x=df['age'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    g.ax.set_title('Age distribution', fontsize=14, fontweight="bold")
    plt.show()


def workclass_distribution():
    workclass_names = df['workclass'].unique().tolist()
    print(workclass_names)
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


def education_num_distribution():
    """
    TODO: switch the numbers on the X axis with corresponding education (or find a way to sort education manually)
    :return:
    """
    workclass_names = df['education.num'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df['education.num'], discrete=True)
    plt.tight_layout()
    plt.xticks(np.arange(0, 17, 1))
    plt.title('Education num distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
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
    g = sns.displot(df, x=df['fnlwgt'], hue="income", bins=20, legend=False)
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
    g = sns.displot(df, x=df['capital.loss'], hue="income", bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital loss distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    # plt.xticks(np.arange(0, 1_200_000, 250_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


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


def distribution_workclass_income(df):
    class_order = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov',
                   'Without-pay', 'Never-worked']
    hue_order = ['<=50K', '>50K']
    f, ax = plt.subplots(figsize=(12, 8))
    ax = sns.countplot(x="workclass", hue="income", data=df, palette="rocket", order=class_order, hue_order=hue_order)
    ax.set_title("Frequency distribution of workclass vs income", fontsize=25, fontweight="bold")
    ax.legend(loc='upper right')
    counts = df.workclass.value_counts()
    i = 0
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(12)
    plt.legend(fontsize=20, loc='upper right')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / counts[i])
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center')
        i += 1
        if i == 8:
            i = 0
    plt.show()


def box_plots(df):
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle('Box-plots', fontweight="bold")
    df.boxplot(column=['age'], ax=axs[0, 0], widths=(0.5),  color= 'darkred')
    df.boxplot(column=['fnlwgt'], ax=axs[0, 1], widths=(0.5),  color= 'darkred')
    df.boxplot(column=['education.num'], ax=axs[0, 2], widths=(0.5), color= 'darkred')
    df.boxplot(column=['capital.gain'], ax=axs[1, 0], widths=(0.5),  color= 'darkred')
    df.boxplot(column=['capital.loss'], ax=axs[1, 1], widths=(0.5),  color= 'darkred')
    df.boxplot(column=['hours.per.week'], ax=axs[1, 2], widths=(0.5), color= 'darkred')
    plt.show()


def box_plot3(df):
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="income", y="age", hue="sex", data=df, palette="rocket")
    ax.set_title("Box plot: Income vs age and sex", fontsize=22, fontweight="bold")
    ax.legend(loc='upper right')
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(15)
    plt.legend(fontsize=17, loc='upper right')
    plt.show()


def income_plot(df):
    pair = sns.pairplot(df, hue="income", palette="rocket")
    plt.show()


def income_plot2(df):
    ## Income by age and hours per week
    ax = sns.scatterplot(x="age", y="hours.per.week", hue="income",
                         data=df, palette='rocket')
    ax.set_title("Income vs age and hours.per.week", fontsize=15, fontweight="bold")
    ax.legend(loc='upper right')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(12)
    plt.legend(fontsize=11, loc='upper right')
    plt.show()


def income_plot3(df):
    ## Income By age and race
    ax = sns.boxplot(x="income", y="age", hue="race",
                     data=df, palette="rocket")
    ax.set_title("Income vs age and race", fontsize=15, fontweight="bold")
    ax.legend(loc='upper right')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(12)
    plt.yticks(np.arange(10,130,10))
    plt.legend(fontsize=8, loc='upper right')
    plt.show()


def handle_MVs(df):
    print("Number of Missing values of 3 variables:")
    print(df.loc[df['occupation'].isnull() & (df['workclass'].isnull()) & (df['native.country'].isnull())].count())
    print("Number of Missing values of workclass and native.country:")
    print(df.loc[df['occupation'].isnull() & (df['workclass'].isnull())].count())
    """ Dropping records:"""
    df = df.drop(df[(df['occupation'].isnull() ) & (df['workclass'].isnull()) & df['native.country'].isnull()].index)
    print('The shape of the dataset : ', df.shape)
    df = df.drop(df[(df['occupation'].isnull()) & (df['workclass'].isnull())].index)
    print('The shape of the dataset : ', df.shape)
    return df

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


if __name__ == '__main__':
    df = read_data()
    print('The shape of the dataset : ', df.shape)
    # describe_df(df)
    df = fix_value_spaces_and_names(df)
    """ -----------------------------  """
    # TODO - to aggregation of occupation - professionals
    missing_values(df)
    df = handle_MVs(df)
    # missing_values(df)
    result = cramers_v( df['native.country'], df['capital.loss'])
    print(result)

    # print(df.describe(include='all').T)
    # target value
    # check_columns()
    # print(df['income'].isnull().sum())
    # print(df['income'].nunique())
    # print(df['income'].value_counts())
    # print(df["native.country"].str.startswith(' '))
    # corr_matrices(df)
    # stat, p, dof, expected = chi2_contingency(df['occupation'], df['workclass'])
    # plt.matshow(df.corr())
    # plt.show()
    # plot_descriptive_statistics(df)
    # age_distribution(df)

    # workclass_distribution()
    # distribution_per_variable(df,names)

    # X, y = df.iloc[:,:14], df.iloc[:, 14:15]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # tune_pipeline(pipeline, hyper_parameters_grid, X, y, n_folds=5, n_jobs=1)