import random

import pandas as pd
import numpy as np
from time import localtime, strftime, time
import joblib
import itertools

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FuncFormatter, EngFormatter

import seaborn as sns

from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,\
    roc_auc_score, roc_curve, f1_score

train_mode = False

pd.set_option('display.max_columns', None)
pd.set_option('max_columns', None)


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
    data_frame = pd.read_csv(data)
    return data_frame


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
    income_map = {'<=50K': 0, '>50K': 1}
    df['income'] = df['income'].map(income_map)
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


def age_distribution():
    g = sns.displot(df, x='age', hue='income', bins=20, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    g.ax.set_title('Age distribution', fontsize=14, fontweight="bold")
    plt.show()


def workclass_distribution():
    df_1 = df.replace(to_replace=[np.nan], value=None)
    workclass_names = df_1['workclass'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df_1, x='workclass',hue='income', legend='full', multiple="stack", shrink=.9)
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
    ax = sns.histplot(data=df,x='education', hue='income', legend='full', multiple="stack", shrink=.9)
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
    education = df['education'].unique().tolist()
    education_num = df['education.num'].unique().tolist()
    education_sorted = [x for _, x in sorted(zip(education_num, education))]
    xlabels_new = [label.replace('-', '-\n') for label in education_sorted]
    xlabels_new = [""] + xlabels_new
    plt.xticks(np.arange(0, 17, 1), xlabels_new, rotation=90)
    ax = sns.histplot(data=df, x='education.num', hue='income', legend='full', multiple="stack")
    plt.tight_layout()
    plt.title('Education num distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def marital_status_distribution():
    workclass_names = df['marital.status'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df, x='marital.status', shrink=.9, hue='income', legend='full', multiple="stack")
    # data = df, x = 'education.num'
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
    ax = sns.histplot(data=df, x='relationship', shrink=.9, hue='income', legend='full', multiple="stack")
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
    df_1 = df.replace(to_replace=[np.nan], value=None)
    workclass_names = df_1['occupation'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df_1, x='occupation', hue='income', legend='full', multiple="stack")
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
    ax = sns.histplot(data=df, x='race', shrink=.9, hue='income', legend='full', multiple="stack")
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
    ax = sns.histplot(data=df, x='sex', shrink=.9, hue='income', legend='full', multiple="stack")
    plt.tight_layout()
    plt.title('Gender distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def hours_per_week_distribution():
    g = sns.displot(df, x='hours.per.week', hue='income', bins=20, legend='full', multiple="stack")
    # g = sns.displot(df, x=df['hours.per.week'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Hours per week distribution', fontsize=14, fontweight="bold")
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def fnlwgt_distribution():
    g = sns.displot(df, x='fnlwgt', hue='income', bins=20, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('fnlwgt distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    plt.xticks(np.arange(0, 1_200_000, 250_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def capital_gain_distribution():
    g = sns.displot(df, x='capital.gain', hue='income', bins=10, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital gain distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    plt.xticks(np.arange(0, 100_000, 20_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def capital_loss_distribution():
    g = sns.displot(df, x='capital.loss', hue='income', bins=10, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital loss distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def native_country_distribution():
    def autopct(pct):  # only show the label when it's > 10%
        return ('%.2f' % pct) if pct > 10 else ''
    my_labels = df['native.country'].unique().tolist()
    ax = df['native.country'].value_counts().plot(kind='pie', figsize=(28, 12), autopct=autopct)
    ax.axes.get_yaxis().set_visible(False)
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


def handle_mis_val(df):
    print("Number of Missing values of 3 variables:")
    print(df.loc[df['native.country'].isnull()].count())
    print(df.loc[df['occupation'].isnull() & (df['workclass'].isnull())].count())
    print("Number of Missing values of workclass and native.country:")
    print(df.loc[df['occupation'].isnull() & (df['workclass'].isnull())].count())
    """ Dropping records:"""
    # df = df.drop(df[(df['occupation'].isnull() ) & (df['workclass'].isnull()) & df['native.country'].isnull()].index)
    # print('The shape of the dataset : ', df.shape)
    df = df.drop(df[(df['occupation'].isnull()) & (df['workclass'].isnull())].index)
    df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
    print('The shape of the dataset : ', df.shape)
    print("Number of Missing values of 1ariables:")
    print(df.loc[df['occupation'].isnull() | df['workclass'].isnull() | df['native.country'].isnull()].count())
    df['native.country'].fillna(df['native.country'].mode()[0], inplace=True)
    print(df.loc[df['native.country'].isnull()].count())
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


def encode_and_bind(original_dataframe, features_to_encode):
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([dummies, original_dataframe], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return(res)


def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size = 18)
        plt.xlabel('Predicted label', size = 18)
    plt.show()


def evaluate_model(y_pred, probs,train_predictions, train_probs):
    baseline = {}
    baseline['recall']=recall_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,
                    [1 for _ in range(len(y_test))])
    baseline['roc'] = 0.5
    results = {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results = {}
    train_results['recall'] = recall_score(y_train,       train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} '
              f'Baseline: {round(baseline[metric], 2)} '
              f'Test: {round(results[metric], 2)} '
              f'Train: {round(train_results[metric], 2)} ')
     # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


def XGBoost_pipe():
    checking = True
    random_grid = {}
    XGBoost_classifier = GradientBoostingClassifier()
    pipe = make_pipeline(col_trans, XGBoost_classifier)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 3) * 100} %")

    n_estimators = [int(x) for x in np.linspace(start=50, stop=450, num=5)]
    max_depth = [int(x) for x in np.linspace(10, 110, num=3)]  # Maximum number of levels in tree
    max_depth.append(None)
    min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
    bootstrap = [True]  # Method of selecting samples for training each tree
    if checking:
        random_grid = {'n_estimators': [50,100],
                       'max_depth': [None, 5]}
    else:
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
                       'bootstrap': bootstrap,
                       }
    # Create base model to tune
    XG = GradientBoostingClassifier()
    # Create random search model and fit the data
    XG_random = GridSearchCV(
        estimator=XG,
        param_grid=random_grid,
        n_jobs=-1, cv=5,
        verbose=2,
        scoring='f1')
    X_train_encoded = encode_and_bind(X_train, features_to_encode)
    X_test_encoded = encode_and_bind(X_test, features_to_encode)
    if train_mode:
        XG_random.fit(X_train_encoded, y_train)
        joblib.dump(XG_random, 'model_XG_final.pkl')  # save your model or results
    else:
        XG_random = joblib.load("model_XG_final.pkl")  # load your model for further usage
    y_pred_acc = XG_random.predict(X_test_encoded)
    cm = confusion_matrix(y_test, y_pred_acc)
    plot_confusion_matrix(cm, classes=['0 - <=50K', '1 - >50K'],
                          title='income Confusion Matrix')
    probs = XG_random.predict_proba(X_test_encoded)[:,1]
    train_probs = XG_random.predict_proba(X_train_encoded)[:,1]
    train_predictions = XG_random.predict(X_train_encoded)
    evaluate_model(y_pred_acc,probs,train_predictions,train_probs)


def rf_pipe():
    checking = True
    random_grid = {}
    rf_classifier = RandomForestClassifier()
    pipe = make_pipeline(col_trans, rf_classifier)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 3) * 100} %")

    n_estimators = [int(x) for x in np.linspace(start=50, stop=450, num=5)]
    max_depth = [int(x) for x in np.linspace(10, 110, num=3)]  # Maximum number of levels in tree
    max_depth.append(None)
    min_samples_split = [2, 5, 10]  # Minimum number of samples required to split a node
    bootstrap = [True]  # Method of selecting samples for training each tree
    if checking:
        random_grid = {'n_estimators': [50,100],
                       'max_depth': [None, 5]}
    else:
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
                       'bootstrap': bootstrap,
                       }
    # Create base model to tune
    rf = RandomForestClassifier(oob_score=True)
    # Create random search model and fit the data
    rf_random = GridSearchCV(
        estimator=rf,
        param_grid=random_grid,
        n_jobs=-1, cv=5,
        verbose=2,
        scoring='f1')
    X_train_encoded = encode_and_bind(X_train, features_to_encode)
    X_test_encoded = encode_and_bind(X_test, features_to_encode)
    if train_mode:
        rf_random.fit(X_train_encoded, y_train)
        joblib.dump(rf_random, 'model_gs_final.pkl')  # save your model or results
    else:
        rf_random = joblib.load("model_gs1.pkl")  # load your model for further usage
    y_pred_acc = rf_random.predict(X_test_encoded)
    cm = confusion_matrix(y_test, y_pred_acc)
    plot_confusion_matrix(cm, classes=['0 - <=50K', '1 - >50K'],
                          title='income Confusion Matrix')
    probs = rf_random.predict_proba(X_test_encoded)[:,1]
    train_probs = rf_random.predict_proba(X_train_encoded)[:,1]
    train_predictions = rf_random.predict(X_train_encoded)
    evaluate_model(y_pred_acc,probs,train_predictions,train_probs)



if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    df = read_data()
    df = fix_value_spaces_and_names(df)
    """ -----------------------------  """
    # TODO - to aggregation of occupation - professionals
    df = handle_mis_val(df)
    # df_1 = df.replace(to_replace=[None], value=np.nan)
    # result = cramers_v( df['occupation'], df['workclass'])
    # print(result)
    # y = df.iloc[:, 14:15]
    # X = df.iloc[:, :14]

    y = df.pop('income')
    df = df.drop('native.country', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    features_to_encode = df.columns[df.dtypes == object].tolist()
    col_trans = make_column_transformer((OneHotEncoder(), features_to_encode), remainder="passthrough")
    XGBoost_pipe()
# def rf_pipe():






