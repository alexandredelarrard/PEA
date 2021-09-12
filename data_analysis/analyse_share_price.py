import seaborn as sns 
sns.set_style(style="dark")


def correlation_analysis(full):

    all_op = [x for x in full.columns if "OPEN" in x ]
    sns.heatmap(full[all_op].corr())