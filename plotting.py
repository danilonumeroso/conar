from matplotlib import pyplot as plt
import seaborn as sns
def plot_histogram_boxplot(samples, name=None):
    sns.displot(samples, bins=10).set(title=name)
    plt.show()
    sns.boxplot(samples).set(title=name)
    plt.show()
