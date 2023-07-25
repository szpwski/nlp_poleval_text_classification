import nltk
import matplotlib.pyplot as plt
import pandas as pd

def update_seaborn_plot_labels_title(plot, xlabel=None, ylabel=None, title=None, legend=None):
    """
    Function to update x-axis label, y-axis label, and title of a Seaborn plot.
    
    Args:
        plot - The Seaborn plot you want to update.
        xlabel - The new label for the x-axis. If not provided, the label remains unchanged.
        ylabel - The new label for the y-axis. If not provided, the label remains unchanged.
        title - The new title for the plot. If not provided, the title remains unchanged.
        legend - The new title for the legend. If note provided, the title remains unchanged. 
    """
    if xlabel is not None:
        plot.set(xlabel=xlabel)
    if ylabel is not None:
        plot.set(ylabel=ylabel)
    if title is not None:
        plot.set(title = title)
    if legend is not None:
        plot._legend.set_title(legend)

def plot_ngram(non_harmful : list, harmful : list, n : int = 1, top : int = 10, filter_words : list = [], lang : str = 'pl'):
    """
    Function plots n-gram of given input list

    Args:
        non_harmful - input list with tokens
        harmful - input list with tokens of harmful tweets
        n - integer specyfing n-gram model
        top - specyfing maximum number of most frequent words displayed
        filter_words - list of common words to filter out
        lang - language of plot
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout = True)
    # Retrieve n-grams
    if lang == 'pl':
        label_0 = 'klasa nieszkodliwe'
        label_1 = 'klasa szkodliwe'
        xaxis = 'Słowo'
        yaxis = 'Częstotliwość'
    else:
        label_0 = 'harmless class'
        label_1 = 'harmful class'
        xaxis = 'Word'
        yaxis = 'Frequency'

    for flat, desc, i in zip([non_harmful, harmful], [label_0, label_1], [0, 1]):
        dic_words_freq = nltk.FreqDist(nltk.ngrams(flat, n))

        # Create pandas dataframe
        dtf_bi = pd.DataFrame(dic_words_freq.most_common(), 
                            columns=[xaxis, yaxis])
        dtf_bi[xaxis] = dtf_bi[xaxis].apply(lambda x: " ".join(
                        string for string in x) )
        
        dtf_bi = dtf_bi[[x not in filter_words for x in dtf_bi[xaxis]]]

        # Plot unigram
        dtf_bi.set_index(xaxis).iloc[:top,:].sort_values(by=yaxis).plot(
                        kind="barh",
                        legend=False, ax=axes[i], subplots=True)
        
        axes[i].set_title(str(n) + "-gram: " + desc)
    plt.show()