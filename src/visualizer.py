import os
import numpy as np
import pandas as pd
import matplotlib as mlt
# mlt.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from itertools import combinations
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings
filterwarnings('ignore')
from math import pi
from utils import *


class Visualizer:
  def __init__(self, df, target_col, num_cols=None, cat_cols=None, ignore_cols=None, problem_type='classification'):
    self.df           = df
    self.target_col   = target_col
    self.num_cols     = num_cols if num_cols != None else list(df.select_dtypes(np.number).columns)
    self.cat_cols     = cat_cols if cat_cols != None else list(df.select_dtypes('O').columns)
    self.ignore_cols  = ignore_cols
    self.problem_type = problem_type

    # Remove target_col from the numerical or categorical columns.
    if self.target_col in self.num_cols: self.num_cols.remove(self.target_col)
    if self.target_col in self.cat_cols: self.cat_cols.remove(self.target_col)

    # Remove the ignore columns
    if (self.num_cols is None or self.cat_cols is None) and self.ignore_cols:
      for col in ignore_cols:
        if col in self.num_cols: self.num_cols.remove(col)
        if col in self.cat_cols: self.cat_cols.remove(col)

  ############################ Count Plot (Cat)
  @staticmethod
  def create_countplot(df, col_name, figsize=(8, 6), annot=True, rotate=False, folder_name=None):
    len_unique = len(df[col_name].unique())
    fig = plt.figure(figsize=figsize)
    counts = df[col_name].value_counts().index
    if len_unique >= 12:
      color  = sns.color_palette()[0]
      ax = sns.countplot(df[col_name], edgecolor='k', zorder=3, order=counts, color=color)
    else:
      ax = sns.countplot(df[col_name], edgecolor='k', zorder=3, order=counts)
    plt.title('Distribution of "'+col_name+'" column', y=1.05, size=16)
    if annot: show_annotation(ax, 10, 12, df.shape[0])
    if rotate: plt.xticks(rotation=90)
    plt.grid(zorder=0)
    if folder_name: 
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', folder_name, f"{col_name}_count.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ############################### Pie Plot (Cat)
  @staticmethod
  def create_pieplot(df, col_name, figsize=(8, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    sorted_counts = df[col_name].value_counts()
    patches, texts = plt.pie(sorted_counts, labels=sorted_counts.index, startangle=90,
    counterclock=False, shadow=True, explode=(0.04, 0), autopct='%1.1f%%')[0:2]
    fig.tight_layout()
    plt.title('Distribution of "'+col_name+'" column', y=1.05, size=16)
    plt.axis('equal')
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', folder_name, f"{col_name}_pie.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ############################### Historgram (Num)
  @staticmethod
  def create_hist(df, col_name, figsize=(8, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    step = float(.05*(df[col_name].max()-df[col_name].min()))
    if step < 1:
      bins = range(df[col_name].min(), df[col_name].max(), int((df[col_name].max()-df[col_name].min())))
      plt.hist(x=df[col_name], bins=bins, edgecolor='k', zorder=3)
      plt.xticks(bins)
    else:
      plt.hist(x=df[col_name], edgecolor='k', zorder=3)
    plt.grid(zorder=0)
    plt.title(f'Histogram of "{col_name}"', y=1.05, size=16)
    plt.xlabel(f"{col_name}")
    plt.ylabel("count")
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', folder_name, f"{col_name}_histogram.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ############################### KDE (Num)
  @staticmethod
  def create_kde(df, col_name, figsize=(8, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    sns.kdeplot(df[col_name], shade=True)
    plt.grid()
    plt.title(f'KDE for "{col_name}"', y=1.05, size=16)
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', folder_name, f"{col_name}_kde.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ############################### Word Cloud (Cat)
  @staticmethod
  def create_wordcloud(df, col_name, figsize=(15, 12), folder_name=None):
    # Lets first convert the 'result' dictionary to 'list of tuples'
    tup = dict(df[col_name].value_counts())
    #Initializing WordCloud using frequencies of tags.
    wordcloud = WordCloud(background_color='black',
                          width=1000,
                          height=800,
                          # stopwords=set(STOPWORDS),
                        ).generate_from_frequencies(tup)

    fig = plt.figure(figsize=figsize)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(f'WordCloud for {col_name}', y=1.05, size=20)
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', folder_name, f"{col_name}_wordcloud.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ############################### Histogram Plot for high cardinality (Cat)
  @staticmethod
  def create_hist_for_high_cardinality(df, col_name, figsize=(16, 6), annot=True, folder_name=None):
    counts = df[col_name].value_counts().values
    step = int(.05*(counts.max()-counts.min()))
    fig = plt.figure(figsize=figsize)
    if step < 1:
      ax = plt.hist(x=counts, edgecolor='k', zorder=3)
    else:
      bins = range(counts.min(), counts.max(), step)
      ax = plt.hist(x=counts, bins=bins, edgecolor='k', zorder=3)
      plt.xticks(bins)
    plt.grid(zorder=0)
    plt.title(f"Range of Occurances for labels in '{col_name}'", y=1.05, size=20)
    plt.ylabel("Counts of Labels")
    plt.xlabel("Range of Occurances")

    if annot:
      sizes = []  # Get highest value in y
      for p in ax[2]:
          height = int(p.get_height())
          sizes.append(height)
          plt.text(p.get_x() + p.get_width() / 2.,          # At the center of each bar. (x-axis)
                    height+1,                            # Set the (y-axis)
                    '{}'.format(height),  # Set the text to be written
                    ha='center', fontsize=12)
      plt.ylim(0, max(sizes) * 1.15)  # set y limit based on highest heights
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', folder_name, f"{col_name}_hist_highCardinality.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

############################### line with index (Num)
  @staticmethod
  def create_line_with_index(df, col_name, target_col=None, figsize=(25, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    if target_col:
      ax  = fig.add_axes([0, 0, 1, 1])
      uniques = df[target_col].unique()
      colors = sns.color_palette(n_colors=len(uniques))
      for i, label in enumerate(uniques):
        ax.plot(df[df[target_col] == label][col_name], lw=1, c=colors[i], label=label)
      plt.legend()
    else:
      df[col_name].plot.line(figsize=figsize, lw=1)
    plt.grid()
    plt.title(f"Distribution of '{col_name}' along the index", y=1.05, size=16)
    plt.ylabel(f'{col_name} values')
    plt.xlabel("Row Index")
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), 'visualizer', "4_index", folder_name, f"{col_name}_line.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

############################### point with index (Num)
  @staticmethod
  def create_point_with_index(df, col_name, target_col=None, figsize=(28, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    if target_col:
        uniques = df[target_col].unique()
        colors = sns.color_palette(n_colors=len(uniques))
        for i, label in enumerate(uniques):
            plt.plot(df[df[target_col] == label][col_name], '.', c=colors[i], label=label)
        plt.legend()
    else:
        plt.plot(df[col_name], '.')
      
    plt.grid()
    plt.title(f"Distribution of '{col_name}' along the index", y=1.05, size=16)
    plt.xlabel('Row Index')
    plt.ylabel(f'{col_name} Values')
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", "4_index", folder_name, f"{col_name}_points.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ################################ Clustered Bar Plot (Cat - Cat)
  @staticmethod
  def create_clustered_bar_plot(df, cat_1, cat_2, annot=True, figsize=(14, 8), folder_name=None):
    fig = plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x=cat_1, hue=cat_2, edgecolor='k', zorder= 3)
    plt.grid(zorder=0)
    plt.title(f'Distribution of "{cat_1}" clustered by "{cat_2}"', y=1.05, size=16)
    if annot:
      show_annotation(ax, 2, 12, df.shape[0])
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{cat_1} VS {cat_2}_clustered_bar.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

  ################################ Bubble Plot (Cat - Cat)
  @staticmethod
  def create_bubble_plot(df, cat_1, cat_2, figsize=(12, 6), folder_name=None):
    dfu = pd.crosstab(df[cat_1], df[cat_2], normalize=True).unstack().reset_index()
    dfu.columns = [cat_1, cat_2, 'counts']
    dfu['counts'] *= 10000
    mini = int(dfu.counts.min())
    maxi = int(dfu.counts.max())
    msizes = list(range(mini, maxi, int((maxi-mini)/5)))
    markers = []
    for size in msizes:
        plt.ioff()
        temp_fig = plt.figure()
        scatter_markers = plt.scatter([], [], s=size,
                                  label=f'{size}',
                                   color='lightgreen',
                                  alpha=.6, edgecolor='k', linewidth=1.5)
        markers.append(scatter_markers)
        plt.close(temp_fig)

    fig = plt.figure(figsize=figsize)
    plt.scatter(x=cat_1, y=cat_2, s='counts', data=dfu, zorder=3, alpha=.8, edgecolor='k', color='lightgreen')
    plt.margins(.1 if len(df[cat_1].unique())*len(df[cat_2].unique()) > 20 else .4)
    plt.xlabel(cat_2, size=12); plt.ylabel(cat_1, size=12)
    plt.xticks(dfu[cat_1].unique()); plt.yticks(dfu[cat_2].unique())
    plt.title(f'"{cat_1}" Vs. "{cat_2}"', y=1.05, size=16); plt.grid(zorder=0)
    plt.legend(handles=markers, title='Counts',
              labelspacing=3, handletextpad=2,
              fontsize=14, loc=(1.10, .05))
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{cat_1} VS {cat_2}_bubble.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()


  ################################ Scatter Plot (Num - Num)
  @staticmethod
  def create_scatter(df, num_1, num_2, target_col=None, figsize=(12, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    if target_col:
      sns.scatterplot(x=num_1, y=num_2, data=df, zorder=3, hue=target_col)
    else:
      sns.scatterplot(x=num_1, y=num_2, data=df, zorder=3)

    plt.grid(zorder=0)
    plt.title(f"Scatter Plot of '{num_1}' and '{num_2}'", y=1.05, size=16)

    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{num_1} VS {num_2}_scatter.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

################################ Density Plot (Num - Num)
  @staticmethod
  def create_density(df, num_1, num_2, figsize=(8, 6), folder_name=None):
    mlt.rcParams['figure.figsize'] = figsize
    g = sns.jointplot(x=df[num_1], y=df[num_2], kind='kde', cmap="Blues", shade=True, shade_lowest=True)
    # sns.plt.title(f"Density Plot of '{num_1}' and '{num_2}'", y=3, size=16)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Density Plot of '{num_1}' and '{num_2}'", fontsize=16)
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{num_1} VS {num_2}_density.png"), bbox_inches='tight')
      plt.close(g.fig)
    else:
      plt.show()


################################ Box Plot (Num - Cat)
  @staticmethod
  def create_box_plot(df, cat_col, num_col, figsize=(8, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    sns.boxplot(x=cat_col, y=num_col, data=df, zorder=3)
    plt.title(f'Boxplot for "{cat_col}" and "{num_col}"', y=1.05, size=16)
    plt.grid(zorder=0)

    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{cat_col}_and_{num_col}_boxplot.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()


################################ Violin Plot (Num - Cat)
  @staticmethod
  def create_violin_plot(df, cat_col, num_col, figsize=(8, 6), folder_name=None):
    fig = plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=cat_col, y=num_col, inner='quartile', zorder=10)
    plt.xticks(rotation=15)
    plt.title(f"{cat_col} Vs. {num_col}", y=1.05, size=20)
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{cat_col}_and_{num_col}_violinplot.png"), bbox_inches='tight')
      plt.close(fig)
    else:
      plt.show()

##################################### Ridge Plot (Num - Cat)
  @staticmethod
  def create_ridge_plot(df, cat_col, num_col, folder_name=None):
    group_means = df.groupby([cat_col]).mean()
    group_order = group_means.sort_values([num_col], ascending = False).index

    g = sns.FacetGrid(data=df, row=cat_col, size=1, aspect=7, row_order=group_order)
    g.map(sns.kdeplot, num_col, shade = True)
    g.set_titles('{row_name}')
    g.set(yticks=[])
    g.axes[0,0].set_ylabel(cat_col)
    # g.fig.subplots_adjust(top=1.4)
    # g.fig.suptitle(f"Ridge Plot for '{cat_col} with '{num_col}'", fontsize=16)
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{cat_col}_and_{num_col}_ridgePlot.png"), bbox_inches='tight')
      plt.close(g.fig)
    else:
      plt.show()

######################################## Parallel Plot (Multi-Num with Cat)
  @staticmethod
  def create_parallel_plot(df, num_cols, target_col=None, figsize=(20, 10), folder_name=None):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[num_cols])
    df_scaled = pd.DataFrame(df_scaled, columns=num_cols)
    
    # Make the plot
    if target_col:
      df_scaled[target_col] = df[target_col]
    else:
      df_scaled['regression'] = 0

    for i in [10, 50, 100, 500, 1000]:
      fig = plt.figure(figsize=figsize) 
      parallel_coordinates(df_scaled.sample(i),
                           target_col if target_col else 'regression',
                           lw=2, 
                           colormap=plt.get_cmap("winter"))
      plt.title(f"{i} samples with target", y=1.05, size=20)

      if folder_name:
        plt.ioff()
        plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{i}_parallel_plot.png"), bbox_inches='tight')
        plt.close(fig)
      else:
        plt.show()


  ############################################ Radar Plot
  @staticmethod
  def create_radar_plot(df, num_cols, cat_col, figsize=(20, 20), folder_name=None):
    scaler             = MinMaxScaler()
    df_scaled          = scaler.fit_transform(df[num_cols])
    df_scaled          = pd.DataFrame(df_scaled, columns=num_cols)
    df_scaled[cat_col] = df[cat_col]
    df_scaled          = df_scaled.loc[:,~df_scaled.columns.duplicated()]

    grouped_df = df_scaled.groupby(cat_col)[num_cols].median()
    categories = grouped_df[num_cols].columns
    N          = len(categories)

    for i, label in enumerate(df[cat_col].unique()):
      values  = grouped_df.loc[label].values.flatten().tolist()
      values += values[:1]

      angles  = [n / float(N) * 2 * pi for n in range(N)]
      angles += angles[:1]

      mlt.rcParams['figure.figsize'] = figsize
      ax = plt.subplot((len(df[cat_col].unique())+1)//3, 3, i+1, polar=True)
      plt.xticks(angles[:-1], categories, color='grey', size=13)
      ax.set_rlabel_position(0)
      plt.yticks([.25, .5, .75], [".25", ".5", ".75"], color="grey", size=13)
      plt.ylim(0, 1)
      ax.plot(angles, values, linewidth=1, linestyle='solid')
      ax.fill(angles, values, 'b', alpha=0.1)
      plt.title(f"'{cat_col} = {label}' with num cols")

    plt.tight_layout()
    if folder_name:
      plt.ioff()
      plt.savefig(os.path.join(os.getcwd(), "visualizer", folder_name, f"{cat_col}_radar_plot.png"), bbox_inches='tight')
      plt.close()
    else:
      plt.show()

  #####################################
  #        Uni-variate Target         #
  #####################################
  def visualize_target(self):
    create_folder(folder_name=os.path.join('visualizer', '1_target'), verbose=False)

    if self.problem_type.startswith('clas'):
      self.create_countplot(df=self.df, col_name=self.target_col, folder_name='1_target')
      self.create_pieplot(df=self.df, col_name=self.target_col, folder_name='1_target')
    else:
      self.create_hist(df=self.df, col_name=self.target_col, folder_name='1_target')
      self.create_kde(df=self.df, col_name=self.target_col, folder_name='1_target')

  #####################################
  #      Uni-variate Categorical      #
  #####################################
  def visualize_cat(self):
    create_folder(folder_name=os.path.join('visualizer', '2_cat_features'), verbose=False)

    for i, col in enumerate(self.cat_cols):
      print(f'\r{bg("Uni-variate Cat", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.cat_cols)}', end='') 
      unique_len = len(self.df[col].unique())
      if unique_len <= 27:
        self.create_countplot(df=self.df, col_name=col, folder_name='2_cat_features')
      else:
        self.create_wordcloud(df=self.df, col_name=col, folder_name='2_cat_features')
        self.create_hist_for_high_cardinality(df=self.df, col_name=col, folder_name='2_cat_features')
    print()

  #####################################
  #      uni-variate numerical        #
  #####################################
  def visualize_num(self):
    create_folder(folder_name=os.path.join('visualizer', "3_num_features", "3.1_histogram"), verbose=False)
    create_folder(folder_name=os.path.join('visualizer', "3_num_features", "3.2_kde"), verbose=False)

    for i, col in enumerate(self.num_cols):
      print(f'\r{bg("Uni-variate Num", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
      # TODO: look for an equation on how to set the bins properly.
      self.create_hist(df=self.df, col_name=col, folder_name=os.path.join("3_num_features", "3.1_histogram"))
      self.create_kde(df=self.df, col_name=col, folder_name=os.path.join("3_num_features", "3.2_kde"))
    print()

  ########################################
  #    Bi-variate numerical with index   #
  ########################################
  def visualize_num_with_idx(self):
    create_folder(folder_name=os.path.join('visualizer', "4_index", "1_num_features", "1_line"), verbose=False)
    create_folder(folder_name=os.path.join('visualizer', "4_index", "1_num_features", "2_points"), verbose=False)

    if self.problem_type.startswith('clas'):
      for i, col in enumerate(self.num_cols):
        print(f'\r{bg("Bi-variate Num with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
        self.create_line_with_index(df=self.df, col_name=col, target_col=self.target_col, folder_name=os.path.join("1_num_features", "1_line"))
        self.create_point_with_index(df=self.df, col_name=col, target_col=self.target_col, folder_name=os.path.join("1_num_features", "2_points"))
      print()
    else:
      for i, col in enumerate(self.num_cols):
        print(f'\r{bg("Bi-variate Num with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
        self.create_line_with_index(df=self.df, col_name=col, folder_name=os.path.join("1_num_features", "1_line"))
        self.create_point_with_index(df=self.df, col_name=col, folder_name=os.path.join("1_num_features", "2_points"))
      print()

  ########################################
  #    Bi-variate categorical with index #
  ########################################
  def visualize_cat_with_idx(self):
    create_folder(folder_name=os.path.join('visualizer', "4_index", "2_cat_features"), verbose=False)

    if self.problem_type.startswith('class'):
      for i, col in enumerate(self.cat_cols):
        print(f'\r{bg("Bi-variate Cat with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.cat_cols)}', end='')
        self.create_point_with_index(df=self.df, col_name=col, target_col=self.target_col, folder_name='2_cat_features')
      print()
    else:
      for i, col in enumerate(self.cat_cols):
        print(f'\r{bg("Bi-variate Num with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
        self.create_point_with_index(df=self.df, col_name=col, target_col=self.target_col, folder_name='2_cat_features')
      print()

  ########################################
  #       Bi-variate cat with cat        #
  ########################################
  def visualize_cat_with_cat(self):
    create_folder(folder_name=os.path.join('visualizer', '5_cat_with_cat'), verbose=False)

    comb_len = ncr(len(self.cat_cols), 2)
    for i, (cat_1, cat_2) in enumerate(combinations(self.cat_cols, 2)):
      print(f'\r{bg("Bi-variate Cat with Cat", "s", "green")}: finished {bg(i+1, color="yellow")} out of {comb_len}', end='') 
      if cat_1 == cat_2: continue
      uniques_len_1 = len(self.df[cat_1].unique())
      uniques_len_2 = len(self.df[cat_2].unique())
      if uniques_len_1 * uniques_len_2 <= 30:
        self.create_clustered_bar_plot(df=self.df, cat_1=cat_1, cat_2=cat_2, folder_name="5_cat_with_cat")
        self.create_bubble_plot(df=self.df, cat_1=cat_1, cat_2=cat_2, folder_name="5_cat_with_cat")
    print()

  ########################################
  #       Bi-variate num with num        #
  ########################################
  def visualize_num_with_num(self):
    create_folder(folder_name=os.path.join('visualizer', '6_num_with_num'), verbose=False)

    comb_len = ncr(len(self.num_cols), 2)
    if self.problem_type.startswith("class"):
      for i, (num_1, num_2) in enumerate(combinations(self.num_cols, 2)):
        print(f'\r{bg("Bi-variate Num with Num", "s", "green")}: finished {bg(i+1, color="yellow")} out of {comb_len}', end='')
        self.create_scatter(df=self.df, num_1=num_1, num_2=num_2, target_col=self.target_col, folder_name="6_num_with_num")
        self.create_density(df=self.df, num_1=num_1, num_2=num_2, folder_name="6_num_with_num")
      print()
    else:
      for i, (num_1, num_2) in enumerate(combinations(self.num_cols, 2)):
        print(f'\r{bg("Bi-variate Num with Num", "s", "green")}: finished {bg(i+1, color="yellow")} out of {comb_len}', end='')
        self.create_scatter(df=self.df, num_1=num_1, num_2=num_2, folder_name="6_num_with_num")
        self.create_density(df=self.df, num_1=num_1, num_2=num_2, folder_name="6_num_with_num")
      print()     

  ########################################
  #       Bi-variate Num with Cat        #
  ########################################
  def visualize_num_with_cat(self):
    create_folder(folder_name=os.path.join('visualizer', "7_num_with_cat", "1_box_plot"), verbose=False)
    create_folder(folder_name=os.path.join('visualizer', "7_num_with_cat", "2_violin_plot"), verbose=False)
    create_folder(folder_name=os.path.join('visualizer', "7_num_with_cat", "3_ridge_plot"), verbose=False)

    n_plots, i = len(self.cat_cols) * len(self.num_cols), 1
    for cat_col in self.cat_cols:
      for num_col in self.num_cols:
        if len(self.df[cat_col].unique()) <= 27:
          print(f'\r{bg("Bi-variate Num Vs Cat", "s", "green")}: finished {bg(i, color="yellow")} out of {n_plots}', end='')
          self.create_box_plot(df=self.df, cat_col=cat_col, num_col=num_col, folder_name=os.path.join("7_num_with_cat", "1_box_plot"))
          self.create_violin_plot(df=self.df, cat_col=cat_col, num_col=num_col, folder_name=os.path.join("7_num_with_cat", "2_violin_plot"))
          self.create_ridge_plot(df=self.df, cat_col=cat_col, num_col=num_col, folder_name=os.path.join("7_num_with_cat", "3_ridge_plot"))
        i += 1

  ########################################
  #             Multi-variate            #
  ########################################
  def visualize_multi_variate(self):
    create_folder(folder_name=os.path.join('visualizer', "8_multi_variate", "1_parallel_plot"), verbose=False)
    create_folder(folder_name=os.path.join('visualizer', "8_multi_variate", "2_radar_plot"), verbose=False)

    self.create_parallel_plot(df=self.df,
                              num_cols=self.num_cols,
                              target_col=self.target_col if self.problem_type.startswith("class") else None,
                              folder_name=os.path.join("8_multi_variate", "1_parallel_plot"))

    n_cat_cols = len(self.cat_cols)
    for i, cat_col in enumerate(self.cat_cols):
      print(f'\r{bg("Multi-variate Nums with Cat", "s", "green")}: finished {bg(i+1, color="yellow")} out of {n_cat_cols}', end='')
      if len(self.df[cat_col].unique()) <= 30:
        self.create_radar_plot(df=self.df, num_cols=self.num_cols, cat_col=cat_col, folder_name=os.path.join("8_multi_variate", "2_radar_plot"))


  # def visualize_cat_with_target(self):
  #     pass

  # def visualize_num_with_target(self):
  #     pass

  def visualize_all(self):
    self.visualize_target()
    self.visualize_cat()
    self.visualize_num()
    self.visualize_num_with_idx()
    self.visualize_cat_with_idx()
    self.visualize_cat_with_cat()
    self.visualize_num_with_num()
    self.visualize_num_with_cat()
    self.visualize_multi_variate()
