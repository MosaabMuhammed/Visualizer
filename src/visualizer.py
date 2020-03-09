import os
import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from itertools import combinations
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from math import pi
import warnings
warnings.filterwarnings('ignore')
from utils import show_annotation, ncr, bg, colored


class Visualizer:
  def __init__(self, path, df, target_col, num_cols=None, cat_cols=None, problem_type='classification'):
    self.path         = path
    self.df           = df
    self.target_col   = target_col
    self.num_cols     = num_cols if num_cols != None else list(df.select_dtypes(np.number).columns)
    self.cat_cols     = cat_cols if cat_cols != None else list(df.select_dtypes('O').columns)
    self.problem_type = problem_type

    # Remove target_col from the numerical or categorical columns.
    if self.target_col in self.num_cols: self.num_cols.remove(self.target_col)
    if self.target_col in self.cat_cols: self.cat_cols.remove(self.target_col)

    # Create the parent folder.
    self.create_folder('visualizer')
    self.path += '/visualizer'

  def create_folder(self, folder_name, verbose=True):
    # Create visulizer Directory if don't exist
    if not os.path.exists(self.path+"/"+folder_name):
      os.makedirs(self.path+"/"+folder_name)
      if verbose: print("Directory " , folder_name ,  " Created ")
    # else:    
      if verbose: print("Directory " , folder_name ,  " already exists")

  ############################ Count Plot
  def __create_countplot(self, col_name, folder_name, figsize=(6, 4), annot=True):
    len_unique = len(self.df[col_name].unique())
    fig = plt.figure(figsize=(6, 4) if len_unique < 8 else (20, 8))
    counts = self.df[col_name].value_counts().index
    if len_unique >= 12:
      color  = sns.color_palette()[0]
      ax = sns.countplot(self.df[col_name], edgecolor='k', zorder=3, order=counts, color=color)
    else:
      ax = sns.countplot(self.df[col_name], edgecolor='k', zorder=3, order=counts)
    plt.title('Distribution of "'+col_name+'" column', y=1.05, size=16)
    if annot: show_annotation(ax, 10, 12, self.df.shape[0])
    plt.grid(zorder=0)
    plt.savefig(self.path+'/'+folder_name+'/'+col_name+'_count.png', bbox_inches='tight')

  ############################### Pie Plot
  def __create_pieplot(self, folder_name):
    fig = plt.figure(figsize=(6, 5))
    sorted_counts = self.df[self.target_col].value_counts()
    patches, texts = plt.pie(sorted_counts, labels=sorted_counts.index, startangle=90,
    counterclock=False, shadow=True, explode=(0.04, 0), autopct='%1.1f%%')[0:2]
    fig.tight_layout()
    plt.title('Distribution of "'+self.target_col+'" column', y=1.05, size=16)
    plt.axis('equal')
    fig.savefig(self.path+'/'+folder_name+'/target_pie.png', bbox_inches='tight')

  ############################### Historgram
  def __create_hist(self, col_name, folder_name):
    fig = plt.figure()
    step = float(.05*(self.df[col_name].max()-self.df[col_name].min()))
    if step < 1:
      bins = range(self.df[col_name].min(), self.df[col_name].max(), int((self.df[col_name].max()-self.df[col_name].min())))
      plt.hist(x=self.df[col_name], bins=bins, edgecolor='k', zorder=3)
      plt.xticks(bins)
    else:
      plt.hist(x=self.df[col_name], edgecolor='k', zorder=3)
    plt.grid(zorder=0)
    plt.title(f'Histogram of "{col_name}"', y=1.05, size=16)
    plt.xlabel(f"{col_name}")
    plt.ylabel("count")
    fig.savefig(f"{self.path}/{folder_name}/{col_name}_histogram.png", bbox_inches='tight')

  ############################### KDE
  def __create_kde(self, col_name, folder_name):
      fig = plt.figure()
      sns.kdeplot(self.df[col_name], shade=True)
      plt.grid()
      plt.title(f'KDE for "{col_name}"', y=1.05, size=16)
      fig.savefig(f"{self.path}/{folder_name}/{col_name}_kde.png", bbox_inches='tight')

  ############################### Word Cloud
  def __create_wordcloud(self, col_name, folder_name):
    # Lets first convert the 'result' dictionary to 'list of tuples'
    tup = dict(self.df[col_name].value_counts())
    #Initializing WordCloud using frequencies of tags.
    wordcloud = WordCloud(background_color='black',
                          width=1000,
                          height=800,
                          # stopwords=set(STOPWORDS),
                        ).generate_from_frequencies(tup)

    fig = plt.figure(figsize=(20,18))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(f'WordCloud for {col_name}', y=1.05, size=20)
    fig.savefig(f"{self.path}/{folder_name}/{col_name}_wordcloud.png", bbox_inches='tight')

  ############################### Histogram Plot for high cardinality categorical features.
  def __create_hist_for_high_cardinality(self, col_name, folder_name):
    counts = self.df[col_name].value_counts().values
    step = int(.05*(counts.max()-counts.min()))
    fig = plt.figure(figsize=(16, 6))
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

    sizes = []  # Get highest value in y
    for p in ax[2]:
        height = int(p.get_height())
        sizes.append(height)

        plt.text(p.get_x() + p.get_width() / 2.,          # At the center of each bar. (x-axis)
                  height+0.5,                            # Set the (y-axis)
                  '{}'.format(height),  # Set the text to be written
                  ha='center', fontsize=12)
    plt.ylim(0, max(sizes) * 1.15)  # set y limit based on highest heights
    fig.savefig(f"{self.path}/{folder_name}/{col_name}_hist_highCardinality.png", bbox_inches='tight')

############################### line with index
  def __create_line_with_index(self, col_name, folder_name, target=False):
    fig = plt.figure(figsize=(25, 6))
    ax  = fig.add_axes([0, 0, 1, 1])
    if target:
      uniques = self.df[self.target_col].unique()
      colors = sns.color_palette(n_colors=len(uniques))
      for i, label in enumerate(uniques):
        ax.plot(self.df[self.df[self.target_col] == label][col_name], lw=1, c=colors[i], label=label)
      plt.legend()
    else: df.plot.line(y=col_name ,figsize=(20,6), lw=1)
    plt.grid() 
    plt.title(f"Distribution of '{col_name}' along the index", y=1.05, size=16)
    plt.ylabel(f'{col_name} values')
    plt.xlabel("Row Index")
    plt.savefig(f"{self.path}/4_index/{folder_name}/{col_name}_line.png", bbox_inches='tight')

############################### point with index
  def __create_point_with_index(self, col_name, folder_name, target=False):
    plt.figure(figsize=(28, 6))
    if target:
        uniques = self.df[self.target_col].unique()
        colors = sns.color_palette(n_colors=len(uniques))
        for i, label in enumerate(uniques):
            plt.plot(self.df[self.df[self.target_col] == label][col_name], '.', c=colors[i], label=label)
        plt.legend()
    else:
        plt.plot(df[col_name], '.')
      
    plt.grid()
    plt.title(f"Distribution of '{col_name}' along the index", y=1.05, size=16)
    plt.xlabel('Row Index')
    plt.ylabel(f'{col_name} Values')
    plt.savefig(f"{self.path}/4_index/{folder_name}/{col_name}_points.png", bbox_inches='tight')

  ################################ Clustered Bar Plot (Cat - Cat)
  def create_clustered_bar_plot(self, cat_1, cat_2):
    fig = plt.figure(figsize=(14, 8))
    ax = sns.countplot(data=self.df, x=cat_1, hue=cat_2, edgecolor='k', zorder= 3)
    plt.grid(zorder=0)
    plt.title(f'Distribution of "{cat_1}" clustered by "{cat_2}"', y=1.05, size=16)
    show_annotation(ax, 50, 12, self.df.shape[0])
    fig.savefig(f"{self.path}/5_cat_with_cat/{cat_1} VS {cat_2}_clustered_bar.png", bbox_inches='tight')

  ################################ Bubble Plot (Cat - Cat)
  def create_bubble_plot(self, cat_1, cat_2):
    dfu = pd.crosstab(self.df[cat_1], df[cat_2], normalize=True).unstack().reset_index()
    dfu.columns = [cat_1, cat_2, 'counts']
    dfu['counts'] *= 10000
    mini = int(dfu.counts.min())
    maxi = int(dfu.counts.max())
    msizes = list(range(mini, maxi, int((maxi-mini)/5)))
    markers = []
    for size in msizes:
        markers.append(plt.scatter([], [], s=size,
                                  label=f'{size}',
                                   color='lightgreen',
                                  alpha=.6, edgecolor='k', linewidth=1.5))
    fig = plt.figure(figsize=(12, 6))
    plt.scatter(x=cat_1, y=cat_2, s='counts', data=dfu, zorder=3, alpha=.8, edgecolor='k', color='lightgreen')
    plt.margins(.1 if len(self.df[cat_1].unique())*len(self.df[cat_2].unique()) > 20 else .4)
    plt.xlabel(cat_2, size=12); plt.ylabel(cat_1, size=12)
    plt.xticks(dfu[cat_1].unique()); plt.yticks(dfu[cat_2].unique())
    plt.title(f'"{cat_1}" Vs. "{cat_2}"', y=1.05, size=16); plt.grid(zorder=0)
    plt.legend(handles=markers, title='Counts',
              labelspacing=3, handletextpad=2,
              fontsize=14, loc=(1.10, .05))
    fig.savefig(f"{self.path}/5_cat_with_cat/{cat_1} VS {cat_2}_bubble.png", bbox_inches='tight')  

  ################################ Scatter Plot (Num - Num)
  def create_scatter(self, num_1, num_2):
    fig = plt.figure(figsize=(12, 6))
    sns.scatterplot(x=num_1, y=num_2, data=self.df, zorder=3, hue=self.target_col)
    plt.grid(zorder=0)
    plt.title(f"Scatter Plot of '{num_1}' and '{num_2}'", y=1.05, size=16)
    fig.savefig(f"{self.path}/6_num_with_num/{num_1} VS {num_2}_scatter.png", bbox_inches='tight')  

################################ Density Plot (Num - Num)
  def create_density(self, num_1, num_2):
    fig = plt.figure(figsize=(8, 6))
    sns.jointplot(x=self.df[num_1], y=self.df[num_2], kind='kde', cmap="Blues", shade=True, shade_lowest=True)
    fig.savefig(f"{self.path}/6_num_with_num/{num_1} VS {num_2}_density.png", bbox_inches='tight')

################################ Box Plot (Num - Cat)
  def create_box_plot(self, cat_col, num_col, folder_name):
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(x=cat_col, y=num_col, data=self.df, zorder=3)
    plt.title(f'Boxplot for "{cat_col}" and "{num_col}"', y=1.05, size=16); plt.grid(zorder=0)
    fig.savefig(f"{self.path}/{folder_name}/{cat_col}_and_{num_col}_boxplot.png", bbox_inches='tight')

################################ Violin Plot (Num - Cat)
  def create_violin_plot(self, cat_col, num_col, folder_name):
    fig = plt.figure(figsize=(8, 6))
    sns.violinplot(data=self.df, x=cat_col, y=num_col, inner='quartile', zorder=10)
    plt.xticks(rotation=15)
    plt.title(f"{cat_col} Vs. {num_col}", y=1.05, size=20)
    fig.savefig(f"{self.path}/{folder_name}/{cat_col}_and_{num_col}_violinplot.png", bbox_inches='tight')

##################################### Ridge Plot (Num - Cat)
  def create_ridge_plot(self, cat_col, num_col, folder_name):
    group_means = df.groupby([cat_col]).mean()
    group_order = group_means.sort_values([num_col], ascending = False).index

    g = sns.FacetGrid(data = df, row = cat_col, size = 1, aspect = 7,
                     row_order = group_order)
    g.map(sns.kdeplot, num_col, shade = True)
    g.set_titles('{row_name}')
    g.set(yticks=[])
    plt.savefig(f"{self.path}/{folder_name}/{cat_col}_and_{num_col}_ridgePlot.png", bbox_inches='tight')

######################################## Parallel Plot (Multi-Num with Cat)
  def create_parallel_plot(self):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(self.df[self.num_cols])
    df_scaled = pd.DataFrame(df_scaled, columns=self.num_cols)
    
    # Make the plot
    if self.problem_type == "classification":
        df_scaled[self.target_col] = self.df[self.target_col]
        for i in [10, 50, 100, 500, 1000]:
          fig = plt.figure(figsize=(20, 10)) 
          parallel_coordinates(df_scaled.sample(i), self.target_col, lw=2, colormap=plt.get_cmap("winter"))
          fig.savefig(f"{self.path}/8_multi_variate/1_parallel_plot/{i}_parallel_plot.png", bbox_inches='tight')
    else:
        df_scaled['regression'] = 0
        for i in [10, 50, 100, 500, 1000]:
          fig = plt.figure(figsize=(20, 10))
          parallel_coordinates(df_scaled.sample(i), 'regression', lw=2, colormap=plt.get_cmap("winter"))
          fig.savefig(f"{self.path}/8_multi_variate/1_parallel_plot/{i}_parallel_plot.png", bbox_inches='tight')


  ############################################ Radar Plot
  def create_radar_plot(self):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(self.df[self.num_cols])
    df_scaled = pd.DataFrame(df_scaled, columns=self.num_cols)
    df_scaled[self.cat_cols] = self.df[self.cat_cols]

    n_cat_cols = len(self.cat_cols)
    for i, cat_col in enumerate(self.cat_cols):
      if len(self.df[cat_col].unique()) <= 27:
        print(f'\r{bg("Multi-variate cat col label", "s", "green")}: finished {bg(i+1, color="yellow")} out of {n_cat_cols}', end='')
        self.create_folder(f"8_multi_variate/2_radar_plot/{cat_col}", verbose=False)
        grouped_df = df_scaled.groupby(cat_col)[self.num_cols].mean()
        categories = grouped_df[self.num_cols].columns
        N          = len(categories)

        for label in list(self.df[cat_col].unique()):
          values  = grouped_df.loc[label].values.flatten().tolist()
          values += values[:1]

          angles  = [n / float(N) * 2 * pi for n in range(N)]
          angles += angles[:1]

          mlt.rc('figure', figsize=(8, 8))
          ax = plt.subplot(111, polar=True)
          plt.xticks(angles[:-1], categories, color='grey', size=13)
          ax.set_rlabel_position(0)
          plt.yticks([.25, .5, .75], [".25", ".5", ".75"], color="grey", size=13)
          plt.ylim(0, 1)
          ax.plot(angles, values, linewidth=1, linestyle='solid')
          ax.fill(angles, values, 'b', alpha=0.1)
          plt.title(f"'{cat_col} = {label}' with num cols")
          plt.savefig(f"{self.path}/8_multi_variate/2_radar_plot/{cat_col}/{label}_radar_plot.png", bbox_inches='tight')
          plt.clf()

  #####################################
  #        Uni-variate Target         #
  #####################################
  def visualize_target(self):
    self.create_folder('1_target', verbose=False)      # Create a new folder for target visualization.

    if self.problem_type == 'classification':
      self.__create_countplot(col_name=self.target_col, folder_name='1_target') # 1. Count plot: see the distribution of the data.
      self.__create_pieplot(folder_name='1_target')                             # 2. Pie Plot: see the distribution from a different way.
    else:
      # apply the numerical plotting here.
      pass

  #####################################
  #      Uni-variate Categorical      #
  #####################################
  def visualize_cat(self):
    self.create_folder('2_cat_features', verbose=False)

    for i, col in enumerate(self.cat_cols):
      print(f'\r{bg("Uni-variate Cat", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.cat_cols)}', end='') 
      unique_len = len(self.df[col].unique())
      if unique_len <= 27:
        self.__create_countplot(col_name=col, folder_name='2_cat_features')
      else:
        self.__create_wordcloud(col_name=col, folder_name='2_cat_features')
        self.__create_hist_for_high_cardinality(col_name=col, folder_name='2_cat_features')
    print()

    # TODO: How to handle high cardinality features.
    # TODO: Add the Pie plot.


  #####################################
  #      uni-variate numerical        #
  #####################################
  def visualize_num(self):
    self.create_folder('3_num_features/3.1_histogram', verbose=False)
    self.create_folder('3_num_features/3.2_kde', verbose=False)

    for i, col in enumerate(self.num_cols):
      print(f'\r{bg("Uni-variate Num", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
      # TODO: look for an equation on how to set the bins properly.
      self.__create_hist(col_name=col, folder_name='3_num_features/3.1_histogram')
      self.__create_kde(col_name=col, folder_name='3_num_features/3.2_kde')
    print()

  ########################################
  #    Bi-variate numerical with index   #
  ########################################
  def visualize_num_with_idx(self):
    self.create_folder('4_index/1_num_features/1_line', verbose=False)
    self.create_folder('4_index/1_num_features/2_points', verbose=False)

    if self.problem_type == 'classification':
      for i, col in enumerate(self.num_cols):
        print(f'\r{bg("Bi-variate Num with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
        self.__create_line_with_index(col_name=col, folder_name='1_num_features/1_line', target=True)
        self.__create_point_with_index(col_name=col, folder_name='1_num_features/2_points', target=True)
      print()
    else:
      for i, col in enumerate(self.num_cols):
        print(f'\r{bg("Bi-variate Num with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
        self.__create_line_with_index(col_name=col, folder_name='1_num_features/1_line', target=False)
        self.__create_point_with_index(col_name=col, folder_name='1_num_features/2_points', target=False)
      print()

  ########################################
  #    Bi-variate categorical with index #
  ########################################
  def visualize_cat_with_idx(self):
    # self.create_folder('4_index')
    self.create_folder('4_index/2_cat_features', verbose=False)

    if self.problem_type == 'classification':
      for i, col in enumerate(self.cat_cols):
        print(f'\r{bg("Bi-variate Cat with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.cat_cols)}', end='')
        self.__create_point_with_index(col_name=col, folder_name='2_cat_features', target=True)
      print()
    else:
      for i, col in enumerate(self.cat_cols):
        print(f'\r{bg("Bi-variate Num with Index", type="s", color="green")}: finished {bg(i+1, color="yellow")} out of {len(self.num_cols)}', end='')
        self.__create_point_with_index(col_name=col, folder_name='2_cat_features', target=False)
      print()

  ########################################
  #       Bi-variate cat with cat        #
  ########################################
  def visualize_cat_with_cat(self):
    self.create_folder('5_cat_with_cat', verbose=False)

    comb_len = ncr(len(self.cat_cols), 2)
    for i, (col_1, col_2) in enumerate(combinations(self.cat_cols, 2)):
      print(f'\r{bg("Bi-variate Cat with Cat", "s", "green")}: finished {bg(i+1, color="yellow")} out of {comb_len}', end='') 
      if col_1 == col_2: continue
      uniques_len_1 = len(self.df[col_1].unique())
      uniques_len_2 = len(self.df[col_2].unique())
      if uniques_len_1 * uniques_len_2 <= 30:
        self.create_clustered_bar_plot(col_1, col_2)
        self.create_bubble_plot(col_1, col_2)
    print()

  ########################################
  #       Bi-variate num with num        #
  ########################################
  def visualize_num_with_num(self):
    self.create_folder('6_num_with_num', verbose=False)

    comb_len = ncr(len(self.num_cols), 2)
    for i, (num_1, num_2) in enumerate(combinations(self.num_cols, 2)):
      print(f'\r{bg("Bi-variate Num with Num", "s", "green")}: finished {bg(i+1, color="yellow")} out of {comb_len}', end='') 
      if num_1 == num_2: continue
      self.create_scatter(num_1, num_2)
    print()

  ########################################
  #       Bi-variate Num with Cat        #
  ########################################
  def visualize_num_with_cat(self):
    self.create_folder('7_num_with_cat/1_box_plot', verbose=False)
    self.create_folder('7_num_with_cat/2_violin_plot', verbose=False)
    self.create_folder('7_num_with_cat/3_ridge_plot', verbose=False)

    n_plots, i = len(self.cat_cols) * len(self.num_cols), 1
    for cat_col in self.cat_cols:
      for num_col in self.num_cols:
        if len(self.df[cat_col].unique()) <= 27:
          print(f'\r{bg("Bi-variate Num Vs Cat", "s", "green")}: finished {bg(i, color="yellow")} out of {n_plots}', end='')
          self.create_box_plot(cat_col, num_col, folder_name="7_num_with_cat/1_box_plot")
          self.create_violin_plot(cat_col, num_col, folder_name="7_num_with_cat/2_violin_plot")
          self.create_ridge_plot(cat_col, num_col, folder_name="7_num_with_cat/3_ridge_plot")
        i += 1

  ########################################
  #             Multi-variate            #
  ########################################
  def visualize_multi_variate(self):
    self.create_folder("8_multi_variate/1_parallel_plot", verbose=False)
    self.create_folder("8_multi_variate/2_radar_plot", verbose=False)

    self.create_parallel_plot()
    self.create_radar_plot()


  # def visualize_cat_with_target(self):
  #     pass

  # def visualize_num_with_target(self):
  #     pass

  def visualize_all(self):
    # self.visualize_target()
    # self.visualize_cat()
    # self.visualize_num()
    # self.visualize_num_with_idx()
    # self.visualize_cat_with_idx()
    # self.visualize_cat_with_cat()
    # self.visualize_num_with_num()
    # self.visualize_num_with_cat()
    self.visualize_multi_variate()

if __name__ == '__main__':
    df = pd.read_csv("/media/mosaab/Volume/Personal/Development/Courses Docs/ML GrandMaster/ml_project/input/new_df.csv")
    path = '/media/mosaab/Volume/Personal/Development/Courses Docs/ML GrandMaster/ml_project'
    num_cols  = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
            'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    vis = Visualizer(path=path, df=df, target_col='SalePrice', problem_type="regression")
    vis.visualize_all()