import umap
import networkx as nx
import cylouvain
import matplotlib.pyplot as plt
import seaborn as sns



def perform_umap_louvain_clustering_binneddata(df_norm, labels, labelskeys, titlestring, palette_dict,
                                    NN=15, min_dist=0.05, metric='euclidean', 
                                    louvain_resolution=1, 
                                    random_state=111, louvain_plot=True,
                                    supervised=False, supervision_labels=None):

    # UMAP reduction
    reducer = umap.UMAP(random_state=random_state, n_components=2, 
                        n_neighbors=NN, min_dist=min_dist, metric=metric)
    if supervised:
        mapper = reducer.fit(df_norm, y=supervision_labels) #supervised UMAP
    else:
        mapper = reducer.fit(df_norm)
        
    # Get UMAP graph and apply Louvain clustering
    G = nx.from_scipy_sparse_array(mapper.graph_)
    clustering = cylouvain.best_partition(G, resolution=louvain_resolution)
    clustering_solution = list(clustering.values())

    # Apply embedding to data
    if supervised:
        embedding = reducer.fit_transform(df_norm, y=supervision_labels) #supervised UMAP
    else:
        embedding = reducer.fit_transform(df_norm)

    #embedding = reducer.fit_transform(df_norm)
    umap_df = pd.DataFrame(embedding, columns=('x', 'y'))
    umap_df['cluster_id'] = clustering_solution
    umap_df['labelskeys_behaviour'] = labelskeys #labels STRINGS
    umap_df['labels_behaviour'] = labels #labels Numeric


    # Create plots
    if louvain_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7.5), dpi=300)

    # First subplot: UMAP space colored by activity
    sns.scatterplot(data=umap_df, x='x', y='y', hue='labelskeys_behaviour', 
                    palette=palette_dict, ax=ax1, s=15,alpha=0.75)
    legend = ax1.legend(scatterpoints=1, markerscale=2.5)
    legend.set_title('Behaviour')
    ax1.set_title(f'{titlestring}. \n UMAP, Colored by Behaviour \n '
                  f'NN={NN}, min_dist={min_dist}, metric={metric}')
    ax1.set_xlabel('UMAP X')
    ax1.set_ylabel('UMAP Y')

    if louvain_plot:
        # Second subplot: Louvain graph clustering
        sns.scatterplot(data=umap_df, x='x', y='y', hue='cluster_id', 
                        palette='turbo', ax=ax2, legend='full', s=13,alpha=0.75)
        ax2.legend(scatterpoints=1, markerscale=2.5)
        ax2.set_title(f'UMAP, Louvain high-dim graph clustering. '
                      f'louvain_resolution={louvain_resolution}')
        ax2.set_xlabel('UMAP X')
        ax2.set_ylabel('UMAP Y')
    
    plt.tight_layout()

    return fig, umap_df



def group_shap_analysis_plots(shap_values, feature_names, absolute=True, threshold=None, 
                         plot_type='boxplot', n_bins=50, kde=True, 
                         group1_label='Group 1', group2_label='Group 2',
                         arraygroup1=['46v','45a'], arraygroup2=['F5hand','F5mouth'],
                         figsize=(15, 5), dpi=300,
                         background_color='whitesmoke',
                         group1_color='blue',
                         group2_color='green'):    
    """
    Parameters remain the same as before
    """
    
    if type(shap_values) is not list:
        shap_values=[shap_values]
    
    # Define feature groups
    group1 = [f for f in feature_names if any(term in f for term in arraygroup1)]
    group2 = [f for f in feature_names if any(term in f for term in arraygroup2)]
    
    # Function to extract SHAP values for a specific group and class
    def get_group_shap(group, class_index):
        group_indices = [feature_names.index(f) for f in group]
        group_shaps = shap_values[class_index][:, group_indices].ravel()
        return np.abs(group_shaps) if absolute else group_shaps
    
    # Prepare data for plotting
    group1_shaps = [get_group_shap(group1, cls) for cls in range(len(shap_values))]
    group2_shaps = [get_group_shap(group2, cls) for cls in range(len(shap_values))]
    
    # Create figure with custom background color
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(background_color)
    
    if absolute:
        # Combine SHAP values across all classes
        group1_combined = np.concatenate(group1_shaps)
        group2_combined = np.concatenate(group2_shaps)
        
        # Apply threshold if specified
        if threshold is not None:
            group1_combined = group1_combined[np.abs(group1_combined) > threshold]
            group2_combined = group2_combined[np.abs(group2_combined) > threshold]
        
        data = [group1_combined, group2_combined]
        if plot_type == 'boxplot':
            ax = plt.subplot(1, 1, 1)
            ax.set_facecolor(background_color)
            
            # Create boxplot with custom colors
            bp = plt.boxplot(data, labels=[group1_label, group2_label], 
                           patch_artist=True)  # Enable fill color
            
            # Set colors for each box
            colors = [group1_color, group2_color]
            for i, box in enumerate(bp['boxes']):
                # Set white face color and colored edges
                box.set(facecolor='white')  # Set face color to white
                box.set(edgecolor=colors[i])  # Set edge color to group color
                
            # Set colors for other elements
            for i, element in enumerate(['whiskers', 'caps', 'medians']):
                for j, item in enumerate(bp[element]):
                    item.set(color=colors[j//2])  # Integer division to match colors with boxes
                    
            for i, flier in enumerate(bp['fliers']):
                flier.set(markerfacecolor='white', markeredgecolor=colors[i])
                
            plt.title('Absolute SHAP Values by Group')
            plt.ylabel('Absolute SHAP Value')
            
        elif plot_type == 'violin':
            ax = plt.subplot(1, 2, 1)
            ax.set_facecolor(background_color)
            plt.violinplot(data, showmeans=True)
            plt.xticks([1, 2], [group1_label, group2_label])
            plt.title('Absolute SHAP Values')
            plt.ylabel('Absolute SHAP Value')
        else:  # histogram
            ax = plt.subplot(1, 2, 1)
            ax.set_facecolor(background_color)
            plt.hist(group1_combined, bins=n_bins, alpha=0.6, label=group1_label, density=True)
            plt.hist(group2_combined, bins=n_bins, alpha=0.6, label=group2_label, density=True)
            if kde:
                from scipy import stats
                for data, label in zip([group1_combined, group2_combined], [group1_label, group2_label]):
                    kernel = stats.gaussian_kde(data)
                    x_range = np.linspace(min(data), max(data), 200)
                    plt.plot(x_range, kernel(x_range), label=f'{label} KDE')
            plt.title('Absolute SHAP Values')
            plt.xlabel('SHAP Value')
            plt.ylabel('Density')
            plt.legend()
    else:
        # Separate positive and negative SHAP values
        group1_pos = [g[g > 0] for g in group1_shaps]
        group1_neg = [g[g < 0] for g in group1_shaps]
        group1_pos_combined = np.concatenate(group1_pos)
        group1_neg_combined = np.concatenate(group1_neg)
        
        group2_pos = [g[g > 0] for g in group2_shaps]
        group2_neg = [g[g < 0] for g in group2_shaps]
        group2_pos_combined = np.concatenate(group2_pos)
        group2_neg_combined = np.concatenate(group2_neg)
        
        # Apply threshold if specified
        if threshold is not None:
            group1_pos_combined = group1_pos_combined[np.abs(group1_pos_combined) > threshold]
            group2_pos_combined = group2_pos_combined[np.abs(group2_pos_combined) > threshold]
            group1_neg_combined = group1_neg_combined[np.abs(group1_neg_combined) > threshold]
            group2_neg_combined = group2_neg_combined[np.abs(group2_neg_combined) > threshold]
        
        if plot_type == 'boxplot':
            # Positive values plot
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_facecolor(background_color)
            if len(group1_pos_combined) > 0 and len(group2_pos_combined) > 0:
                bp1 = plt.boxplot([group1_pos_combined, group2_pos_combined], 
                                labels=[group1_label, group2_label],
                                patch_artist=True)
                
                # Set colors for positive values
                colors = [group1_color, group2_color]
                for i, box in enumerate(bp1['boxes']):
                    box.set(facecolor='white')
                    box.set(edgecolor=colors[i])
                
                for i, element in enumerate(['whiskers', 'caps', 'medians']):
                    for j, item in enumerate(bp1[element]):
                        item.set(color=colors[j//2])
                        
                for i, flier in enumerate(bp1['fliers']):
                    flier.set(markerfacecolor='white', markeredgecolor=colors[i])
                    
            plt.title('Positive SHAP Values by Group')
            plt.ylabel('Positive SHAP Value')
            
            # Negative values plot
            ax2 = plt.subplot(1, 2, 2)
            ax2.set_facecolor(background_color)
            if len(group1_neg_combined) > 0 and len(group2_neg_combined) > 0:
                bp2 = plt.boxplot([group1_neg_combined, group2_neg_combined], 
                                labels=[group1_label, group2_label],
                                patch_artist=True)
                
                # Set colors for negative values
                for i, box in enumerate(bp2['boxes']):
                    box.set(facecolor='white')
                    box.set(edgecolor=colors[i])
                
                for i, element in enumerate(['whiskers', 'caps', 'medians']):
                    for j, item in enumerate(bp2[element]):
                        item.set(color=colors[j//2])
                        
                for i, flier in enumerate(bp2['fliers']):
                    flier.set(markerfacecolor='white', markeredgecolor=colors[i])
                    
            plt.title('Negative SHAP Values by Group')
            plt.ylabel('Negative SHAP Value')
            
        elif plot_type == 'violin':
            plt.subplot(1, 2, 1)
            ax = plt.gca()
            ax.set_facecolor(background_color)
            if len(group1_pos_combined) > 0 and len(group2_pos_combined) > 0:
                plt.violinplot([group1_pos_combined, group2_pos_combined], showmeans=True)
            plt.xticks([1, 2], [group1_label, group2_label])
            plt.title('Positive SHAP Values')
            plt.ylabel('Positive SHAP Value')
            
            plt.subplot(1, 2, 2)
            ax = plt.gca()
            ax.set_facecolor(background_color)
            if len(group1_neg_combined) > 0 and len(group2_neg_combined) > 0:
                plt.violinplot([group1_neg_combined, group2_neg_combined], showmeans=True)
            plt.xticks([1, 2], [group1_label, group2_label])
            plt.title('Negative SHAP Values')
            plt.ylabel('Negative SHAP Value')
            
        else:  # histogram
            plt.subplot(1, 2, 1)
            ax = plt.gca()
            ax.set_facecolor(background_color)
            plt.hist(group1_pos_combined, bins=n_bins, alpha=0.6, 
                    label=group1_label, density=True, color=group1_color)
            plt.hist(group2_pos_combined, bins=n_bins, alpha=0.6, 
                    label=group2_label, density=True, color=group2_color)
            if kde:
                from scipy import stats
                for data, label, color in zip([group1_pos_combined, group2_pos_combined], 
                                            [group1_label, group2_label],
                                            [group1_color, group2_color]):
                    if len(data) > 0:
                        kernel = stats.gaussian_kde(data)
                        x_range = np.linspace(min(data), max(data), 200)
                        plt.plot(x_range, kernel(x_range), label=f'{label} KDE', color=color)
            plt.title('Positive SHAP Values')
            plt.xlabel('SHAP Value')
            plt.ylabel('Density')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            ax = plt.gca()
            ax.set_facecolor(background_color)
            plt.hist(group1_neg_combined, bins=n_bins, alpha=0.6, 
                    label=group1_label, density=True, color=group1_color)
            plt.hist(group2_neg_combined, bins=n_bins, alpha=0.6, 
                    label=group2_label, density=True, color=group2_color)
            if kde:
                for data, label, color in zip([group1_neg_combined, group2_neg_combined], 
                                            [group1_label, group2_label],
                                            [group1_color, group2_color]):
                    if len(data) > 0:
                        kernel = stats.gaussian_kde(data)
                        x_range = np.linspace(min(data), max(data), 200)
                        plt.plot(x_range, kernel(x_range), label=f'{label} KDE', color=color)
            plt.title('Negative SHAP Values')
            plt.xlabel('SHAP Value')
            plt.ylabel('Density')
            plt.legend()
    
    plt.tight_layout()
    return fig


def group_shap_analysis_plots_by_category(shap_values, feature_names, class_names, absolute=True, threshold=None,  
                            group1_label='Group 1', group2_label='Group 2',
                            arraygroup1=['46v','45a'], arraygroup2=['F5hand','F5mouth'],
                            figsize=(15, 10), dpi=300):       
    """
    Parameters:
    -----------
    shap_values : array-like
        SHAP values to analyze
    feature_names : list
        List of feature names corresponding to the features
    class_names : list
        List of class names corresponding to the shap_values classes
    absolute : bool, default=True
        Whether to use absolute SHAP values
    threshold : float or None
        If provided, only show SHAP values with absolute value greater than threshold
    group1_label : str, default='Group 1'
        Label for the first group of features
    group2_label : str, default='Group 2'
        Label for the second group of features
    arraygroup1 : list, default=['46v','45a']
        Terms to identify features belonging to group 1
    arraygroup2 : list, default=['F5hand','F5mouth']
        Terms to identify features belonging to group 2
    """

    if type(shap_values) is not list: #if the problem is binary
        shap_values=[shap_values]
        is_binary_problem=True
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        return fig
        
    # Define feature groups
    group1 = [f for f in feature_names if any(term in f for term in arraygroup1)]
    group2 = [f for f in feature_names if any(term in f for term in arraygroup2)]
    
    # Function to extract SHAP values for a specific group and class
    def get_group_shap(group, class_index):
        group_indices = [feature_names.index(f) for f in group]
        group_shaps = shap_values[class_index][:, group_indices].ravel()
        return np.abs(group_shaps) if absolute else group_shaps
    
    # Define colors for each group (for box edges)
    group1_color = 'blue'
    group2_color = 'green'
    
    num_classes = len(shap_values)
    if absolute:
        # Create a single figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Create legend handles
        legend_elements = [
            plt.Line2D([0], [0], color=group1_color, label=group1_label, linewidth=2),
            plt.Line2D([0], [0], color=group2_color, label=group2_label, linewidth=2)]
        # Add the legend to the axis
        ax.legend(handles=legend_elements, loc='upper right')
        
        
        # Prepare data for all classes
        all_data = []
        labels = []
        colors = []
        
        # Get data for each class and group
        for i, class_name in enumerate(class_names):
            group1_shaps = get_group_shap(group1, i)
            group2_shaps = get_group_shap(group2, i)
            
            # Apply threshold if specified
            if threshold is not None:
                group1_shaps = group1_shaps[np.abs(group1_shaps) > threshold]
                group2_shaps = group2_shaps[np.abs(group2_shaps) > threshold]
            
            
            class_name_nospace=class_name.replace('_',' ') #replace the underscore for a space in the clas name
            all_data.extend([group1_shaps, group2_shaps])
            labels.extend([f'{class_name_nospace}', f'{class_name_nospace}'])
            #labels.extend([f'{class_name} {group1_label}', f'{class_name} {group2_label}'])
            colors.extend([group1_color, group2_color])
        
        # Create boxplot with patch_artist=True to allow filling boxes with color
        bp = ax.boxplot(all_data, labels=labels, patch_artist=True)
        
        # Set colors for boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor('white')  # Transparent fill
            patch.set_alpha(1.0)          # Full opacity for the white fill
            patch.set_edgecolor(color)    # Colored edges
            
        # Set colors for other elements
        for element in ['whiskers', 'caps', 'medians']:
            for idx in range(len(bp[element])):
                # Integer division by 2 since there are two whiskers/caps per box
                box_idx = idx // 2 if element == 'medians' else idx // 2
                bp[element][idx].set_color(colors[box_idx])
        
        for i, flier in enumerate(bp['fliers']):
            flier.set(markerfacecolor='white', markeredgecolor=colors[i])
            
        
        # Customize plot
        ax.set_title('Absolute SHAP Values by Class and Group')
        ax.set_ylabel('Absolute SHAP Value')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        
        # Add alternating background colors to distinguish classes
        # for i in range(0, num_classes*2, 2):
        #     ax.axvspan(i+0.5, i+2.5, facecolor='white', alpha=0.1)
        # Set a single background color for the plot
        ax.set_facecolor('whitesmoke')  # Light gray background

    else:
        # Create two subplots - one for positive and one for negative values
        fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        
        # Create legend handles
        legend_elements = [
            plt.Line2D([0], [0], color=group1_color, label=group1_label, linewidth=2),
            plt.Line2D([0], [0], color=group2_color, label=group2_label, linewidth=2)]
        # Add the legend to both axes
        ax_pos.legend(handles=legend_elements, loc='upper left')
        ax_neg.legend(handles=legend_elements, loc='lower left')
        
        
        # Prepare data for all classes
        pos_data = []
        neg_data = []
        pos_labels = []
        neg_labels = []
        pos_colors = []
        neg_colors = []
        
        # Get data for each class and group
        for i, class_name in enumerate(class_names):
            group1_shaps = get_group_shap(group1, i)
            group2_shaps = get_group_shap(group2, i)
            
            # Separate positive and negative values
            group1_pos = group1_shaps[group1_shaps > 0]
            group1_neg = group1_shaps[group1_shaps < 0]
            group2_pos = group2_shaps[group2_shaps > 0]
            group2_neg = group2_shaps[group2_shaps < 0]
            
            # Apply threshold if specified
            if threshold is not None:
                group1_pos = group1_pos[np.abs(group1_pos) > threshold]
                group1_neg = group1_neg[np.abs(group1_neg) > threshold]
                group2_pos = group2_pos[np.abs(group2_pos) > threshold]
                group2_neg = group2_neg[np.abs(group2_neg) > threshold]
            
            
            class_name_nospace=class_name.replace('_',' ') #replace the underscore for a space
            # Add to data lists if there are values
            if len(group1_pos) > 0:
                pos_data.append(group1_pos)
                pos_labels.append(f'{class_name_nospace}')
                #pos_labels.append(f'{class_name} {group1_label}')
                pos_colors.append(group1_color)
            if len(group2_pos) > 0:
                pos_data.append(group2_pos)
                pos_labels.append(f'{class_name_nospace}')
                #pos_labels.append(f'{class_name} {group2_label}')
                pos_colors.append(group2_color)
            if len(group1_neg) > 0:
                neg_data.append(group1_neg)
                neg_labels.append(f'{class_name_nospace}')
                #neg_labels.append(f'{class_name} {group1_label}')
                neg_colors.append(group1_color)
            if len(group2_neg) > 0:
                neg_data.append(group2_neg)
                neg_labels.append(f'{class_name_nospace}')
                #neg_labels.append(f'{class_name} {group2_label}')
                neg_colors.append(group2_color)
        
        # Plot positive values
        if pos_data:
            bp_pos = ax_pos.boxplot(pos_data, labels=pos_labels, patch_artist=True)
            # Set colors for boxes
            for patch, color in zip(bp_pos['boxes'], pos_colors):
                patch.set_facecolor('white')  # Transparent fill
                patch.set_alpha(1.0)          # Full opacity for the white fill
                patch.set_edgecolor(color)    # Colored edges
                
            # Set colors for other elements
            for element in ['whiskers', 'caps', 'medians']:
                for idx in range(len(bp_pos[element])):
                    # Integer division by 2 since there are two whiskers/caps per box
                    box_idx = idx // 2 if element == 'medians' else idx // 2
                    bp_pos[element][idx].set_color(pos_colors[box_idx])
            
            for i, flier in enumerate(bp_pos['fliers']):
                flier.set(markerfacecolor='white', markeredgecolor=pos_colors[i])
                
                     
            # Add alternating background colors to distinguish classes
            # for i in range(0, len(pos_data), 2):
            #     ax_pos.axvspan(i+0.5, i+2.5, facecolor='gray', alpha=0.1)
            ax_pos.set_facecolor('whitesmoke')
            
            
        # Plot negative values
        if neg_data:
            bp_neg = ax_neg.boxplot(neg_data, labels=neg_labels, patch_artist=True)
            # Set colors for boxes
            for patch, color in zip(bp_neg['boxes'], neg_colors):
                patch.set_facecolor('white')  # Transparent fill
                patch.set_alpha(1.0)          # Full opacity for the white fill
                patch.set_edgecolor(color)    # Colored edges
                
            # Set colors for other elements
            for element in ['whiskers', 'caps', 'medians']:
                for idx in range(len(bp_neg[element])):
                    # Integer division by 2 since there are two whiskers/caps per box
                    box_idx = idx // 2 if element == 'medians' else idx // 2
                    bp_neg[element][idx].set_color(neg_colors[box_idx])
            
            for i, flier in enumerate(bp_neg['fliers']):
                flier.set(markerfacecolor='white', markeredgecolor=neg_colors[i])
                
            
            # Add alternating background colors to distinguish classes
            # for i in range(0, len(neg_data), 2):
            #     ax_neg.axvspan(i+0.5, i+2.5, facecolor='gray', alpha=0.1)
            ax_neg.set_facecolor('whitesmoke')
            
        # Customize plots
        ax_pos.set_title('Positive SHAP Values by Class and Group')
        ax_pos.set_ylabel('SHAP Value')
        ax_neg.set_title('Negative SHAP Values by Class and Group')
        ax_neg.set_ylabel('SHAP Value')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax_pos.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.setp(ax_neg.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.tight_layout()
    return fig

