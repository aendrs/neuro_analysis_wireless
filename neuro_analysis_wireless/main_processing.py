from .data_loading import load_datasets_Jacopo_v2, load_and_normalize_datasets
from .data_analysis import perform_umap_louvain_clustering_binneddata
from .model_training import xgboost_train_and_eval_gridsearch_stratifiedkfold
from .reporting import insert_plot_ppt, insert_txt_ppt
from .utils import get_labelsdict_palette_SamovarJanuary2025, get_array_folders_january2025


def plot_umap_by_session_ppt_stratifiedkfold(folders,labelsdict, monkey, session, arrayname, palette_dict,
                             pptfile, stateevents=False,
                             metric='euclidean', louvain_resolution=1, map_categories=None,
                             supervised_UMAP=False, shap_analysis=False, class_pairs=None):
    
    if stateevents:
        stateorpointevents='State events'
    else:
        stateorpointevents='Point events'
            
    if len(labelsdict)==0:
        return
        
    normtype='standardscaler'
    titlestring=f"{monkey} session {session}, {arrayname}. {stateorpointevents}."
    NN=20
    min_dist=0.05


    # UMAP
    df_norm, labels, labelskeys, modifiers, sessions, conditions, col_names, times = load_and_normalize_datasets(folders, 
                                                                                               labelsdict, 
                                                                                               removemodifiers=True, 
                                                                                               normtype=normtype)
    
    
    if map_categories:
        labelsdict, labels, labelskeys, palette_dict = unify_categories(map_categories, labelsdict, labels, labelskeys, palette_dict)
    
    
    
    #clean labels vector and class labels from labelsdict 
    #this is mainly a quick fix for putting the right names on the plots, it could be improved for better generalisation
    clean_labelskeys=remove_substring(labelskeys, substring='_-500_500')
    clean_labelskeys=[item.replace('_', ' ') for item in clean_labelskeys]

    clean_class_labels = [key.replace('_-500_500', '') for key in labelsdict.keys()]
    clean_class_labels = [item.replace('_', ' ') for item in clean_class_labels]

    clean_palette_dict = {key.replace('_-500_500', ''): value for key, value in palette_dict.items()}
    clean_palette_dict = {key.replace('_', ' '): value for key, value in clean_palette_dict.items()}

    clean_labelsdict = {key.replace('_-500_500', ''): value for key, value in labelsdict.items()}
    clean_labelsdict = {key.replace('_', ' '): value for key, value in clean_labelsdict.items()}

    # Louvain clustering
    fig, umap_df = perform_umap_louvain_clustering_binneddata(df_norm, labels, clean_labelskeys, titlestring, 
                                                              clean_palette_dict,
                                                              NN=NN, min_dist=min_dist, metric=metric, 
                                                              louvain_resolution=1, normtype=normtype, 
                                                              random_state=111, louvain_plot=False)
    
    # Umap and Louv clustering plot embedding in ppt
    #insert_plot_ppt(pptfile, left=Inches(0), top=Inches(1.2),width=Inches(16), insertinlastslide=False)
    insert_plot_ppt(pptfile, left=Inches(0), top=Inches(1.2),width=Inches(8), insertinlastslide=False)

    insert_txt_ppt(pptfile, left=Inches(0.5), top=Inches(0.1), width=Inches(10), height=Inches(2), 
                   stringx=titlestring, fontsize=20)


    # CLASSIFIER =========================================================
    X_train, X_test, y_train, y_test = train_test_split(df_norm, 
                                                        labels, 
                                                        test_size=0.3, 
                                                        random_state=111, 
                                                        stratify=labels)
    
    grid_search, report, _= xgboost_train_and_eval_gridsearch_stratifiedkfold(X_train, 
                                                                       X_test, 
                                                                       y_train, 
                                                                       y_test,
                                                                       feature_names=col_names,
                                                                       random_state=111, 
                                                                       weigh_classes=True,
                                                                       labelsdict=clean_labelsdict)
    
    # show results from single partition, will most likely be superseded by results from average CV folds
    y_pred=grid_search.predict(X_test)
   
    insert_classification_report(report, pptfile, 
                                 left=1,top=2,width=5,height=4,insertinlastslide=False)
    insert_normalized_confusion_matrix(y_test, y_pred, pptfile, clean_labelsdict,
                                       left=8,top=0.5,width=8,insertinlastslide=True)   
    insert_txt_ppt(pptfile, left=Inches(1), top=Inches(0.1), width=Inches(10), height=Inches(2), 
                   stringx=f'{titlestring}\n Classification Report (performed on Single patition), XgBoost, \n stratified_K_fold parameter grid search', fontsize=20)

    
    # AVERAGE PEFORMANCE across CV folds
    # cross validation using the best grid_search model, get the average confmat and classification metrics

    grid_search_cv, report_dict, report_lines =  average_cv_from_gridsearch_bestmodel(
                                                                                    grid_search, 
                                                                                    X=df_norm, 
                                                                                    y=labels, 
                                                                                    RAND_STATE=111,
                                                                                    n_splits=5,
                                                                                    weigh_classes=True, 
                                                                                    labelsdict=clean_labelsdict,
                                                                                    additional_metrics=None
                                                                                )
    
    insert_classification_report(report_lines,pptfile,left=0,top=1,width=8,height=8,insertinlastslide=False)

    insert_txt_ppt(pptfile, left=Inches(1), top=Inches(0.1), width=Inches(10), height=Inches(2), 
                   stringx=f'{titlestring}\n Classification Report (5 fold Cross Val), XgBoost, \n stratified_K_fold parameter grid search', fontsize=20)

    insert_matrix_ppt(report_dict['normalized_confusion_matrix'], pptfile, clean_labelsdict, 
                      left=8,top=1,width=8, insertinlastslide=True, 
                      matrixtitle='')




    if supervised_UMAP:
        # Louvain clustering
        fig, umap_df = perform_umap_louvain_clustering_binneddata(df_norm, labels, labelskeys, titlestring, palette_dict,
                                                                  NN=NN, min_dist=min_dist, metric=metric, 
                                                                  louvain_resolution=1, normtype=normtype, 
                                                                  random_state=111, louvain_plot=False,
                                                                  supervised=True, supervision_labels=labels)
        
        # Umap and Louv clustering plot embedding in ppt
        insert_plot_ppt(pptfile, left=Inches(0), top=Inches(1.2),width=Inches(8), insertinlastslide=False)
        insert_txt_ppt(pptfile, left=Inches(0.5), top=Inches(0.1), width=Inches(10), height=Inches(2), 
                       stringx=titlestring+'\n SUPERVISED UMAP', fontsize=20)
        
   
    

    #clean labels vector a and class labels from labelsdict (edit: it was done before in this same function)
    clean_labelskeys2=remove_substring(labelskeys, substring='_-500_500')
    #clean_class_labels = [key.rsplit('_-500_500')[0] for key in labelsdict.keys()]
    
    # CALCULATE class separation METRICS on ORIGINIAL SPACE, build a DF, print a table in the pptx

    if class_pairs is None: 
        class_pairs=[['Grasping_grooming','Spread_grooming'],['Grasping_foraging','Spread_foraging'],
                     ['Grasping_foraging','Grasping_grooming'],['Spread_foraging','Spread_grooming'],
                     (['Grasping_grooming','Spread_grooming'],['Grasping_foraging','Spread_foraging']),
                     (['Spread_foraging','Spread_grooming'],['Grasping_foraging','Grasping_grooming'])]
    
    separability_df=create_separability_metrics_df(df_norm, clean_labelskeys2, class_pairs)
    
    insert_df_to_pptx(
        separability_df,
        pptfile,
        slide_title="Class separability metrics, original vector space",
        layout_idx=5,
        font_name='Arial',
        title_font_size=28,
        table_font_size=14,
        decimal_places=4,
        slide_width_inches=16,
        slide_height_inches=9,
        table_start_x=0.5,
        table_start_y=1.5,
        table_width=15,
        table_height=4,
    )
    
    # CALCULATE class separation METRICS on UMAP SPACE, build a DF, print a table in the pptx
    # class_pairs=[['Grasping_grooming','Spread_grooming'],['Grasping_foraging','Spread_foraging'],
    #              ['Grasping_foraging','Grasping_grooming'],['Spread_foraging','Spread_grooming'],
    #              (['Grasping_grooming','Spread_grooming'],['Grasping_foraging','Spread_foraging']),
    #              (['Spread_foraging','Spread_grooming'],['Grasping_foraging','Grasping_grooming'])]
    separability_df=create_separability_metrics_df(umap_df[['x','y']], clean_labelskeys2, class_pairs)
    
    insert_df_to_pptx(
        separability_df,
        pptfile,
        slide_title="Class separability metrics, calculated on UMAP 2D space",
        layout_idx=5,
        font_name='Arial',
        title_font_size=24,
        table_font_size=14,
        decimal_places=4,
        slide_width_inches=16,
        slide_height_inches=9,
        table_start_x=0.5,
        table_start_y=1.5,
        table_width=15,
        table_height=4,
    )

    
    if shap_analysis:
        if len(np.unique(y_train)) == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softmax'
            
        xgbModel = xgb.XGBClassifier(**grid_search.best_params_, objective=objective, verbosity=1)    
        xgbModel.fit(df_norm, labels)
        explainer = shap.TreeExplainer(xgbModel)
        shap_values = explainer.shap_values(df_norm)
        shap.summary_plot(shap_values, df_norm, feature_names=col_names, 
                  class_names=list(labelsdict.keys()), 
                  class_inds='original', show=True, plot_size=(10,8))


        insert_shap_summary_ppt(pptfile,shap_values, df_norm, col_names, labelsdict, title=titlestring)
        if len(np.unique(y_train)) > 2:
            insert_shap_images_by_category_ppt(pptfile, shap_values, df_norm, labelsdict, col_names,
                                           slidetitle=titlestring, imagesperslide=4, slidewidth=16, slideheight=9)
        
        
        
        #choose the right arrays to form groups based on the case under consideration
        if arrayname=='All Arrays': #all arrays
            group1_label='Pre-Frontal'
            group2_label='Pre-Motor'
            arraygroup1=['46v','45a']
            arraygroup2=['F5hand','F5mouth']
        elif arrayname=='Pre-Frontal': #Pre Frontal
            group1_label='46v'
            group2_label='45a'
            arraygroup1=['46v']
            arraygroup2=['45a']
        elif arrayname=='Pre-Motor': #Pre Motor
            group1_label='F5hand'
            group2_label='F5mouth'
            arraygroup1=['F5hand']
            arraygroup2=['F5mouth']
        else:
            raise ValueError ('arrayname is not valid. Should be either "All Arrays", "Pre-Frontal" or "Pre-Motor"')
        
        shap_abs_std=get_shap_std(shap_values)
        
        # PLOT general overall Boxplots (by group)      
        for i, thresh in enumerate([None, shap_abs_std, 1.0]):
            
            if i%2==0:
                vertical_offset=0 
                insertinlastslide=False
            else:
                vertical_offset=4.5
                insertinlastslide=True
            
            group_shap_analysis_plots(shap_values, feature_names=col_names, absolute=True, 
                                threshold=thresh, plot_type='boxplot', n_bins=50, kde=True, 
                                group1_label=group1_label, group2_label=group2_label, 
                                arraygroup1=arraygroup1, arraygroup2=arraygroup2, figsize=(6, 5), dpi=300)
            insert_plot_ppt(pptfile, left=Inches(0.75), top=Inches(0.5+vertical_offset),width=Inches(4.3), insertinlastslide=insertinlastslide)


            group_shap_analysis_plots(shap_values, feature_names=col_names, absolute=False,  
                                threshold=thresh, plot_type='boxplot', n_bins=50, kde=True, 
                                group1_label=group1_label, group2_label=group2_label, 
                                arraygroup1=arraygroup1, arraygroup2=arraygroup2, figsize=(12, 5), dpi=300)
            insert_plot_ppt(pptfile, left=Inches(6), top=Inches(0.5+vertical_offset),width=Inches(8.6), insertinlastslide=True)

            insert_txt_ppt(pptfile, left=Inches(0.5), top=Inches(vertical_offset-0.25), width=Inches(10), height=Inches(2), 
                           stringx=f'Shap values distributions. Threshold ={thresh}', fontsize=20)

        # Boxplots by CATEGORY
        for i, thresh in enumerate([None, shap_abs_std, 1.0]): #it is necessary to do a new loop in order to insert everything correctly in its appropriate ppt slide         
            if i%2==0:
                vertical_offset=0 
                insertinlastslide=False
            else:
                vertical_offset=4.5
                insertinlastslide=True
                
            clean_class_labels = [key.rsplit('_-500_500')[0] for key in labelsdict.keys()]

            group_shap_analysis_plots_by_category(shap_values, feature_names=col_names, class_names=clean_class_labels ,absolute=True, 
                                threshold=thresh, group1_label=group1_label, group2_label=group2_label, 
                                arraygroup1=arraygroup1, arraygroup2=arraygroup2, figsize=(6, 6), dpi=300)
            insert_plot_ppt(pptfile, left=Inches(0.75), top=Inches(0.5+vertical_offset),width=Inches(4.3), insertinlastslide=insertinlastslide)

            group_shap_analysis_plots_by_category(shap_values, feature_names=col_names, class_names=clean_class_labels ,absolute=False, 
                                threshold=thresh, group1_label=group1_label, group2_label=group2_label, 
                                arraygroup1=arraygroup1, arraygroup2=arraygroup2, figsize=(12, 6), dpi=300)
            insert_plot_ppt(pptfile, left=Inches(6), top=Inches(0.5+vertical_offset),width=Inches(8.6), insertinlastslide=True)
            insert_txt_ppt(pptfile, left=Inches(0.5), top=Inches(vertical_offset-0.25), width=Inches(10), height=Inches(2), 
                           stringx=f'Shap values distributions by Behaviour. Threshold ={thresh}', fontsize=20)
        

        for i, thresh in enumerate([None,shap_abs_std, 1.0]):
            #create a cool table and save it in the ppt
            shapstats = calculate_shap_statistics_by_category_and_group(shap_values, feature_names=col_names,
                                                                 class_names=clean_class_labels,
                                                                 group1_label=group1_label, group2_label=group2_label,
                                                                 arraygroup1=arraygroup1, arraygroup2=arraygroup2,
                                                                 threshold=thresh)
            shapstats=shapstats.assign(Overall_stats=calculate_overall_shap_statistics(shap_values))
            append_shap_table_to_pptx(shapstats, pptfile, f"SHAP Statistics by Class and Group. Threshold={thresh}")

        # save_variables_to_disk({'shap_values': shap_values, 
        #                         'shap_abs_std': shap_abs_std, 
        #                         'feature_names': col_names, 
        #                         'labels_num': labels,
        #                         'labels_str':clean_labelskeys}, 
        #                        file_path=f'C:\\Users\\amendez\\Documents\\Jacopo\\shap_variables_{session}_{arrayselection}.pkl')


