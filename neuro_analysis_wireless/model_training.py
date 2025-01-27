from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
import xgboost as xgb
import numpy as np
import logging
from datetime import datetime



def xgboost_train_and_eval_gridsearch_stratifiedkfold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
    random_state: int = 42,
    weigh_classes: bool = False,
    labelsdict: Optional[Dict[str, int]] = None,
    scoring: str = 'f1_macro',
    n_jobs: int = -1,
    save_model: bool = False,
    early_stopping_rounds: int = 10,
    model_name: Optional[str] = None
) -> Tuple[GridSearchCV, str, Dict]:
    """
    Train and evaluate XGBoost classifier using GridSearchCV with improved parameters and functionality.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        feature_names: List of feature names for importance analysis
        random_state: Random state for reproducibility
        weigh_classes: Whether to use class weights
        labelsdict: Dictionary mapping class names to class indices 
                    Format: {'class0_name':0, 'class1_name':1, ...}
        scoring: Scoring metric for GridSearchCV
        n_jobs: Number of parallel jobs
        save_model: Whether to save the best model
        early_stopping_rounds: Number of rounds for early stopping
        model_name: Custom name for saved model
        
    Returns:
        Tuple of (GridSearchCV object, classification report, feature importance dict)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    if type(y_test)==list:
        y_train=np.array(y_train)
        y_test=np.array(y_test)
    
    # Input validation
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    
    # Determine number of classes
    n_classes = len(np.unique(y_train))
    
    # Set up base model
    objective = 'binary:logistic' if n_classes == 2 else 'multi:softmax'
    model = xgb.XGBClassifier(
        objective=objective,
        verbosity=0,
        random_state=random_state,
        use_label_encoder=False  # Prevents warning about label encoder
    )
    
    # Define parameter grid
    '''
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5, 7],
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2]
    }
    '''
    
    param_grid = {"max_depth": [3,5,7,9],
          "min_child_weight" : [1 ,2, 3,4 ], 
          "n_estimators": [10,25,50,100,200], #10,25
          "learning_rate": [0.05,0.1,0.3,0.5],
          "subsample": [0.8, 0.9, 1.0],
          "seed": [random_state]
          }
        
    # Set up cross-validation
    cv = StratifiedKFold(
        n_splits=5, #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        shuffle=True,
        random_state=random_state
    )
    
    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        verbose=0,
        n_jobs=n_jobs,
        scoring=scoring,
        return_train_score=True
    )
    
    try:
        # Compute class weights if requested
        if weigh_classes:
            logger.info("Computing class weights...")
            sample_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=y_train
            )
        else:
            sample_weights = None
            
        # Create evaluation set for early stopping
        eval_set = [(X_test, y_test)]
        
        # Fit the model
        logger.info("Starting grid search...")
        start_time = datetime.now()
        
        grid_search.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=['error', 'auc'] if n_classes == 2 else ['merror', 'mlogloss']
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Generate predictions and classification report
        y_pred = grid_search.predict(X_test)
        
        # Prepare target names for classification report
        if labelsdict:
            # Invert the dictionary to map indices to names
            # Assumes labelsdict is in format {'class0_name':0, 'class1_name':1, ...}
            target_names = [
                name for name, _ in sorted(labelsdict.items(), key=lambda x: x[1])
            ]
            
            report = classification_report(
                y_test,
                y_pred,
                target_names=target_names
            )
        else:
            report = classification_report(y_test, y_pred)
            
        # Calculate and sort feature importance
        feature_importance = {}
        if feature_names is not None:
            importance = grid_search.best_estimator_.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        # Save the best model if requested
        if save_model:
            model_filename = model_name or f'xgboost_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(grid_search.best_estimator_, model_filename)
            logger.info(f"Best model saved as {model_filename}")
            
        # Log best parameters and scores
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        logger.info(f"Test set score: {grid_search.score(X_test, y_test):.4f}")
        
        return grid_search, report, feature_importance
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise



def xgboost_train_and_eval_randomizedsearch_stratifiedkfold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
    random_state: int = 42,
    weigh_classes: bool = False,
    labelsdict: Optional[Dict[str, int]] = None,
    scoring: str = 'f1_macro',
    n_jobs: int = -1,
    save_model: bool = False,
    early_stopping_rounds: int = 10,
    model_name: Optional[str] = None,
    n_iter: int = 100  # Number of parameter iterations to sample
     ) -> Tuple[RandomizedSearchCV, str, Dict]:
    """
    Train and evaluate XGBoost classifier using RandomizedSearchCV with improved parameters and functionality.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        feature_names: List of feature names for importance analysis
        random_state: Random state for reproducibility
        weigh_classes: Whether to use class weights
        labelsdict: Dictionary mapping class names to class indices 
        scoring: Scoring metric for RandomizedSearchCV
        n_jobs: Number of parallel jobs
        save_model: Whether to save the best model
        early_stopping_rounds: Number of rounds for early stopping
        model_name: Custom name for saved model
        n_iter: Number of parameter settings sampled
        
    Returns:
        Tuple of (RandomizedSearchCV object, classification report, feature importance dict)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if type(y_test)==list:
        y_train=np.array(y_train)
        y_test=np.array(y_test)
    
    # Input validation
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples")
    
    # Determine number of classes
    n_classes = len(np.unique(y_train))
    
    # Set up base model
    objective = 'binary:logistic' if n_classes == 2 else 'multi:softmax'
    model = xgb.XGBClassifier(
        objective=objective,
        verbosity=0,
        random_state=random_state,
        use_label_encoder=False
    )
    
    # Define parameter distributions for RandomizedSearchCV
    from scipy.stats import uniform, randint
    
    param_distributions = {
        # Tree-specific parameters
        "max_depth": randint(3, 10),  # Random integers between 3 and 10
        "min_child_weight": randint(1, 6),  # Random integers between 1 and 6
        
        # Boosting parameters
        "learning_rate": uniform(0.01, 0.4),  # Uniform between 0.01 and 0.41
        "n_estimators": randint(50, 300),  # Random integers between 50 and 300
        
        # Regularization parameters
        "gamma": uniform(0, 0.5),  # Uniform between 0 and 0.5
        
        # Sampling parameters
        "subsample": uniform(0.6, 0.4),  # Uniform between 0.6 and 1.0
        "colsample_bytree": uniform(0.6, 0.4),  # Uniform between 0.6 and 1.0
        
        # Regularization
        "reg_alpha": uniform(0, 1),  # L1 regularization
        "reg_lambda": uniform(0, 1)   # L2 regularization
    }
    
    # Set up cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )
    
    # Configure RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,  # Number of parameter settings to sample
        cv=cv,
        verbose=0,
        n_jobs=n_jobs,
        scoring=scoring,
        random_state=random_state,
        return_train_score=True
    )
    
    try:
        # Compute class weights if requested
        if weigh_classes:
            logger.info("Computing class weights...")
            sample_weights = class_weight.compute_sample_weight(
                class_weight='balanced',
                y=y_train
            )
        else:
            sample_weights = None
            
        # Create evaluation set for early stopping
        eval_set = [(X_test, y_test)]
        
        # Fit the model
        logger.info("Starting randomized search...")
        start_time = datetime.now()
        
        random_search.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=['error', 'auc'] if n_classes == 2 else ['merror', 'mlogloss']
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Generate predictions and classification report
        y_pred = random_search.predict(X_test)
        
        # Prepare target names for classification report
        if labelsdict:
            target_names = [
                name for name, _ in sorted(labelsdict.items(), key=lambda x: x[1])
            ]
            
            report = classification_report(
                y_test,
                y_pred,
                target_names=target_names
            )
        else:
            report = classification_report(y_test, y_pred)
            
        # Calculate and sort feature importance
        feature_importance = {}
        if feature_names is not None:
            importance = random_search.best_estimator_.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        # Save the best model if requested
        if save_model:
            model_filename = model_name or f'xgboost_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(random_search.best_estimator_, model_filename)
            logger.info(f"Best model saved as {model_filename}")
            
        # Log best parameters and scores
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        logger.info(f"Test set score: {random_search.score(X_test, y_test):.4f}")
        
        return random_search, report, feature_importance
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise



def average_cv_from_gridsearch_bestmodel(
    grid_search, 
    X, 
    y, 
    RAND_STATE=111,
    n_splits=5,
    weigh_classes=True, 
    labelsdict=None,
    additional_metrics=None
):
    """
    Perform cross-validation analysis on the best model from a grid search.
    
    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted GridSearchCV object containing the best model
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    RAND_STATE : int, default=111
        Random state for reproducibility
    n_splits : int, default=5
        Number of cross-validation splits
    weigh_classes : bool, default=True
        Whether to use sample weights for imbalanced classes
    labelsdict : dict, default=None
        Dictionary mapping class names to class indices (e.g., {'Class_A': 0, 'Class_B': 1})
    additional_metrics : dict, default=None
        Additional scoring metrics to include in cross-validation
    """
    # Define base scoring metrics
    scoring = {
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_weighted': make_scorer(recall_score, average='weighted'),
        'f1_weighted': make_scorer(f1_score, average='weighted')
    }
    
    if additional_metrics:
        scoring.update(additional_metrics)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RAND_STATE)
    best_model = grid_search.best_estimator_
    
    # Calculate sample weights if needed
    sample_weight = compute_sample_weight('balanced', y) if weigh_classes else None
    
    # Store confusion matrices and classification reports for each fold
    confusion_matrices = []
    classification_reports = []
    
    # Get class names if labelsdict is provided
    if labelsdict:
        inverse_labels = {v: k for k, v in labelsdict.items()}
        class_names = [inverse_labels[i] for i in sorted(inverse_labels.keys())]
    else:
        class_names = None
        
    y=np.asarray(y) #make sure y is a nparray
    
    # Compute metrics for each fold
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on fold
        model_clone = clone(best_model)
        if weigh_classes:
            fold_sample_weight = compute_sample_weight('balanced', y_train_fold)
            model_clone.fit(X_train_fold, y_train_fold, sample_weight=fold_sample_weight)
        else:
            model_clone.fit(X_train_fold, y_train_fold)
            
        # Get predictions and compute metrics
        y_pred_fold = model_clone.predict(X_val_fold)
        
        # Compute and store confusion matrix
        conf_matrix = confusion_matrix(y_val_fold, y_pred_fold)
        confusion_matrices.append(conf_matrix)
        
        # Compute and store classification report
        fold_report = classification_report(
            y_val_fold, 
            y_pred_fold,
            target_names=class_names,
            output_dict=True
        )
        classification_reports.append(fold_report)
    
    # Compute mean and std of confusion matrix
    mean_conf_matrix = np.mean(confusion_matrices, axis=0)
    std_conf_matrix = np.std(confusion_matrices, axis=0)
    
    # Compute normalized confusion matrix
    norm_conf_matrix = normalize_confusion_matrix(mean_conf_matrix)
    
    # Compute mean and std of classification report metrics
    mean_classification_report = {}
    std_classification_report = {}
    
    # Get all unique keys from the classification reports
    metric_keys = set()
    for report in classification_reports:
        metric_keys.update(report.keys())
    
    # Calculate mean and std for each metric
    for key in metric_keys:
        if key != 'accuracy':  # Handle nested metrics
            try:
                metric_values = np.array([report[key] for report in classification_reports])
                if isinstance(classification_reports[0][key], dict):
                    # Handle nested metrics (precision, recall, f1-score, support)
                    mean_classification_report[key] = {
                        k: np.mean([report[key][k] for report in classification_reports])
                        for k in classification_reports[0][key].keys()
                    }
                    std_classification_report[key] = {
                        k: np.std([report[key][k] for report in classification_reports])
                        for k in classification_reports[0][key].keys()
                    }
                else:
                    # Handle flat metrics
                    mean_classification_report[key] = np.mean(metric_values)
                    std_classification_report[key] = np.std(metric_values)
            except KeyError:
                continue
    
    # Perform standard cross-validation for other metrics
    cv_results = cross_validate(
        best_model, 
        X, 
        y, 
        cv=skf, 
        scoring=scoring,
        fit_params={'sample_weight': sample_weight} if weigh_classes else None,
        return_train_score=True
    )
    
    # Create results dictionary
    report_dict = {
        'cv_results': cv_results,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'mean_confusion_matrix': mean_conf_matrix,
        'std_confusion_matrix': std_conf_matrix,
        'normalized_confusion_matrix': norm_conf_matrix,
        'mean_classification_report': mean_classification_report,
        'std_classification_report': std_classification_report
    }
    
    # Generate report
    report_lines = ["Cross-validation Results:"]
    report_lines.append("-" * 50)
    report_lines.append(f"Best Parameters: {grid_search.best_params_}")
    report_lines.append(f"Best CV Score: {grid_search.best_score_:.3f}")
    report_lines.append("\nDetailed Metrics:")
    
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            report_lines.append(
                f"{metric[5:]:20} : {mean_score:.3f} (+/- {std_score * 2:.3f})"
            )
            report_dict[f'{metric[5:]}_mean'] = mean_score
            report_dict[f'{metric[5:]}_std'] = std_score
    
    # Add confusion matrices to report
    # report_lines.append("\nMean Confusion Matrix (Counts):")
    # report_lines.append("-" * 50)
    
    if labelsdict:
        # Create DataFrames for prettier confusion matrix display
        conf_matrix_df = pd.DataFrame(
            mean_conf_matrix,
            index=[f'True {label}' for label in class_names],
            columns=[f'Pred {label}' for label in class_names]
        )
        # report_lines.append("\n" + str(conf_matrix_df))
        
        # Add standard deviations
        # report_lines.append("\nConfusion Matrix Standard Deviations:")
        std_matrix_df = pd.DataFrame(
            std_conf_matrix,
            index=[f'True {label}' for label in class_names],
            columns=[f'Pred {label}' for label in class_names]
        )
        # report_lines.append("\n" + str(std_matrix_df))
        
        # Add normalized confusion matrix
        # report_lines.append("\nNormalized Confusion Matrix (Row Percentages):")
        norm_matrix_df = pd.DataFrame(
            norm_conf_matrix,
            index=[f'True {label}' for label in class_names],
            columns=[f'Pred {label}' for label in class_names]
        ).round(3)
        # report_lines.append("\n" + str(norm_matrix_df))
    else:
        # report_lines.append("\nCounts:\n" + str(mean_conf_matrix))
        # report_lines.append("\nStandard Deviations:\n" + str(std_conf_matrix))
        # report_lines.append("\nNormalized (Row Percentages):\n" + str(norm_conf_matrix.round(3)))
        pass
    
    # Add mean classification report to output
    report_lines.append("\nMean Classification Report across CV Folds:")
    report_lines.append("-" * 50)
    
    # Format the mean classification report
    if class_names:
        headers = ['precision', 'recall', 'f1-score', 'support']
        row_format = "{:15} " + " ".join(["{:>10}" for _ in headers])
        
        report_lines.append(row_format.format("", *headers))
        report_lines.append("-" * 65)
        
        for class_name in class_names:
            metrics = mean_classification_report[class_name]
            values = [
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']:.0f}"
            ]
            report_lines.append(row_format.format(class_name, *values))
            
            # Add standard deviations
            std_metrics = std_classification_report[class_name]
            std_values = [
                f"(±{std_metrics['precision']:.3f})",
                f"(±{std_metrics['recall']:.3f})",
                f"(±{std_metrics['f1-score']:.3f})",
                f"(±{std_metrics['support']:.0f})"
            ]
            report_lines.append(row_format.format("", *std_values))
        
        # Add macro and weighted averages
        for avg in ['macro avg', 'weighted avg']:
            report_lines.append("-" * 65)
            metrics = mean_classification_report[avg]
            values = [
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']:.0f}"
            ]
            report_lines.append(row_format.format(avg, *values))
            
            # Add standard deviations
            std_metrics = std_classification_report[avg]
            std_values = [
                f"(±{std_metrics['precision']:.3f})",
                f"(±{std_metrics['recall']:.3f})",
                f"(±{std_metrics['f1-score']:.3f})",
                f"(±{std_metrics['support']:.0f})"
            ]
            report_lines.append(row_format.format("", *std_values))
    
    return grid_search, report_dict, '\n'.join(report_lines)
