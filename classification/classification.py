

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA, TruncatedSVD
import xgboost as xgb

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import get_environment

from config.config import data_import_pkl, data_export_pkl, data_import_pandas, data_export_pandas

def run():
    ENV = get_environment(
        env_path="../environments",
        env_name="env.json"
    )

    # content_date = datetime.now().date() + timedelta(days=0)
    content_date = ENV['CONTENT_DATE']
    website = ENV['SOURCE']['NAME']
    version = ENV['VERSION']

    grid_search = ENV['CLASSIFICATION']['GRID_SEARCH']
    load_pipeline = ENV['CLASSIFICATION']['LOAD_PIPELINE']
    is_weight = ENV['CLASSIFICATION']['IS_WEIGHT']
    is_pca = ENV['CLASSIFICATION']['IS_PCA']

    ## Classification
    df_embed = data_import_pandas(
        website=website,
        content_date=content_date,
        version=version,
        folder_name='embeddings',
        additional_info='embeddings'
    )
    ### Preprocessing

    # Feature Extraction
    # 3. Load saved embeddings
    X_qual_embed = df_embed['qualification_embedding'].to_list()
    X_qual_embed = np.array(X_qual_embed)

    # 4 TF-IDF for related_experience
    df_embed['related_experience'] = df_embed['related_experience'].fillna('')
    tfidf_exp = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    X_exp = tfidf_exp.fit_transform(df_embed['related_experience'])

    # 5. Numeric features
    ## Encode categorical
    degree_map = {
        '': 0,
        '0': 0,
        'unspecified': 0,

        'high school diploma': 1,
        'mbo': 1,
        'hbo': 1,

        'associate': 2,
        'associates': 2,

        'bachelor': 3,
        'bachelors': 3,
        'ba/bs': 3,

        'master': 4,
        'mba': 4,

        'phd': 5,
        'doctoral': 5,
        'doctor': 5,      # Sometimes appears as "doctor"
        'jd': 5,

        # Domain-specific advanced license → treat as postgraduate level
        'faa airline transport pilot certificate': 5
    }

    df_embed['min_years'] = df_embed['min_years'].fillna(0).replace('', 0).astype(int)
    df_embed['min_degree'] = df_embed['min_degree'].map(degree_map).fillna(0).astype(int)
    df_embed['country_encoded'] = LabelEncoder().fit_transform(df_embed['country'])

    X_numeric = df_embed[['min_years', 'min_degree', 'country_encoded']].fillna(0).values

    le = LabelEncoder()
    df_embed['level_encoded'] = le.fit_transform(df_embed['level'])

    # Target
    y = df_embed['level_encoded']  # e.g., 'Entry', 'Mid', 'Senior'
    y_level_count = df_embed['level'].value_counts().to_dict()
    ### Weight
    if is_weight:
        # Handle Weights due to Imbalance Class
        level_counts = df_embed['level'].value_counts()

        # Compute weights
        total_samples = len(df_embed)
        class_weights = {level: total_samples/count for level, count in level_counts.items()}
        print(class_weights)

        # Map weights to each sample
        sample_weights = df_embed['level'].map(class_weights)

    else:
        pass
    ### Load Saved Pipeline
    if load_pipeline:
        # Load the saved pipeline
        pipeline_objects = data_import_pkl(
            website=website,
            folder_name='classification',
            version=version,
            content_date=content_date,
            additional_info='pipeline-job_level'
        )

        # Extract objects
        model = pipeline_objects['model']
        tfidf_exp = pipeline_objects['tfidf_exp']
        X_qual_embed = pipeline_objects['embeddings_qual']
        X_exp = pipeline_objects['X_exp']
        X_numeric = pipeline_objects['X_numeric']
        y_level_count = pipeline_objects['y_level_count']
        y = pipeline_objects['y_level_encoded']
        le = pipeline_objects['label_encoder_level']
        best_params_ = pipeline_objects['best_params_']

        print("Pipeline and model loaded successfully!")
    else:
        pass
    ### PCA
    if is_pca:
        # Apply PCA to reduce tf-idf dimensionality
        svd_exp = TruncatedSVD(n_components=50, random_state=42)
        X_exp_reduced = svd_exp.fit_transform(X_exp)

        # Apply PCA to reduce embeddings dimensionality
        pca_dim = min(X_qual_embed.shape[0], X_qual_embed.shape[1], 100)  # automatically safe reduce 1536-dim embedding to 100 if sample less then min
        pca_qual = PCA(n_components=pca_dim, random_state=42)
        pca_qual.fit(X_qual_embed)  # fit on source embeddings

        X_qual_embed_reduced = pca_qual.transform(X_qual_embed)

    else:
        X_exp_reduced = X_exp
        X_qual_embed_reduced = X_qual_embed
    ### Sparse and Split Train Test
    # Convert Embeddings and Numeric to Compressed Sparse Row due to TF-IDF
    # Convert to sparse to stack with TF-IDF
    X_qual_embed_sparse = csr_matrix(X_qual_embed_reduced)

    # Convert numeric to sparse to stack with TF-IDF
    X_numeric_sparse = csr_matrix(X_numeric)

    # Combine features
    X = hstack([X_exp_reduced, X_qual_embed_sparse, X_numeric_sparse])

    # Split Train Test (optional weight)
    if is_weight:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    ### Grid Search to get the best hyperparameter

    if grid_search and not load_pipeline:
        param_grid = {
            'max_depth': [4,6,8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=42,
            eval_metric='mlogloss'
        )

        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=5,
            verbose=3,
            n_jobs=-2
        )

        if is_weight:
            grid_search.fit(X_train, y_train, sample_weight=w_train)
        else:
            grid_search.fit(X_train, y_train)
        print("Best params:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        best_params_ = grid_search.best_params_

        y_pred = best_model.predict(X_test)

    ### Direct Train XGBoost
    else:
        if load_pipeline:
            model = xgb.XGBClassifier(
                n_estimators=best_params_['n_estimators'],
                max_depth=best_params_['max_depth'],
                learning_rate=best_params_['learning_rate'],
                subsample=best_params_['subsample'],
                colsample_bytree=best_params_['colsample_bytree'],
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=42
            )

        else:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=42
            )
            best_params_ = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }

        if is_weight:
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    ### Evaluate Performance
    print("Accuracy on test set:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    df_report = pd.DataFrame(report).transpose().apply(lambda value: round(value, 2))

    additional_info = 'performance-job_level'
    if is_pca:
        additional_info=additional_info+'-pca'
    if is_weight:
        additional_info=additional_info+'-weight'
    if grid_search:
        additional_info=additional_info+'-grid_search'

    data_export_pandas(
        df_output=df_report,
        website=website,
        content_date=content_date,
        version=version,
        folder_name='classification',
        additional_info=additional_info,
    )

    ### Feed Complete Data
    # Predict Overall Data
    y_pred_all = best_model.predict(X)
    df_embed['predicted_level_encoded'] = y_pred_all
    df_embed['predicted_level'] = le.inverse_transform(y_pred_all)
    data_export_pandas(
        df_output=df_embed,
        website=website,
        content_date=content_date,
        version=version,
        folder_name='classification',
        additional_info='classification',
        incl_excel=True
    )

    ### Export pipeline
    # Save pipeline + features
    pipeline_objects = {
        'model': model,
        'tfidf_exp': tfidf_exp,
        'embeddings_qual': X_qual_embed,    # embeddings
        'X_exp': X_exp,                     # TF-IDF
        'X_numeric': X_numeric,             # numeric features
        'y_level_count': y_level_count,
        'y_level_encoded': y,
        'label_encoder_level': LabelEncoder().fit(df_embed['level']),  # for decoding
        'best_params_': best_params_
    }

    data_export_pkl(
        pipeline_objects=pipeline_objects,
            website=website,
            folder_name='classification',
            version=version,
            content_date=content_date,
            additional_info='pipeline-job_level'
        )
    print("All features, embeddings, preprocessing, and model saved successfully!")

    return pipeline_objects