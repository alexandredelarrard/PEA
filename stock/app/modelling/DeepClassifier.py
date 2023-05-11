import numpy as np 
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def create_data(text, embeddings, cluster_name="CLUSTER"):
    non_zero = text.loc[text[cluster_name] != -1].index
    y = text.loc[non_zero, cluster_name].astype(str)
    y = y.reset_index()
    X = embeddings[non_zero]
    return X, y


def get_model(input_dim, output_dm):

    model = Sequential()
    model.add(Dense(196, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dm, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                optimizer='rmsprop', 
                metrics=['accuracy'])
    return model 


def train_k_fold(text, X, y, progress_function, cluster_name="CLUSTER"):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    output= pd.DataFrame()
    models = []
    percentage_threshold = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y[cluster_name])):

        sentences_train, sentences_test = X[train_index], X[test_index]
        y_train, y_test = pd.get_dummies(y).loc[train_index], pd.get_dummies(y).loc[test_index]

        model = get_model(sentences_train.shape[1], len(text[cluster_name].unique()) -1)

        history = model.fit(sentences_train, y_train.drop(["index"], axis=1),
                            epochs=300,
                            verbose=False,
                            validation_data=(sentences_test, y_test.drop(["index"], axis=1)),
                            batch_size=128)
        
        v = model.predict_proba(sentences_test)
        models.append(model)

        a = sum( v.argmax(axis=1) == np.array(y_test.drop(["index"], axis=1)).argmax(axis=1))
        progress_function(f"[FOLD {i}] ACCURACY = {a/y_test.shape[0]:.3f} %", 80)

        percentage_threshold.append(a/y_test.shape[0])

        y_test["PROBA"] = v.max(axis=1)
        y_test["LABEL"] = v.argmax(axis=1)
        output = pd.concat([output, y_test], axis=0)

        output = output.sort_values("index")
        output.index = output["index"]

    return output, models


def reallocate_cluster(output, text, min_percent = 0.01):

    # if label already there, drop it 
    if "LABEL" in text.columns:
        del text["LABEL"]

    # minimum cluster size to keep in classification : 1% total comments
    min_size = int(min_percent*output.shape[0])

    a = output.loc[output["PROBA"] >= 0.99]
    volume_labels = a["LABEL"].value_counts().loc[a["LABEL"].value_counts() < min_size].index
    a["LABEL"] = np.where(a["LABEL"].isin(volume_labels), -1, a["LABEL"])
    text = pd.merge(text, a["LABEL"], left_index=True, right_index=True, validate="1:1", how="left") 
    text["LABEL"].fillna(-1, inplace=True)

    return text


def group_per_seed_words(text):

    a = text.loc[text["INDEX"] == -1]

    if a.shape[0] > 0:
        clusters_id = text["LABEL"].unique()

        agg1 = a[["TARGET", "LABEL"]].groupby("TARGET").aggregate({"LABEL" : lambda x : list(x.value_counts().tolist())})
        agg1= agg1.explode("LABEL").reset_index()
        agg2 = a[["TARGET", "LABEL"]].groupby("TARGET").aggregate({"LABEL" : lambda x : list(x.value_counts().index)})
        agg2 = agg2.explode("LABEL").reset_index()
        agg1["LABEL_ID"] = agg2["LABEL"]
        agg1 = agg1.sort_values(["LABEL_ID", "LABEL"], ascending = [0,0])
        agg1 = agg1.drop_duplicates("LABEL_ID")
        group_labels = agg1[["LABEL_ID", "TARGET"]].groupby("TARGET").aggregate({"LABEL_ID" : list})
        group_labels["LABEL_REF"] = group_labels["LABEL_ID"].apply(lambda x: max(x))
        group_labels = group_labels.explode("LABEL_ID").reset_index(drop=True)
        group_labels = group_labels.loc[group_labels["LABEL_ID"] != -1]

        # TODO: check if should remove LABEL ID less than 20% of seed words category ? 

        group_cluster_id = group_labels["LABEL_ID"].unique()
        for id_cl in list(set(clusters_id) - set(group_cluster_id)):
            group_labels = group_labels.append({"LABEL_ID" : id_cl, "LABEL_REF" : id_cl}, ignore_index=True)
        
        #seed words clusters 
        mapping_cluster = group_labels[["LABEL_ID", "LABEL_REF"]].set_index("LABEL_ID").to_dict()["LABEL_REF"]
        text["FINAL_CLUSTER"] = text["LABEL"].map(mapping_cluster)
        
    else:
        print("No seed words found skip manual gathering")
        text["FINAL_CLUSTER"] = text["LABEL"]

    return text


def classification_fine_tune(text, embeddings, cluster_method, progress_function):
    """
    Classification of cluster ID from CAH. 
    process is the following one:
    - We train a classifier 

    TODO: Sigmoid instead of softmax to allow for multi label 
    classification

    Args:
        text ([type]): [description]
        embeddings ([type]): [description]

    Returns:
        [type]: [description]
    """

    text = text.reset_index(drop=True)

    # first pass to remove outliers based on proba threshold 
    X, y = create_data(text, embeddings, cluster_method)
    output, models = train_k_fold(text, X, y, progress_function, cluster_method)
    text = reallocate_cluster(output, text)

    #second pass to predict cluster 
    X, y = create_data(text, embeddings, cluster_name="LABEL")
    output, models = train_k_fold(text, X, y, progress_function, cluster_name="LABEL")

    # average of 5 folds as prediction 
    for i, m in enumerate(models):
        if i == 0:
            preds = m.predict_proba(embeddings)
        else:
            preds = preds + m.predict_proba(embeddings)
    preds = preds/5

    text["PROBA"] = preds.max(axis=1)
    text["LABEL"] = preds.argmax(axis=1)
    text["LABEL"] = np.where((text["PROBA"] < 0.99)&(text["INDEX"] != -1), -1, text["LABEL"])
    
    # group based on seed words 
    text = group_per_seed_words(text)

    size = (sum(text["LABEL"] != -1)*100 / text.shape[0])
    progress_function(f"Proportion of clustered comments :  {size:.2f}", 85)
    time.sleep(5)

    return text, models
