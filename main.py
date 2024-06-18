import pandas as pd  # manipulation des données
from sklearn.model_selection import train_test_split  # diviser le dataset en train, validation et test sets
import numpy as np
import fuzzywuzzy.process  # afin de trouver les correspendances
from sklearn.preprocessing import LabelEncoder  # coder les attributs categoriques
from sklearn.model_selection import cross_val_score, KFold
import catboost
from catboost import CatBoostClassifier
import optuna
# importation du scaler et des indicateurs de precision
from sklearn import metrics
# importation de la bibliotheque du sauvegardement du modele
import pickle
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score

# Creation du DataFrame
df = pd.read_csv("Churn_data.csv")

# suppression des collonnes sans noms ( de 14 à 20)
columns_to_drop = df.columns[14:21]
df = df.drop(columns=columns_to_drop)

# ***********************************************************************
# Comme le SubsId n'est pas important, nous allons le supprimer
df.drop('Subs_id', axis=1, inplace=True)
# Renommer les colonnes : tout en miniscule !
df.columns = df.columns.str.lower()
# ***********************************************************************
# Vérifier combien de valeurs uniques y'en a pour chaque attribut
'''
for attribut in df.columns:
    print("\nLe nombre de valeurs uniques de", attribut, ":")
    print(df[attribut].nunique())
'''
# ***********************************************************************
# Vérifiez combien de valeurs nulles il y a pour chaque attribut
'''
print("-----------------------------------------------------------------------------")
print("Le nombre de valeurs nulles par attribut : ")
print(df.isna().sum())
print("-----------------------------------------------------------------------------")
print("Le pourcentage de lignes où la valeur devicetype est nulle est de :")
print(df["devicetype"].isnull().sum() / df.shape[0] * 100, "%")
print("-----------------------------------------------------------------------------")
print("Le pourcentage de lignes où la valeur value_segment est nulle est de :")
print(df["value_segment"].isnull().sum() / df.shape[0] * 100, "%")
'''
# **************************************************************************************
# Vérifions la valeur la plus fréquente du devicetype
'''
print("-----------------------------------------------------------------------------")
print("le pourcentage de la valeur", df["devicetype"].describe()['top'],
      "la plus fréquente de l'attribut devicetype est de : ")
print((df['devicetype'].describe()['freq'] / df['devicetype'].describe()['count']) * 100, "%")
# On ne peut pas le remplir avec la valeur la plus frequente car elle n'est pas tres recurrente (67.91%)
'''
# **************************************************************************************
# Remplir les cases vides du devicetype avec "UNKNOWN"
df['devicetype'] = df['devicetype'].fillna("UNKNOWN")

# ***********************************************************************************
""""
Passons maintenant au value_segment :
nous allons remplir les valeurs manquantes en utilisant behavior_segments : behavior_segments est une description du value_segment 
On doit d'abord afficher la valeur du behavior_segments des clients qui ont une value_segment nulle.
print(df[df["value_segment"].isnull()]["behavior_segments"])
Les clients qui ont un (value_segment=NaN) ont un (behavior_segment=New Customer)
"""
df['value_segment'] = df['value_segment'].fillna("NEW")
# Nous pouvons supprimer la colonne du behavior_segment car c'est une description du Value_Segment
df.drop(columns=['behavior_segments'], inplace=True)
# **********************************************************************************
# Reglons maintenant le problème des wilayas
# Mettre les wilayas en majuscules
df['wilaya'] = df['wilaya'].str.upper()

def attribut_matches(dataFrame, attribute, percentage):
    matches_dict = {}
    # parcourir les valeurs d'un attribut donné
    for value in dataFrame[attribute].unique():
        matches = fuzzywuzzy.process.extract(value, dataFrame[attribute], limit=10,
                                             scorer=fuzzywuzzy.fuzz.token_sort_ratio)

        # Enlever la valeur elle mm
        matches = [match for match in matches if match[0] != value]
        # laisser que les matches qui ne sont pas égaux et ont une inferieur au pourcentage
        non_egal_matches = [match for match in matches if match[1] <= 100 and match[1] >= percentage]
        matches_dict[value] = non_egal_matches
        sorted_matches_dict = dict(sorted(matches_dict.items()))

    return sorted_matches_dict


def print_matches(matches_dict):
    for key, matches in matches_dict.items():
        if matches:
            print(f"matches for '{key}':")
            for match in matches:
                print(f"  - {match[0]}")

'''
w = attribut_matches(df, 'wilaya', 1)
print_matches(w)
'''
# ------------------------------------------------------------------------------
# Traitement des valeurs incohérentes des Wilaya's
df.loc[df["wilaya"] == "BORDJ-BOU-ARRERID", "wilaya"] = "BORDJ-BOU-ARRERIDJ"
df.loc[df["wilaya"] == "EL-MIANIAA", "wilaya"] = "EL-MENIAA"
df.loc[df["wilaya"] == "EL-M'-GHAIER", "wilaya"] = "EL-MGHAIER"
df.loc[df["wilaya"] == "BORDJ-BADJI-MOKHTA", "wilaya"] = "BORDJ-BADJI-MOKHTAR"
df.loc[df["wilaya"] == "EL-TAREF", "wilaya"] = "EL-TARF"
# ------------------------------------------------------------------------------
# codant les Wilaya's
wilayas_code = {
    "ADRAR": 1,
    "CHLEF": 2,
    "LAGHOUAT": 3,
    "OUM-EL-BOUAGHI": 4,
    "BATNA": 5,
    "BEJAIA": 6,
    "BISKRA": 7,
    "BECHAR": 8,
    "BLIDA": 9,
    "BOUIRA": 10,
    "TAMANRASSET": 11,
    "TEBESSA": 12,
    "TLEMCEN": 13,
    "TIARET": 14,
    "TIZI-OUZOU": 15,
    "ALGER": 16,
    "DJELFA": 17,
    "JIJEL": 18,
    "SETIF": 19,
    "SAIDA": 20,
    "SKIKDA": 21,
    "SIDI-BELABBES": 22,
    "ANNABA": 23,
    "GUELMA": 24,
    "CONSTANTINE": 25,
    "MEDEA": 26,
    "MOSTAGANEM": 27,
    "MSILA": 28,
    "MASCARA": 29,
    "OUARGLA": 30,
    "ORAN": 31,
    "EL-BAYADH": 32,
    "ILLIZI": 33,
    "BORDJ-BOU-ARRERIDJ": 34,
    "BOUMERDES": 35,
    "EL-TARF": 36,
    "TINDOUF": 37,
    "TISSEMSILT": 38,
    "EL-OUED": 39,
    "KHENCHELA": 40,
    "SOUK-AHRAS": 41,
    "TIPAZA": 42,
    "MILA": 43,
    "AIN-DEFLA": 44,
    "NAAMA": 45,
    "AIN-TEMOUCHENT": 46,
    "GHARDAIA": 47,
    "RELIZANE": 48,
    "TIMIMOUN": 49,
    "BORDJ-BADJI-MOKHTAR": 50,
    "OULED-DJELLAL": 51,
    "BENI-ABBES": 52,
    "IN-SALAH": 53,
    "IN-GUEZZAM": 54,
    "TOUGGOURT": 55,
    "DJANET": 56,
    "EL-MGHAIER": 57,
    "EL-MENIAA": 58,
}
df["wilaya"] = df["wilaya"].map(wilayas_code)
'''
# ------------------------------------------------------------------------------
# Coder le line_type
df['line_type'] = LabelEncoder().fit_transform(df['line_type'])
# ------------------------------------------------------------------------------
# Coder le devicetype
df['devicetype'] = LabelEncoder().fit_transform(df['devicetype'])
# ------------------------------------------------------------------------------
# Coder value segment
df['value_segment'] = LabelEncoder().fit_transform(df['value_segment'])
# ------------------------------------------------------------------------------
# Coder le sex
df = pd.get_dummies(df, columns=['sex'], drop_first=True)
df['sex'] = df['sex_Male'].replace({True: 1, False: 0})
df.drop(['sex_Male'], axis=1, inplace=True)
'''


# Fonction pour remplacer les noms detailles des offres par leur modes de payement (indiqués dans le nom)
def replace_detailed_offer_name_containing_offer_indicator(df, column_name, offer_indicator, offer):
    value_to_replace = df[column_name].str.contains(offer_indicator, case=False)
    df.loc[value_to_replace, column_name] = offer
    return df


# remplacer chaque offre contenant "PRE" par la valeur "PREPAYED"
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "pre", "PREPAYED")
# remplacer chaque offre contenant "POST" par la valeur "POSTPAYED"
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "post", "POSTPAYED")
# remplacer chaque offre contenant "HYB" par la valeur "HYBRID"
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "hyb", "HYBRID")
# remplacer l'offre "Izzy" par la valeur "PREPAYED" (Offre prepayee)
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "izzy", "PREPAYED")
# remplacer l'offre "Djezzy ZID" par la valeur "PREPAYED" (Offre prepayee)
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "Djezzy ZID", "PREPAYED")
# remplacer l'offre "Smart Control" par la valeur "POSTPAYED" (Offre postpayee)
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "Smart Control", "POSTPAYED")
# remplacer l'offre "HAYLA" par la valeur "PREPAYED" (Offre postpayee)
df = replace_detailed_offer_name_containing_offer_indicator(df, "global_profile", "hayla", "PREPAYED")
# ------------------------------------------------------------------------------
# Coder global profile
#df['global_profile'] = LabelEncoder().fit_transform(df['global_profile'])


# ------------------------------------------------------------------------------
def label_age(age):
    if 14 <= age <= 19:
        return "Ado"
    elif 20 <= age <= 59:
        return "Adult"
    else:
        return "Senior"


df["age"] = df["age_years"].apply(label_age)
df.drop(['age_years'], axis=1, inplace=True)
# Coder age_labeled
#df['age'] = LabelEncoder().fit_transform(df['age'])
# ---------------------------------------------------------------------------------
#Scal data numérique

df['yr'] = (df['yr'] - df['yr'].min()) / (df['yr'].max() - df['yr'].min())
df['mr'] = (df['mr'] - df['mr'].min()) / (df['mr'].max() - df['mr'].min())
df['nb_supended'] = (df['nb_supended'] - df['nb_supended'].min()) / (df['nb_supended'].max() - df['nb_supended'].min())
df['number_subscription'] = (df['number_subscription'] - df['number_subscription'].min()) / (df[
    'number_subscription'].max() - df['number_subscription'].min())

desired_order = ['age', 'sex', 'wilaya', 'devicetype', 'line_type', 'global_profile', 'value_segment',
                 'number_subscription', 'nb_supended', 'yr', 'mr', 'churn']

df = df[desired_order]
# -----------------------------------------------------------

'''
 #afficher les produits de correlations entre les features et la colonne de churn
def show_correlations(dataframe, show_chart = True):
    fig = plt.figure(figsize = (20,10))
    corr = dataframe.corr()
    if show_chart == True:
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values,
                    annot=True)
    return corr

correlation_df = show_correlations(df2,show_chart=True)


#Creer la courbe ROC pour tous les modeles
plt.figure(figsize=(10, 8))
for model_name, predictions in zip(model_names, list_of_predictions):
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=model_name)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.show()
'''

# ************************************************************************

X = df.drop(columns=['churn'])  # Features excluding the 'churn' column
y = df['churn'].values
categorical_features_indices = np.where(X.dtypes != float)[0]

# Décomposer le dataframe en train et test
X_train, X_testval, y_train, y_testval = train_test_split(X, df['churn'], test_size=0.2, random_state=1, stratify=df["churn"])
X_test, X_val, y_test, y_val= train_test_split(X_testval, y_testval, test_size=0.2, random_state=1, stratify=y_testval)


# -------------------------------------------------------------------

# Modele avec hyperparametres optimales
catboost_5 = CatBoostClassifier(

    loss_function='Logloss',
    verbose=False,
    colsample_bylevel=0.046579247507969435,
    depth=6,
    boosting_type= 'Ordered',
    bootstrap_type= 'Bayesian',
    scale_pos_weight=24.915879854130356,
    random_seed=7,
    thread_count=11,
    iterations= 703,
    learning_rate= 0.014365614180946378,
    l2_leaf_reg=2.789239829542535,
    bagging_temperature= 3.6401622856311304,
)

catboost_5.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_test, y_test))
pickle.dump(catboost_5, open("churnModel.pkl", 'wb'))
y_pred = catboost_5.predict(X_test)

print("accuracy : ", metrics.accuracy_score(y_test, y_pred) * 100, "%\n")
print("f1 : ", metrics.f1_score(y_test, y_pred), "\n")
print("recall : ", metrics.recall_score(y_test, y_pred), "\n")
print("matrice de confusion \n", pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

'''
# CV : cross validation
cv_strategy = KFold(n_splits=20)
custom_recall_scorer = metrics.make_scorer(metrics.recall_score, greater_is_better=True)
custom_accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)
custom_f1_scorer = metrics.make_scorer(metrics.f1_score)

# Perform cross-validation
scores = cross_val_score(catboost_5, X, y,
                         cv=cv_strategy,
                         scoring=custom_recall_scorer)

print("recall scores:\n", scores)

mean = sum(scores) / len(scores)
print("Mean recall score:", mean)

scores = cross_val_score(catboost_5, X_train, y_train,
                         cv=cv_strategy,
                         scoring=custom_accuracy_scorer)

print("accuracy scores:\n", scores)

mean = sum(scores) / len(scores)
print("Mean accuracy score:", mean)

scores = cross_val_score(catboost_5, X, y,
                         cv=cv_strategy,
                         scoring=custom_f1_scorer)

print("f1 scores:\n", scores)

mean = sum(scores) / len(scores)
print("Mean f1 score:", mean)

# save Model
# pickle.dump(catboost_5, open("churnModel.pkl", 'wb'))

'''
'''
def objective(trial):
    param = {
        "objective": "Logloss",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 25.0),
        "random_seed": trial.suggest_int("random_seed", 0, 100),
        "thread_count": trial.suggest_int("thread_count", 1, 16),
        "iterations": trial.suggest_int("iterations", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "eval_metric": "Recall",  # Change the eval_metric to Recall
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cat_cls = CatBoostClassifier(verbose=False, **param)
    cat_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], cat_features=categorical_features_indices, verbose=0,
                early_stopping_rounds=100)

    preds = cat_cls.predict(X_test)
    recall = recall_score(y_test, preds)  # Calculate recall instead of accuracy

    return recall


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # You can adjust the number of trials

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Train the final model with the best hyperparameters
    best_cat_cls = CatBoostClassifier(verbose=False, **best_params)
    best_cat_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], cat_features=categorical_features_indices)

    # Evaluate the final model
    final_preds = best_cat_cls.predict(X_test)
    final_recall = recall_score(y_test, final_preds)
    print("Final Model Recall:", final_recall)
'''