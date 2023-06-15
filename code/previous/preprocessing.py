# 1. Variance Threshold : Pour enlever les colonnes ayant des variances trop faibles.
# Variance Threshold
 
thresholder = VarianceThreshold()
X_high_variance = thresholder.fit_transform(X)
X_high_variance = pd.DataFrame(X_high_variance)
X_high_variance.columns = X.columns
X_high_variance.head()


# 2. Traitement des paires très corrélées : Pour choisir entre deux colonnes laquelle garder si elles sont très corrélées entre elles. 
# On garde celle avec le plus de variance.
# Python

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
def drop_top_corr(correl):
    drop_corr = []
    for i in range(len(correl)):
        col = correl.index[i]
        if "Build seconds" not in col:
            if merge_var[col[0]] > merge_var[col[1]]:
                drop_corr.append(col[1])
            else:
                drop_corr.append(col[0])
    return list(set(drop_corr))
correl = get_top_abs_correlations(merge_sign_stat.drop(columns=["Name"]), 391) # High correlation for Pearson: > 0.5
drop_corr = drop_top_corr(correl)
merge_sign_stat_var_corr = merge_sign_stat.drop(columns=drop_corr)


# 3. Traitement des valeurs aberrantes
# 4. Standardisation avec StandardScaler :

scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data_clean[data_clean.columns[1:]]))
scaled_data.columns = data_clean.columns[1:]
scaled_data.describe()

# 5. Sélection des attributs
# A. PCA :

pca = PCA(n_components=0.95)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
x_pca = pd.DataFrame(x_pca)
x_pca.columns = ['PCA'+str(i) for i in range(x_pca.shape[1])]
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
plt.figure(figsize=(10, 10))
plt.bar(x= range(1,11), height=percent_variance, tick_label=x_pca.columns)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.show()

# Importance de chaque variable dans le PCA :

loadings = pd.DataFrame(pca.components_, 
                        columns=scaled_data.dropna(axis=1).columns)
maxPC = 1.01 * np.max(np.max(np.abs(loadings)))
f, axes = plt.subplots(10, 1, figsize=(15, 15), sharex=True)
for i, ax in enumerate(axes):
    pc_loadings = loadings.loc[i, :]
    colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
    ax.axhline(color='#888888')
    pc_loadings.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'PC{i+1}')
    ax.set_ylim(-maxPC, maxPC)
plt.tight_layout()
plt.show()


#  B. SelectKBest :
selector = SelectKBest(f_regression, k=10)
z = selector.fit_transform(X, y)
dfscores = pd.DataFrame(z.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores], axis=1)
featureScores.columns = ['Specs','Score']
plt.figure(figsize=(12, 12))
plt.bar(featureScores.nlargest(11, 'Score')["Specs"], featureScores.nlargest(11, 'Score')["Score"])
plt.xticks(rotation=45);

#  C. RFE :
rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=10)
selector = rfe.fit(X, y)
filter = selector.support_
ranking = selector.ranking_
features = np.array(data_clean.drop(columns=["Build seconds"]).columns)
dset = pd.DataFrame()
dset['attr'] = features[filter]
dset['importance'] = rfe.estimator_.feature_importances_
dset = dset.sort_values(by='importance', ascending=False)
plt.figure(figsize=(12, 10))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFE - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()
