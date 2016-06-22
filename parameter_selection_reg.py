def parameter_selection_reg(x_all, y_all, methods = 'Regression',alpha_min = -3, alpha_max = 5, steps = 0.01, max_poly = 2):

    from sklearn import grid_search
    from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
    from sklearn import cross_validation
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn import cross_validation
    from sklearn import linear_model



    x_all.shape
    selector = SelectPercentile(f_regression)
    selector.fit_transform(x_all, y_all)
    score = -np.log10(selector.pvalues_)
    all_score = np.random.rand(2, len(score))
    all_score[0][:] = -np.log10(selector.pvalues_)
    all_score[1][:] = np.arange(0, len(score), 1)
    all_score_trans = all_score.transpose()
    bestpoly = 0
    bestalpha = 0.0
    alpha_power_list = np.arange(alpha_min, alpha_max, steps)
    alpha_select_list = [math.exp(alpha_select_list) for alpha_select_list in alpha_power_list]
    score_record = pd.DataFrame(columns=['feature_num', 'poly', 'penalty_parameter', 'e_out', 'e_in'])
    for featuren in range(24, len(x_all.columns) + 1):
        # n feature want to keep
        # featuren=13
        full_data_processed_new = SelectKBest(f_regression, k=featuren).fit_transform(x_all, y_all)

        # Regression Model with Ridge Regularization
        # 10-fold Cross Validation Involved
        # selection of Polynomial
        # selection of regularization alpha


        for k in range(2, max_poly + 1):
            if k == 1:
                full_data_processed_poly = full_data_processed_new
            else:
                poly = preprocessing.PolynomialFeatures(k)
                full_data_processed_poly = poly.fit_transform(full_data_processed_new)
            kf = cross_validation.KFold(len(full_data_processed_poly), n_folds=10)
            for alpha_select in alpha_select_list:
                E_in = 0.0
                E_out = 0.0
                for train, test in kf:
                    X_train, X_test, Y_train, Y_test = full_data_processed_poly[train], full_data_processed_poly[

                        test], \
                                                       y_all.iloc[train], y_all.iloc[test]
                    # Fitting use Regression model
                    if methods == 'ridge':
                        clf = linear_model.Ridge(alpha=alpha_select, max_iter=1000)
                    elif methods == 'lasso':
                        clf = linear_model.Lasso(alpha=alpha_select, max_iter=1000)
                    elif methods == 'logistic':
                        clf = linear_model.LogisticRegression(C=alpha_select, max_iter=1000)
                    clf = linear_model.Ridge(alpha=alpha_select, max_iter=1000)
                    clf.fit(X_train, Y_train)
                    if methods == 'logistic':
                        E_in += accuracy_score(Y_train, clf.predict(X_train))
                        E_out += accuracy_score(Y_test, clf.predict(X_test))
                    else:
                        E_in += mean_squared_error(Y_train, clf.predict(X_train))
                        E_out += mean_squared_error(Y_test, clf.predict(X_test))
                score_record = score_record.append(
                    pd.DataFrame([[featuren, k, alpha_select, E_out / 10.0, E_in / 10.0]],
                                 columns=['feature_num', 'poly', 'penalty_parameter', 'e_out', 'e_in']))

    print('finished modeling')
    return score_record
