from data import Data
# import vis
from model import Model
from time import time


if __name__ == '__main__':
    crimes = Data()
    # vis.plot_map_contour(crimes.data, 'Month')
    # vis.plot_bar(crimes.data, 'Hour')
    # vis.plot_heatmap(crimes.data, 'Location Description')
    # vis.plot_bar(crimes.data, 'Month')
    # vis.plot_heatmap(crimes.data, 'Month')
    # vis.biplot(crimes.data, 'Month')
    # vis.biplot(crimes.data, 'Hour')
    # vis.biplot(crimes.data, 'Community Area', crimes.community_name)
    # vis.plot_time(crimes.data)
    crimes.random_sample(0.2)
    crimes.preprocessing()

    # param_logit_grid = {'logit_grid':
    #                         {'penalty': ['l1', 'l2'],
    #                          'C': [10 ** i for i in range(-3, 1, 1)]}}
    # logit = Model(crimes, 'logit', param_logit_grid)
    # logit.grid_search_all('accuracy', 3)
    # logit.run_all('accuracy', 3)


    # param_logit = {'logit': {'penalty': 'l2', 'C': 1}}
    # logit = Model(crimes, 'logit', param_logit)
    # start = time()
    # logit.run_all()
    # end = time()
    # print 'Time used: {}.'.format((end-start)/60)

    # start = time()
    # param_xgb = {'xgb_grid': {'learning_rate': [0.1],
    #                      'n_estimators': [650],
    #     		 'gamma': [1],
    #     		 'max_depth': [4],
    #     		 'subsample':[1]}}
    # xgb = Model(crimes, 'xgb', param_xgb)
    # res = xgb.grid_search_all('neg_log_loss', 3)
    # print res
    # res = xgb.grid_search_all('accuracy', 3)
    # print res
    # xgb.run_all()
    # end = time()
    start = time()
    param_xgb = {'xgb': {'learning_rate': 0.1,
                         'n_estimators': 650,
        		 'gamma': 1,
        		 'max_depth': 4,
        		 'subsample': 1}}
    xgb = Model(crimes, 'xgb', param_xgb)
    xgb.run_all()
    end = time()
    print 'Time used: {}.'.format((end-start)/60)


