import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CountryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, n_top_countries=10):
        self.n_top_countries = n_top_countries

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        countries = X.iloc[:, 0]

        self.top_countries_ = (
            countries
            .value_counts()
            .index[:self.n_top_countries]
            .tolist()
        )

        return self

    def transform(self, X):
        X = pd.DataFrame(X)

        countries = X.iloc[:, 0]

        grouped_countries = countries.apply(
            lambda country:
            country if country in self.top_countries_
            else "Other"
        )

        return grouped_countries.to_numpy().reshape(-1, 1)