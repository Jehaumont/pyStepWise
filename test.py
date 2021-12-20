import pandas as pd
from sklearn.datasets import load_diabetes
from stepwiseregression import stepwiseregression as SWR

X, Y = load_diabetes(as_frame=True, return_X_y=True)
Y = pd.DataFrame(Y)

swr = SWR("forward", "BIC")
swr.fit(X, Y)
print(swr.model.summary())

swr = SWR("backward", "BIC")
swr.fit(X, Y)
print(swr.model.summary())

swr = SWR("mixed", "BIC")
swr.fit(X, Y)
print(swr.model.summary())

swr = SWR("mixed", "AIC")
swr.fit(X, Y)
print(swr.model.summary())

swr = SWR("mixed", "pval", threshold=0.05)
swr.fit(X, Y)
print(swr.model.summary())

