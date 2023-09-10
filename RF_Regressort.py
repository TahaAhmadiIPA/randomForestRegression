from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pandas import read_csv

# regressor class
class regressor:
    def __init__(self, adress):
        self.__x_data = read_csv(adress).drop(columns=["PremiumPrice"])
        self.__y_data = read_csv(adress)["PremiumPrice"]
    # train and test split
    def train_test(self, test_percent, rnd_state):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__x_data, self.__y_data , random_state=rnd_state, test_size=test_percent, shuffle=True)
    def train(self):
        self.__reg = RandomForestRegressor()
        self.__reg.fit(self.__x_train, self.__y_train)
    # prediction and R2 score
    def predict(self):
        self.__pred = self.__reg.predict(self.__x_test)
        self.__r2 = r2_score(self.__y_test, self.__pred)
        print(self.__r2)

medic = regressor('c:/Users/Taha Ahmadi/Desktop/medic.csv')
medic.train_test(0.25, 4)
medic.train()
medic.predict()

