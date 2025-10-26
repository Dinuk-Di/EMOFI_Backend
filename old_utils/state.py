import pickle

class AppState:
    def __init__(self):
        self.executed = False
        self.executedApp = False
        self.recommendations = []
        self.recommendation_outputs = []
        self.selectedApp = ""
        self.selectedRecommendation = ""
        self.averageEmotion = ""

    def reset(self):
        self.__init__() 

app_state = AppState()

def pickle_save():
    with open("state.pkl", "wb") as f:
        pickle.dump(app_state, f)

def pickle_load():
    with open("state.pkl", "rb") as f:
        app_state = pickle.load(f)
        return app_state
