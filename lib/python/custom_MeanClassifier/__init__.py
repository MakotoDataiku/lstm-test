from custom_MeanClassifier.MeanClassifier import MeanClassifier
def __init__(self, arg1, arg2, arg3, ..., argN):

    # print("Initializing classifier:\n")

    args, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop("self")

    for arg, val in values.items():
        setattr(self, arg, val)
        # print("{} = {}".format(arg,val)