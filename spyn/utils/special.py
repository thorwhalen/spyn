class DictDefaultDict(dict):
    """
    Acts similarly to collections.defaultdict, except
        (1) the defaults depend on the key (given by a dict of key-->default_val at construction)
        (2) it is not a function that is called to create the default value (so careful with referenced variables)
    """

    def __init__(self, default_dict):
        super(DictDefaultDict, self).__init__()
        self.default_dict = default_dict

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return self.default_dict[item]
