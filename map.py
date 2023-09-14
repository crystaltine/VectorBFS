class Map:
    def __init__(self, key_class: type, value_class: type):
        self.key_class = key_class
        self.value_class = value_class
        self.kv_dict: dict[key_class, value_class] = dict()
        self.vk_dict: dict[value_class, key_class] = dict()
        
    def add_pair(self, key, value):
        self.kv_dict[key] = value
        self.vk_dict[value] = key
    
    def remove_pair(self, key, value):
        self.kv_dict.pop(key)
        self.vk_dict.pop(value)
        
    def __getitem__(self, query):
        if type(query) == self.key_class:
            try: return self.kv_dict[query]
            except: raise KeyError(f"KeyError: {query} of type {self.key_class} not found in keys of map<{self.key_class}, {self.value_class}>")
        if type(query) == self.value_class:
            try: return self.vk_dict[query]
            except: raise KeyError(f"KeyError: {query} of type {self.value_class} not found in values of map<{self.key_class}, {self.value_class}>")
        raise TypeError("Query must be of type " + str(self.key_class) + " or " + str(self.value_class))
    
    def __setitem__(self, item, mapped_value):
        if type(item) == self.key_class:
            self.kv_dict[item] = mapped_value
            self.vk_dict[mapped_value] = item
        elif type(item) == self.value_class:
            self.vk_dict[item] = mapped_value
            self.kv_dict[mapped_value] = item
        else:
            raise TypeError("item must be of type " + str(self.key_class) + " or " + str(self.value_class))
                
    def __sizeof__(self) -> int:
        return len(self.kv_dict)
