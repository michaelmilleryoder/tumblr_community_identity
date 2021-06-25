""" Utility functions """
import pickle

def load_pickle(path):
    """ Load preprocessed blog descriptions, such as
        ./output/new_id2token_114k.pkl
    """
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def string_list2str(string_list):
    """ Convert a list in string form, like '[one, two]', to a space-separated
        string of the items. For exampel '[one, two]' -> 'one two'
    """
    if isinstance(string_list, float):
        return ''
    return ' '.join(string_list[1:-1].split(', '))

def strlist2underscore(string_list):
    """ Convert a list in string form, like '[one, two]', to a space-separated 
        string of the tags, underscored. For example 
        '[one fish, two]' -> 'one_fish two'
    """
    if isinstance(string_list, float):
        return ''
    return ' '.join([tag.lower().replace(' ', '_') for tag in string_list[1:-1].split(
        ', ')])
