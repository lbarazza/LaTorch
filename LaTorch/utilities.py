from .Counter import Counter

# finds non repeated elements in a list
def find_non_repeated_elements(lst):
    # Create a dictionary to count the occurrences of each element
    count_dict = {}
    for element in lst:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    return [key for key, value in count_dict.items() if value == 1]




# function that removes eventual leading and trailing parenthesis
# from a list of tokens
def remove_parenthesis(meta_list):

    # check for cases like (A_i_j B_j_k) (C_k D_m) in which case
    # I should not remove the parenthesis
    counter = Counter(["("], [")"])
    for i in range(len(meta_list)):
        counter.update(meta_list[i]["type"])
        if counter.is_zero() and i != len(meta_list)-1:
            return meta_list
    
    if meta_list[0]["type"] == "(" and meta_list[-1]["type"] == ")":
        return remove_parenthesis(meta_list[1:-1])
    else:
        return meta_list





# function that creates the logical tree string
# TODO: fix the "|" above some nodes

def draw_tree(node, prefix="", is_last=False):
    children = node.children
    # Check if this is the last item in the list
    if node != node.parent.children[-1] if node.parent else False:
        new_prefix = prefix + " │   "
        next_prefix = prefix + " ├── "
    else:
        new_prefix = prefix + "     "
        next_prefix = prefix + " └── "

    ret = next_prefix + "[*]  Type: " + str(node.meta.get("type", None)) + \
        "\n" + new_prefix + (" │   " if children else "     ") + "Toks: " + '"' + str(node.meta.get("token", None)) + '"' + \
        "\n" + new_prefix + (" │   " if children else "     ") + "Meta: " + '"' + str(node.meta) + '"'
        
    if children:
        ret += "\n" + new_prefix + " │"
        for index, child in enumerate(children):
            ret += "\n" + draw_tree(child, new_prefix, index == len(children) - 1)
        ret += "\n" + new_prefix
    elif node.parent and node != node.parent.children[-1]:
        ret += "\n" + new_prefix + " │"
    else:
        ret += "\n" + new_prefix

    return ret
