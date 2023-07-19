import re
import torch
from .Node import Node

# PyTorch model that brings everything together
class LaModel(torch.nn.Module):

    def __init__(self, latex, user_symbols_):
        super(LaModel, self).__init__()

        # greek letters
        self.greek_letters = [
            r'\\alpha',
            r'\\beta',
            r'\\gamma',
            r'\\delta',
            r'\\epsilon',
            r'\\zeta',
            r'\\eta',
            r'\\theta',
            r'\\iota',
            r'\\kappa',
            r'\\lambda',
            r'\\mu',
            r'\\nu',
            r'\\phi',
            r'\\chi',
            r'\\psi',
            r'\\rho',
            r'\\sigma',
            r'\\tau',
            r'\\upsilon',
            r'\\omega',
            r'\\xi',
            r'\\pi',
            r'\\phi',
            r'\\varphi'
            ]

        # regex for all the greek letters
        self.gr_lowlet_re = r'(?:' + '|'.join(self.greek_letters) + r')'

        # regex for all the letters
        self.all_lowlet_re   = r'(?:(?:[a-z])|' + self.gr_lowlet_re + r')'

        #TODO: make the following two lists a single dictionary
        # list of regexes to match
        # put the less specific regexes last
        # so I should put the user defined symbols first
        re_list = [
            r'(?:\^)',                              # pow
            r'(?:\\frac)',                          # frac
            r'(?:\+)',                              # sum
            r'(?:\-)',                              # -
            r'(?:\*|\\cdot)',                       # product
            r'(?:\/)',                              # division
            r'((?:[0-9])+(?:\.?(?:[0-9])*))',       # scalar -> maybe remove the external parenthesis?
                                                    # -> NO, I need it to extract the value of the scalar later
            r'(?:\(|\\left\()',                     # left parenthesis
            r'(?:\)|\\right\))',                    # right parenthesis
            r'(?:\{|\})',                           # curly braces
            r'(?:\=)',                              # equal sign
        ]

        # list of types corresponding to the regexes above
        type_list = [
            "pow",
            "frac",
            "sum",
            "-",
            "product",
            "/",
            "scalar",
            "(",
            ")",
            "curly_braces",
            "="
        ]

        # generate the regexes for the user defined symbols and find the metric and other
        # useful information
        self.metric = None

        user_symbols = ["#" + "^"*i for i in range(100)] + user_symbols_
        user_re, self.user_heads, self.user_idx_structs = self.generate_regex_list(user_symbols)

        # find whther the user defined symbols are tensors or scalars
        user_type = []
        for idx_struct in self.user_idx_structs:
            if len(idx_struct) == 0:
                user_type.append("scalar")
            else:
                user_type.append("tensor")

        # add the regexes generated from the user defined symbols
        # to the list of regexes. Do the same thing with the types
        re_list   = user_re + re_list
        type_list = user_type + type_list

        # Create a regex that matches any of the regexes in the list.
        # This is used to perform the tokenization with the tokenize function
        re_tot = r'(' + r'|'.join(re_list) + r')'

        self.free_indices = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j",\
                             "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                             "u", "v", "w", "x", "y", "z"]

        tokens_list = self.tokenize(latex, re_tot)         
        full_meta_list = self.extract_meta(tokens_list, re_list, type_list)

        #print("Meta list:")
        #for i in full_meta_list:
        #    print(i)

        if len(full_meta_list) < 3:
            raise Exception("You need to specify the left part of the expression as well as a reference.\n \
                            This might not be the problem, but it is a possible cause.")
        
        meta_list = full_meta_list[2:]
        self.reference = full_meta_list[0]

        # get all the used indices from the meta_list
        # and remove the already used indices from the list of free indices
        for meta in meta_list:
            if meta["type"] == "tensor":
                self.free_indices = self.from_list_remove_list(self.free_indices, meta["indices"])

        meta_list = self.pre_parse(meta_list)

        self.tree = Node(meta_list, reference=self.reference, metric_name=self.metric)
        self.tree.parse()

    # remove from list1 all items that are in list2
    def from_list_remove_list(self, list1, list2):
        return [item for item in list1 if item not in list2]

    # TODO: figure out if these should be LaModel methods
    # or if they should be in a separate file as utilities

    # Create the regexes for the user defined symbols
    # and add them to the list of regexes
    # For now it only supports user defined tensors

    # function to split the string into tokens  
    def tokenize(self, string, re_tot):
        # findall to split without removing the delimiters
        tokens_iter = re.finditer(re_tot, string)
        tokens = [mat.group(0) for mat in tokens_iter]
        return tokens

    def generate_regex(self, sym):

        mat = re.match(r'(\@)?(.+?)((?: *(?:\_|\^)(?: *)?)*)$', sym)
        if mat:

            head = mat.group(2)
            idx_struct = re.sub(r' ', r'', mat.group(3))

            # find the metric
            if mat.group(1) == "@":
                if idx_struct == "__":
                    self.metric = head
                else:
                    raise Exception("Invalid metric: Metric must have both indices down")
            
            # the final (?![\_\^]) is a negative lookahead
            # to avoid matching when there are other tensors
            # with the same name
            # The final negative lookahead is a problem if
            # I have a scalar of which I then want to exponentiate it
            # ex. x^2. In this case the negative lookahead will prevent
            # the matching of x because there is a "^" after it.
            # Not a problem anymore because I added "\{" at the end of it
            # to get (?![\_\^]\{)

            reg = r'(?:(?:\@)?(' + str(re.escape(head)) + r')((?:[\_\^]\{ *'\
                + str(self.all_lowlet_re) + r' *\}){' + str(len(idx_struct))\
                + r'}))(?![\_\^]\{)'

            return reg, head, idx_struct
        
        raise Exception("Invalid symbol: " + sym)
    
    def generate_regex_list(self, syms):
        reg_list = []
        head_list = []
        idx_struct_list = []
        for sym in syms:
            reg, head, idx_struct = self.generate_regex(sym)
            reg_list.append(reg)
            head_list.append(head)
            idx_struct_list.append(idx_struct)
        return reg_list, head_list, idx_struct_list
    
    # extract the info from every token
    def extract_meta(self, tokens_list, re_list, type_list):

        # initialize the list of metadata of the tokens
        meta_list = []

        for token in tokens_list:
            for i, regex in enumerate(re_list):
                mat_ = re.match(str(regex), str(token))
                if mat_:

                    meta = {}
                    meta["token"] = token
                    
                    if type_list[i] == "tensor":
                        #USARE mat_ !! non mat

                        meta["type"] = "tensor"
                        meta["head"] = mat_.group(1)
                        meta["idx_struct"] = re.sub(r'\{.*?\}', r'', mat_.group(2))

                        # find the indices and remove all the other stuff
                        indices_iter = re.finditer(r'\{ *(' + self.all_lowlet_re + r') *\}', mat_.group(2))
                        meta["indices"] = [i.group(1) for i in indices_iter]

                        meta["return_meta"] = meta

                    elif type_list[i] == "scalar":

                        meta["type"] = "scalar"
                        meta["head"] = mat_.group(1)
                        
                        # if it is a scalar where the value is explicitly given
                        # i store the value in the meta, otherwise I don't store anyhting
                        # and the user will provide the value later
                        if re.match(r'(?:[0-9])+(?:\.?(?:[0-9])*)', mat_.group(1)): #TODO: change this to a better solution. This is not elegant
                            meta["value"] = float(mat_.group(1))

                        meta["return_meta"] = meta
                        
                    
                    else:
                        meta["type"] = type_list[i]
                
            meta_list.append(meta)
        
        return meta_list

    # if i have multiple user defined symbols with the same name
    # i can check both the head and that the len of the idx_struct is the same
    def pre_parse(self, meta_list):

        # I create a new list because I don't want to modify the original one
        # while I'm iterating over it
        
        #TODO: I think I can actually remove this and substitue
        # the new_meta_list with the meta_list in the following code
        times = 0
        new_meta_list = meta_list.copy()
        old_meta_list = []
        while new_meta_list != old_meta_list:
            times +=1
            
            old_meta_list = new_meta_list.copy()
            k = -1
            while k < len(new_meta_list)-1:
                k += 1

                meta = new_meta_list[k]
                if meta["type"] == "tensor":
                    for i in range(len(self.user_heads)):
                        if (meta["head"] == self.user_heads[i]) and \
                        (len(meta["idx_struct"]) == len(self.user_idx_structs[i])) and\
                            meta["head"] != self.metric: # otherwise it would keep looping

                            for j in range(len(meta["indices"])):
                                char_used = meta["idx_struct"][j]
                                char_required = self.user_idx_structs[i][j]
                                if char_used != char_required:
                                    
                                    new_meta = {}
                                    new_meta["type"] = "tensor"

                                    # TODO: in the future I will heve the user provide g
                                    # and I will automatically calculate the inverse and call it with
                                    # a different name. So, the following line will have to go inside
                                    # the following if statements and "g" will be replaced with the
                                    # name of the inverse metric in case it is needed (according to whether
                                    # we have a "^" or a "_")
                                    
                                    new_meta["head"] = self.metric #TODO: change this
                            
                                    
                                    # Case: A_{i} -> A^{j} g_{j}_{i}
                                    if  char_used == "_":
                                        # i think this could give errors
                                        # if i'm checking the last index
                                        if j != len(meta["indices"]) - 1:
                                            meta["idx_struct"] = meta["idx_struct"][:j] + "^" + meta["idx_struct"][j+1:]
                                        else:
                                            meta["idx_struct"] = meta["idx_struct"][:j] + "^"

                                        new_meta["idx_struct"] = "__"

                                    # Case A^{i} -> A_{j} g^{j}^{i}
                                    elif char_used == "^":
                                        if j != len(meta["indices"]) - 1:
                                            meta["idx_struct"] = meta["idx_struct"][:j] + "_" + meta["idx_struct"][j+1:]
                                        else:
                                            meta["idx_struct"] = meta["idx_struct"][:j] + "_"

                                        new_meta["idx_struct"] = "^^"

                                    old_index = meta["indices"][j]
                                    
                                    # find a dummy index
                                    dummy_index = self.free_indices[0]
                                    self.free_indices = self.free_indices[1:]

                                    # I also need to change the index of the original tensor
                                    meta["indices"][j] = dummy_index

                                    new_meta["indices"] = [dummy_index, old_index]

                                    # every tensor has return_meta = meta
                                    new_meta["return_meta"] = meta

                                    # it is only useful to look at the tree faster
                                    meta["token"]     = self.generate_token_from_meta(meta)
                                    new_meta["token"] = self.generate_token_from_meta(new_meta)

                                    new_meta_list.insert(k+1, new_meta)

                                    # I have to increment k by one because I have added a new element
                                    # and I don't have to check that element
                                    k += 1

                            break
                
                elif meta["type"] == "-":
                    
                    # if the previous type is not an operator then - is the operator
                    
                    previous_is_operator = False
                    if k > 0:
                        if new_meta_list[k-1]["type"] in ["sum", "-", "product", "/"]:
                            previous_is_operator = True

                    if not previous_is_operator:

                        new_meta = {}
                        new_meta["type"] = "sum"
                        new_meta["head"] = "+"

                        new_meta_list.insert(k, new_meta)
                        
                        k -= 1

                
                elif meta["type"] == "sum":
                        
                        is_sign = True
                        # do this if k = 0 or if type of k-1 element is "(" (left parenthesis) or an operator
                        if k > 0:
                            if new_meta_list[k-1]["type"] not in ["("]: # add other future operators
                                is_sign = False
                        
                        if is_sign:
                            
                            dummy_new_meta = {}
                            dummy_new_meta["token"] = "0"
                            dummy_new_meta["type"] = "scalar"
                            dummy_new_meta["head"] = "0"
                            dummy_new_meta["value"] = 0
                            dummy_new_meta["return_meta"] = dummy_new_meta
                            new_meta_list.insert(k, dummy_new_meta)
                            k -= 1
                

        return new_meta_list 

    # generate "token" entry in meta from the rest of the meta
    def generate_token_from_meta(self, meta):
        token = meta["head"]
        for i in range(len(meta["indices"])):
            token +=  meta["idx_struct"][i] + "{" + meta["indices"][i] + "}"
        return token

    # evaluate the tree
    def forward(self, var_dict):
        return self.tree.evaluate(var_dict)
 