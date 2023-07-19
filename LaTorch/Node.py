from .Counter import Counter
from .utilities import draw_tree, find_non_repeated_elements, remove_parenthesis
import torch

class Node():

    def __init__(self, meta_list, parent = None, reference = None, metric_name = None):

        # this contains the information about the node
        #
        # meta["token"]       is the token that the node represents.
        #                     
        # meta["type"]        is the type of the token,
        #                     It is the "type" of this node.
        #
        # meta["return_meta"] is the information that the node returns
        #                     to the parent node

        self.meta = {}

        # these are the tokens to be analysed by the node
        self.meta_list = meta_list

        self.parent = parent

        # this is the meta of the tensor on the left side of the equation
        if reference != None:
            self.reference = reference
        else:
            self.reference = self.parent.reference

        self.children  = []
        if self.parent is not None:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

        if meta_list is not None:
            self.meta_list = remove_parenthesis(meta_list)
        else:
            meta_list = None

        if reference != None:
            self.metric_name = metric_name
        else:
            self.metric_name = self.parent.metric_name

    # returns the end_index of the next block
    # of tokens starting from index i  
    def get_next_block_end_idx(self, i):

        if i >= len(self.meta_list): return None

        is_pow = False
        if self.meta_list[i]["type"] != "(":
            if i != len(self.meta_list)-1:
                if self.meta_list[i+1]["type"] != "pow":
                    return i
                is_pow = True
            else:
                return i

        counter = Counter(["("], [")"])
        has_started = False
        for i in range(i, len(self.meta_list)):
            counter.update(self.meta_list[i]["type"])
            if self.meta_list[i]["type"] == "(":
                has_started = True

            if (is_pow and has_started and counter.is_zero()) or (not is_pow and counter.is_zero()):
                return i
        return None


    ### SUM ### parsing of the sum
    def parse_sum(self):

        counter = Counter(["("], [")"])
        for i in range(len(self.meta_list)):
            tok_type = self.meta_list[i]["type"]
            counter.update(tok_type)
            if tok_type == 'sum' and counter.is_zero():
                self.meta = self.meta_list[i]

                a = self.meta_list[0:i]

                # possible error generator if latex sintax is wrong
                b = self.meta_list[i+1:]
                
                self.children = [Node(a, self), Node(b, self)]
                for child in self.children:
                    child.parse()

                if (self.children[0].meta["return_meta"]["type"] == "tensor") and \
                   (self.children[1].meta["return_meta"]["type"] == "tensor"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"
                    self.meta["return_meta"]["indices"] = self.children[0].meta["return_meta"]["indices"]
                
                elif (self.children[0].meta["return_meta"]["type"] == "scalar") and \
                   (self.children[1].meta["return_meta"]["type"] == "scalar"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "scalar"
                
                # TODO: add the case in which the sum is not between two tensors
                #       but between a tensor and a scalar. Even though this would
                #       probably be a different method

                elif (self.children[0].meta["return_meta"]["type"] == "tensor")  and \
                     (self.children[1].meta["return_meta"].get("value", None) == 0):
                
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"
                    self.meta["return_meta"]["indices"] = self.children[0].meta["return_meta"]["indices"]

                elif (self.children[0].meta["return_meta"].get("value", None) == 0)  and \
                     (self.children[1].meta["return_meta"]["type"] == "tensor"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"
                    self.meta["return_meta"]["indices"] = self.children[1].meta["return_meta"]["indices"]

                return True
        return False
            
    ### EINSUM ### parsing of the product in einstein notation
    def parse_einsum(self):

        counter = Counter(["("], [")"])
        on_streak = False
        had_streak = False
        i = -1

        # i put -1 because i will be incremented at the beginning of the loop
        # I use while loop because I need to increment i in the middle of the loop
        # to skip ahead in some cases
        while i < len(self.meta_list)-1:
            i += 1
            
            tok_type = self.meta_list[i]["type"]
            counter.update(tok_type)

            # case in which the token is a tensor
            if (tok_type == "tensor"):

                had_streak = True

                # I'm on a streak of tensors, so this may be an einsum
                # For now I consider it and einsum and I update the meta accordingly
                # I will check later if it is actually an einsum and if it is not I will
                # change the meta

                self.meta["token"] = None
                self.meta["type"] = "einsum"
                self.meta["return_meta"] = {}
                self.meta["return_meta"]["type"] = "tensor"
                
                # qui dovrei fare il parsing per capire se viene fuori scalare
                # oppure posso trattare in tutto questo programma come tensori
                # anche gli scalari (forse no a causa delle divisioni)
                
                self.children.append(Node([self.meta_list[i]], self))

                # if it's a tensor, we already know the meta
                # without needing to parse it

                self.children[-1].meta = self.meta_list[i]
                self.children[-1].meta["return_meta"] = self.meta_list[i]

            # case in which the token is a parenthesis
            elif tok_type == "(":

                # firstly, I find the closing parenthesis to determine the
                # "block" of tokens inside the parenthesis

                par_stop_idx = self.get_next_block_end_idx(i)

                # now I extract the block of tokens
                a = self.meta_list[i:par_stop_idx+1]

                # this block is not yet an einsum as it may not return a tensor
                # I have to parse it to know if it is a tensor or not. For now
                # I consider it a candidate for an element of the einsum

                candidate_node = Node(a, self)
                candidate_node.parse()

                # consider the case in which the block is a tensor
                if candidate_node.meta["return_meta"]["type"] == "tensor":
                    
                    had_streak = True

                    # not very efficient but it works for now. In the future, I could
                    # use the parsing from before.
                    self.children.append(Node(a, self))

                    # now that I parsed it, I know what the block returns so
                    # I can update the meta

                    self.meta["token"] = None
                    self.meta["type"] = "einsum"
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"

                    # TODO: consider the TODO below. If that case is actually
                    #       not possible, I can avoid having to add ["return_meta"]["indices"]
                    #       or idx_struct. If it can happend I have to add themn

                    # NOW I WOULD NEED TO SKIP AHEAD IN THE FOR LOOP
                    i = par_stop_idx

            # you get here if you are on a streak
            # I need the elif because I don't want this to exectue if
            # I just fount a tensor or block for the einsum
            elif True:
                break
        
        # Check if this actually was an einsum and if not undo what has been done
        # (reset the meta)
        if had_streak:
            
            # If I only have one element in the potential einsum, check if
            # this element has repeated indices and if it does, it is an einsum

            # I have to set them to two equal values so that if len(children) != 1,
            # the second condition of the if statment below (the one after the if-else
            # after this comment) is always true          
            
            indices = []
            non_repeated_indices = []

            if (len(self.children) == 1):

                if self.children[0].meta.get("type", None) == "tensor":
                    indices = self.children[0].meta["indices"]
                    non_repeated_indices = find_non_repeated_elements(indices)
                else:

                    # TODO: I actually think this case will never happen
                    #       If this is the case then I can remove this.
                    #       Think if this is true or not

                    self.children[0].parse()
                    indices = self.children[0].meta["return_meta"]["indices"]
                    non_repeated_indices = find_non_repeated_elements(indices)

            # case in which the one element does not constitue an einsum
            if (len(self.children) == 1) and (len(indices) == len(non_repeated_indices)):

                self.children = []
                self.meta["type"] = {}
            
            # case in which the one element does constitue an einsum
            else:

                ##############################
                # case in which the einsum returns a scalar
                #if (len(self.children) == 1) and (len(non_repeated_indices) == 0):
                #    self.meta["return_meta"] = "scalar"

                indices_list = []
                for child in self.children:

                    # The first condition is to avoid parsing a tensor which would yield
                    # and infinite recursion
                    # The second condition is to avoid parsing twice the same child

                    if (child.meta.get("type", None) != "tensor") and (len(self.children) != 1):
                        child.parse()

                    indices_list += child.meta["return_meta"]["indices"]
                
                # Now I know what the block/tensor returns so I can update the return meta
                # (maybe this is the solution to the TODO above)

                self.meta["return_meta"]["indices"] = find_non_repeated_elements(indices_list)

                if len(self.meta["return_meta"]["indices"]) == 0:
                    self.meta["return_meta"]["type"] = "scalar"
                
                return True
            
        return False

    def parse_product(self):

        counter = Counter(["("], [")"])
        for i in range(len(self.meta_list)):
            tok_type = self.meta_list[i]["type"]
            counter.update(tok_type)
            if tok_type == 'product' and counter.is_zero():
                
                self.meta = self.meta_list[i]

                a = self.meta_list[0:i]

                # possible error generator if latex sintax is wrong
                b = self.meta_list[i+1:]
                
                self.children = [Node(a, self), Node(b, self)]
                for child in self.children:
                    child.parse()

                if (self.children[0].meta["return_meta"]["type"] == "tensor") and \
                   (self.children[1].meta["return_meta"]["type"] == "tensor"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"
                    self.meta["return_meta"]["indices"] = self.children[0].meta["return_meta"]["indices"]
                
                elif (self.children[0].meta["return_meta"]["type"] == "scalar") and \
                   (self.children[1].meta["return_meta"]["type"] == "scalar"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "scalar"
                
                # TODO: add the case in which the sum is not between two tensors
                #       but between a tensor and a scalar. Even though this would
                #       probably be a different method

                elif (self.children[0].meta["return_meta"]["type"] == "tensor")  and \
                     (self.children[1].meta["return_meta"]["type"] == "scalar"):
                
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"
                    self.meta["return_meta"]["indices"] = self.children[0].meta["return_meta"]["indices"]

                elif (self.children[0].meta["return_meta"]["type"] == "scalar")  and \
                     (self.children[1].meta["return_meta"]["type"] == "tensor"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "tensor"
                    self.meta["return_meta"]["indices"] = self.children[1].meta["return_meta"]["indices"]

                return True
        return False

    def parse_minus(self):

        i = -1

        # i put -1 because i will be incremented at the beginning of the loop
        # I use while loop because I need to increment i in the middle of the loop
        # to skip ahead in some cases
        while i < len(self.meta_list)-1:
            i += 1

            if (self.meta_list[i]["type"] == "-"):

                self.meta["type"] = "-"

                block_end_idx = self.get_next_block_end_idx(i+1)
                a = self.meta_list[i+1:block_end_idx+1]
                self.children = [Node(a, self)]

                # I don't think I actually need to update i
                # because I return right after this
                
                self.children[0].parse()
                
                self.meta["return_meta"] = self.children[0].meta["return_meta"]

                return True
            
            return False

    def parse_div(self):

        counter = Counter(["("], [")"])
        for i in range(len(self.meta_list)):
            tok_type = self.meta_list[i]["type"]
            counter.update(tok_type)
            if tok_type == '/' and counter.is_zero():
                
                self.meta = self.meta_list[i]

                a = self.meta_list[0:i]

                # possible error generator if latex sintax is wrong
                b = self.meta_list[i+1:]
                
                self.children = [Node(a, self), Node(b, self)]
                for child in self.children:
                    child.parse()

                if (self.children[0].meta["return_meta"]["type"] == "scalar") and \
                   (self.children[1].meta["return_meta"]["type"] == "scalar"):
                    
                    self.meta["return_meta"] = {}
                    self.meta["return_meta"]["type"] = "scalar"
                
                # TODO: add the case in which the sum is not between two tensors
                #       but between a tensor and a scalar. Even though this would
                #       probably be a different method

                return True
        return False

    def parse_pow(self):

        counter = Counter(["("], [")"])
        for i in range(len(self.meta_list)):
            tok_type = self.meta_list[i]["type"]
            counter.update(tok_type)
            if tok_type == 'pow' and counter.is_zero(): # I don't think I actually need tbe counter here
                self.meta = self.meta_list[i]

                a = self.meta_list[0:i]

                # possible error generator if latex sintax is wrong
                b = self.meta_list[i+1:]
                
                self.children = [Node(a, self), Node(b, self)]
                for child in self.children:
                    child.parse()

                return_type = self.children[0].meta["return_meta"]["type"]
                self.meta["return_meta"] = {}
                self.meta["return_meta"]["type"] = return_type
                
                if return_type == "tensor":
                    self.meta["return_meta"]["indices"] = self.children[0].meta["return_meta"]["indices"]

                return True
        return False

    ### LEAF ### case in which the token is a leaf of the logical tree
    def parse_leaf(self):

        # I cannot have a leaf with more than one token
        if len(self.meta_list) != 1:

            # (The exception also gets called when the token does not
            # match any of the user defined variables)

            raise Exception("Leaf with more than 1 token!")
        
        # If this is indeed a leaf, I already know what it returns,
        # so I update the meta

        self.meta = self.meta_list[0]

        return True

    ### PARSE ### function that calls all the other parse functions
    def parse(self):

        # If one parsing is completed, I don't parse the rest.
        # This defines the order of the operations
        # TODO: check if this is the correct order

        parse_list = [
            self.parse_sum,
            self.parse_product,
            self.parse_einsum,
            self.parse_div,
            self.parse_minus, # it is important that this is one of the last ones
                              # because otherwise because of how it works it would
                              # only take the next block which in cases like -2/3
                              # is not what we want. In this case it would consider
                              # -2 and discard the rest ("/3")
            self.parse_pow,
            self.parse_leaf
            ]

        done = False
        while not done:
            for parse_func in parse_list:
                done = parse_func()
                if done:
                    break

    ### SUM ### function that evaluates the sum node
    def evaluate_sum(self, var_dict):
        # this works for both tensors and scalars
        return self.children[0].evaluate(var_dict) + self.children[1].evaluate(var_dict)
    
    def evaluate_product(self, var_dict):
        return self.children[0].evaluate(var_dict) * self.children[1].evaluate(var_dict)
    
    def evaluate_div(self, var_dict):
        return self.children[0].evaluate(var_dict) / self.children[1].evaluate(var_dict)
    
    def evaluate_minus(self, var_dict):
        return -self.children[0].evaluate(var_dict)
    
    def evaluate_pow(self, var_dict):
        return torch.pow(self.children[0].evaluate(var_dict), self.children[1].evaluate(var_dict))
    
    ### EINSUM ### function that evaluates the einsum node
    def evaluate_einsum(self, var_dict):
        evaluated_child_list = []
        indices_list = []
        indices_str = ""
        
        for child in self.children:
            evaluated_child_list.append(child.evaluate(var_dict))
            indices_list += child.meta["return_meta"]["indices"]

            # NON FUNZIONA SE INDICI SONO LETTERE GRECHE
            indices_str += "".join(child.meta["return_meta"]["indices"] ) + ","
        
        indices_str = indices_str[:-1] # remove last comma
        indices_str = indices_str + "->" + "".join(find_non_repeated_elements(indices_list))
        
        return torch.einsum(indices_str, *evaluated_child_list)
    
    ### TENSOR ### function that evaluates the tensor node
    def evaluate_tensor(self, var_dict):

        # ADD STUFF HERE
        # check the indices in the refercnce tensor and the indices in the
        # tensor that is being evaluated. If they are different, permute
        # the tensor that is being evaluated

        # it's quite complicated though because the tensor considered
        # might have more indices than the reference tensor
        # in which case ???
        
        if self.meta["head"] == self.metric_name:
            if self.meta["idx_struct"] == "^^":
                return torch.inverse(var_dict[self.meta["head"]])

        return var_dict[self.meta["head"]]
    
    def swap_indices(self, obj):

        # ADD STUFF HERE
        # check the indices in the refercnce tensor and the indices in the
        # tensor that is being evaluated. If they are different, permute
        # the tensor that is being evaluated

        # it's quite complicated though because the tensor considered
        # might have more indices than the reference tensor
        # in which case ???

        # I SHOULD ALSO CHECK WHETHER THE INDICES IN THE RETURN_METAS
        # ARE STORED IN THE RIGHT ORDER

        if self.meta["return_meta"]["type"] != "tensor": return obj
        #if self.meta["return_meta"]["type"] != "tensor": return obj

        #### TODO ###
        if self.parent != None:
            ref = self.parent.meta["return_meta"]
        else:
            ref = self.reference

        if ref.get("indices", None) != None:
            ref_ = [idx for idx in ref["indices"] if idx in self.meta["return_meta"]["indices"]]
            #non_repeated_indices = find_non_repeated_elements(self.meta["indices"])
            idx_str = ""
            target_idx_str = ""
            # create list with all lowercase letters
            spare_indices = [chr(i) for i in range(97, 123) if chr(i) not in self.meta["return_meta"]["indices"]]
            for idx in self.meta["return_meta"]["indices"]:

                # this has to be checked against self.reference["indices"]
                # and not reference_copy
                if idx in ref["indices"]:
                    idx_str += ref_.pop(0)
                    target_idx_str += idx
                else:
                    #idx_str += idx
                    spare = spare_indices.pop(0)
                    idx_str += spare
                    target_idx_str += spare

            # as of now the problem is that if I have two indices where one is not in the
            # reference but i need to swap them (like B_i_j = C_j_n B_n_i) it doesn't recognize 
            # that it has to swap as n is not in the reference. How do I solve this?
            # I could add a reference at every node and it would be the return_meta["indices"] of the node above
            # which can still be calculated without having to do the swap before. And if I am in the top node
            # then the reference becomes the one on the left side of the equation. This way, though, I would
            # need to do this swapping when I evaluate every node where the return_type is a tensor and not 
            # only when I evaluate a tensor. I could create a function to add before returning 
            # in the evaluate method like:
            # return swap_indeces(funct_dict[self.meta["type"]](var_dict))

            conversion_string = idx_str + "->" + target_idx_str #"".join(self.meta["indices"])

            return torch.einsum(conversion_string, obj)
        else:
            return obj
        

    def evaluate_scalar(self, var_dict):
        if self.meta.get("value", None) == None:
            return torch.tensor(var_dict[self.meta["head"]])
        else:
            return torch.tensor(self.meta["value"])

    ### EVALUATE ### function that evaluates the logical tree
    def evaluate(self, var_dict):

        # Check for the type of the node (obtained through parsing)
        # and call the appropriate function

        funct_dict = {
            "sum":     self.evaluate_sum,
            "product": self.evaluate_product,
            "/":       self.evaluate_div,
            "einsum":  self.evaluate_einsum,
            "-":       self.evaluate_minus,
            "pow":     self.evaluate_pow,
            "tensor":  self.evaluate_tensor,
            "scalar":  self.evaluate_scalar
        } 

        #return funct_dict[self.meta["type"]](var_dict)
        return self.swap_indices(funct_dict[self.meta["type"]](var_dict))

    ### REPR ### function that prints the logical tree
    def __repr__(self):
        return draw_tree(self)

