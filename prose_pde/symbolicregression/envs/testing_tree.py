import sympy.functions
import torch
import numpy as np
import scipy.special
from logging import getLogger
from scipy.integrate import solve_ivp
from jax import numpy as jnp
from sympy.parsing.sympy_parser import untokenize, generate_tokens, parse_expr
import io

logger = getLogger()

try:
    from symbolicregression.envs.node_utils import Node, NodeList
except:
    from node_utils import Node, NodeList
# from symbolicregression.envs.ode_generator import ODEGenerator

import sympy as sy

def diff_terms(self, lst):
    """
    Generate a tree containing differentiation of terms in lst
    """
    p = self.params
    tree = None
    for i in reversed(range(len(lst))):
        if tree is None:
            tree = Node(lst[i], p)
        else:
            tree = Node("diff", p, [Node(lst[i], p), tree])
    return tree

def sympy_to_lists(expr):
    term_list = []
    op_list = []
    for args in sy.preorder_traversal(expr):
        ops = args.func
        term_list.append(args)
        op_list.append(ops)
    return term_list, op_list

def get_dim(term_list,op_list):
    """
    This is getting dimension values and what the function u depends on.
    """
    function_list = []
    argus_list =[]
    dim_list = []
    for i in range(len(op_list)):
        if str(op_list[i]).startswith('u') and term_list[i] not in function_list:
            function_list.append(term_list[i])
    for fxn in function_list:
        fun = str(fxn).split('_')
        fun = fun[1]
        fun = fun.split('(')
        dim = fun[0]
        argus = '(' + fun[1]
        argus_list.append(argus)
        dim_list.append(dim)
    
    return argus_list, dim_list


def the_remover(term_list, op_list):
    """
    This takes the term_list and op_list and removes the arguments from functions.  That is, if we have u(x,t), it will remove x,t.
    """
    argus, dim = get_dim(term_list,op_list)
    sym_tup = lambda num: sy.symbols('t, x:{}'.format(num))
    create_args = len(sy.sympify(argus[0]))-1     #don't factor t into this.
    args = sym_tup(create_args)
    args = str(args).split('(')
    args = args[1].split(')')
    args = sy.sympify(args[0])

    u = lambda numb, args: sy.Function('u_{}'.format(numb))(args) #this is for multi-dimensional functions
    for ii in range(len(op_list)):
        for numb in dim:    
            if op_list[ii] == type(sy.sympify(u(numb,args))):
                part_to_remove = create_args+1  #+1 for t.
                for j in range(ii+1,ii+part_to_remove+1):   #+1 for next term.
                    term_list[j] = 0
                    op_list[j] = 0
    term_list_final = []
    op_list_final = []
    for lemon in term_list:
        if lemon != 0:
            term_list_final.append(lemon)
    for orange in op_list:
        if orange != 0:
            op_list_final.append(orange)
    return term_list_final, op_list_final


def check_tree_for_derivatives(term_list, op_list):
    """
    This is going to go into the tree_from_sympy function to check the args and see if we need to change it to 
    a derivative/function. i.e. say we are multiplying ux and u, for the encoder to read that, we need
    self.mul_terms(["ux_0","u_0"])

    We can also check if we have a specific number.

    This returns a list:
    [function: u_#, [tuples: (variable, how many derivatives)]
    and a new term and operator list for the trees.
    """
    x, t = sy.symbols('x t')
    u = sy.Function('u_0')(x,t)
    max_list = len(op_list)
    deriv_args = []
    op_list_final=[]
    term_list_final =[]

    for ii in reversed(range(max_list)):
        if op_list[ii] == type(sy.Derivative(u,x)):
            fxn = op_list[ii+1] #get dim from this as well
            j = ii+2
            deriv_tup = []
            term_list[ii+1]=0 #remove function argument from tree
            op_list[ii+1]=0
            while j < max_list:
                if op_list[j] == type(sy.sympify((2,u))):
                    deriv_tup.append(term_list[j])
                    len_tup = len(term_list[j])
                    for jj in range(j,1 + len_tup + j): #remove tuple (+1) then remove the args of tuple (+len_tup)
                        term_list[jj] = 0    #remove tuple and number argument from tree
                        op_list[jj] = 0
                elif op_list[j] != type((x,2)) and op_list[j] != type(sy.sympify(2)) and op_list[j] != type(sy.sympify(1)) and op_list[j] != 0: #may cause an issue if we have ux * const, but idk
                    break
                j = j + 1   
            deriv_args.append((deriv_tup, fxn))

    for i in range(max_list):
        if op_list[i] != 0:
            term_list_final.append(term_list[i])
            op_list_final.append(op_list[i])

    
    return deriv_args, term_list_final, op_list_final

def string_derivatives(deriv_args):
    """
    The input is a list:
    [function: u_#, [tuples: (variable, how many derivatives)]]

    The goal is to turn the derivative into the correct character.

    output:
    list of derivatives as strings IN REVERSE ORDER OF APPEARANCE
    """
    str_deriv_list = []
    for ii in range(len(deriv_args)):
        fxn = str(deriv_args[ii][1])
        indep_var = deriv_args[ii][0]    
        for indep in indep_var:
            ind = str(indep[0])
            mult = indep[1]
            str_deriv = 'd_' + ind*mult
        deriv_tup = (str_deriv, fxn)
        str_deriv_list.append(deriv_tup)
    return str_deriv_list

def make_list_fancy(str_deriv_list, term_list, op_list):
    """
    This is going to make everything that needs to be a string in term_list into a string.
    """
    x, t = sy.symbols('x t')
    u = sy.Function('u_0')(x,t)
    ind_list = []

    for i in reversed(range(len(op_list))):
        if op_list[i] == type(sy.diff(u,x)):
            ind_list.append(i)
        if op_list[i] == type(u):
            term_list[i] = "u_0"
        if op_list[i] == type(sy.sympify(2.3)): #may have to annoyingly add integers as well?
            term_list[i] = str(term_list[i])
    for ii in range(len(ind_list)):
        ind = ind_list[ii]
        term_list[ind] = str_deriv_list[ii]
    return term_list

def tree_from_sympy(self, term_list, op_list):
    """
    Generate a PROSE tree from the sympy tree by iterating over the operations.
    """
    #clean the term_list and op_list.
    term_list, op_list = the_remover(term_list,op_list)
    deriv_args, term_list, op_list = check_tree_for_derivatives(term_list,op_list)
    str_derivs = string_derivatives(deriv_args)
    term_list = make_list_fancy(str_derivs,term_list,op_list)

    p = self.params
    dim = 0 #do the thing where we parse the dimension from u by splitting it at _ then do {'u_' + str(dim)} and check that. don't think we need this here.
    x, t = sy.symbols('x t')
    u = sy.Function('u_0')(x,t)
    final_term_list = term_list     #initialize
    int_term_list = []
    it = 0                          #just in case

    while len(final_term_list) != 1 or it < 1000: 
        itms = []
        tree = None
        for l in reversed(range(len(op_list))):
            if op_list[l] == (x + 2).func:     #Note: x+2 is random.  We are just choosing a sympy function with addition.
                the_comb = len(term_list[l].args)
                for ii in range(l+1, 1 + l + the_comb):  #+1 for the term after add, +the_comb for the total terms to move into the list.
                    itms.append(term_list[ii])
                    final_term_list[ii] = 0
                final_term_list[l] = self.add_terms(itms)
            elif op_list[l] == (2*x).func:
                """
                Sympy treats division as multiplication, so we will have to check the args and if one of the args is 1/x
                then we will have to do "div".  I need to see if we can check that. func for 1/x is pow...
                """
                the_comb = len(term_list[l].args)
                for ii in range(l+1, 1 + l + the_comb):  #+1 for the term after add, +the_comb for the total terms to move into the list.
                    itms.append(term_list[ii])
                    final_term_list[ii] = 0
                final_term_list[l] = self.mul_terms(itms)
            elif op_list[l] == (x**2).func:
                """
                Here is another weird one.  We only really have pow2 and pow3.  I would again need to check the power args
                and then try to raise it to the power.  I am just using the "pow" arg.
                """
                the_comb = len(term_list[l].args)
                for ii in reversed(range(l+1,1 + l + the_comb)):
                    if tree is None:
                        tree = final_term_list[ii]
                    else:
                        tree = Node("pow",p,[Node(final_term_list[ii],p), tree])
                    final_term_list[ii] = 0
            elif op_list[l] == type(sy.diff(u,x)):
                itms.append(term_list[l][0])
                itms.append(term_list[l][1])
                final_term_list[l] = self.diff_terms(itms)
            elif op_list[l] == type(sy.sin(x)):
                the_comb = len(term_list[l].args)
                for ii in reversed(range(l+1,1 + l + the_comb)):
                    if tree is None:
                        tree = final_term_list[ii]
                    else:
                        tree = Node("sin",p,[Node(final_term_list[ii],p), tree])
                    final_term_list[ii] = 0
            elif op_list[l] == type(sy.cos(x)):
                the_comb = len(term_list[l].args)
                for ii in reversed(range(l+1,1 + l + the_comb)):
                    if tree is None:
                        tree = final_term_list[ii]
                    else:
                        tree = Node("cos",p,[Node(final_term_list[ii],p), tree])
                    final_term_list[ii] = 0

            for jj in range(len(final_term_list)):
                if final_term_list[jj] != 0:
                    int_term_list.append(final_term_list[jj])
            final_term_list = int_term_list
        it = it + 1
        
    tree = final_term_list[0]


def _decode(lst, all_operators, symbols):
    """
    Decode list of symbols in prefix notation into a tree
    """
    # The idea for now is to instead turn res into a final Sympy expression by modifying lst until it is one expression?

    if len(lst) == 0:
        return None, 0
    elif "OOD" in lst[0]:
        return None, 0
    elif lst[0] in all_operators.keys():
        res = [lst[0]]
        arity = all_operators[lst[0]]
        pos = 1
        for i in range(arity):
            child, length = _decode(lst[pos:], all_operators, symbols)
            if child is None:
                return None, pos
            try:
                if child.startswith('u_'):
                    x, t = sy.symbols('x t')
                    u = lambda numb: sy.Function('u_{}'.format(numb))(x,
                                                                      t)  # this can be changed to figure out how many spatial dimensions it takes in.
                    num = child.split('_')[1]
                    child = u(num)
            except:
                pass
            res.append(child)
            pos += length
        if res[0] == "diff":  # this fails if "diff" is not a leaf.
            x, t = sy.symbols('x t')
            u = sy.Function('u_0')(x, t)
            # dim = res[2].split('_')[1]  #this is so we can expand in the future
            try:
                # res[2]= sy.sympify(res[2])
                # res[2] = res[2].subs('u_{}'.format(dim),u)
                expr = res[2]
                deriv = res[1].split('_')[1]
                for args in deriv:
                    expr = sy.diff(expr, args)
                res[1] = expr
                res = res[1]
            except:
                pass
        return res, pos
    elif lst[0].startswith("INT"):
        val, length = parse_int(lst)
        return str(val), length
    elif lst[0] == "+" or lst[0] == "-":
        try:
            val = decode(lst[:3])[0]
        except Exception as e:
            # print(e, "error in encoding, lst: {}".format(lst))
            return None, 0
        return str(val), 3
    elif lst[0].startswith("CONSTANT") or lst[0] == "y":  ##added this manually CAREFUL!!
        return lst[0], 1
    elif lst[0] in symbols:
        return lst[0], 1
    else:
        try:
            float(lst[0])  # if number, return leaf
            return lst[0], 1
        except:
            return None, 0


def parse_int(lst):
    """
    Parse a list that starts with an integer.
    Return the integer value, and the position it ends in the list.
    """
    base = 10
    val = 0
    i = 0
    for x in lst[1:]:
        if not (x.rstrip("-").isdigit()):
            break
        val = val * base + int(x)
        i += 1
    if base > 0 and lst[0] == "INT-":
        val = -val
    return val, i + 1


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def decode(lst):
    """
    Parse a list that starts with a float.
    Return the float value, and the position it ends in the list.
    """
    if len(lst) == 0:
        return None
    seq = []
    for val in chunks(lst, 2 + 4):
        for x in val:
            if x[0] not in ["-", "+", "E", "N"]:
                return np.nan
        try:
            sign = 1 if val[0] == "+" else -1
            mant = ""
            for x in val[1:-1]:
                mant += x[1:]
            mant = int(mant)
            exp = int(val[-1][1:])
            value = sign * mant * (10 ** exp)
            value = float(value)
        except Exception:
            value = np.nan
        seq.append(value)
    return seq

def encode(values):
    """
    Write a float number
    """
    precision = 3

    if len(values.shape) == 1:
        seq = []
        value = values
        for val in value:
            assert val not in [-np.inf, np.inf]
            sign = "+" if val >= 0 else "-"
            m, e = (f"%.{precision}e" % val).split("e")
            i, f = m.lstrip("-").split(".")
            i = i + f
            tokens = chunks(i, 4)
            expon = int(e) - precision
            if expon < -10:
                tokens = ["0" *3] *1
                expon = int(0)
            seq.extend([sign, *["N" + token for token in tokens], "E" + str(expon)])
        return seq
    else:
        seqs = [encode(values[0])]
        N = values.shape[0]
        for n in range(1, N):
            seqs += [encode(values[n])]
    return seqs

def write_int(val):
    """
    Convert a decimal integer to a representation in the given base.
    """
    #if not self.params.use_sympy:
    #    return [str(val)]

    base = 4
    res = []
    max_digit = abs(base)
    neg = val < 0
    val = -val if neg else val
    while True:
        rem = val % base
        val = val // base
        if rem < 0 or rem > max_digit:
            rem -= base
            val += 1
        res.append(str(rem))
        if val == 0:
            break
    res.append("INT-" if neg else "INT+")
    return res[::-1]


def format_float_coefficients(expr):
    # Extract the terms and their coefficients
    terms = expr.as_ordered_terms()
    coefficients = [term.as_coeff_Mul()[0] for term in terms]

    # Format the coefficients: only convert floats to scientific notation
    formatted_coefficients = [
        '{:.2e}'.format(coef.evalf()) if coef.is_Float else str(coef)
        for coef in coefficients
    ]

    # Reconstruct the expression with formatted coefficients
    formatted_expr = ' + '.join(
        f'{formatted_coef}*{term.as_coeff_Mul()[1]}' if term.as_coeff_Mul()[1] != 1 else formatted_coef
        for formatted_coef, term in zip(formatted_coefficients, terms)
    )

    return formatted_expr

def sympy_encoder(self,expr):
    formatted_expr = format_float_coefficients(expr)
    expression_str = str(formatted_expr)
    tokens = list(generate_tokens(io.StringIO(expression_str).readline))

    #all this is putting the PROSE float encoder into it.
    res = []
    for token in tokens:
        try:            
            token_str = token.string
            val = float(token_str)
            if token_str.lstrip("-").isdigit():
                #res.extend(write_int(int(token_str)))
                res.extend(token_str)
            else:
                res.extend(self.float_encoder.encode(np.array([val])))
        except ValueError:
            res.append(token_str)
    res.append(token.string)

    return res

def main():
    c = 0.02

    x, t = sy.symbols('x t')
    u = sy.Function('u_0')(x, t)

    sin_gord = sy.diff(u, (t, 2)) - c ** 2 * sy.diff(u, (x, 2)) + c * sy.sin(u)
    print("Origin sympy information")
    print(sin_gord)
    # Reconstruct the expression with formatted coefficients
    formatted_expr = format_float_coefficients(sin_gord)
    print(formatted_expr)
    expression_str = str(formatted_expr)
    print("Convert it to string:")
    print(expression_str)
    sympy_expr = parse_expr(expression_str)
    print("Convert the string back to sympy")
    print(sympy_expr)

    # Generate tokens from the string representation
    tokens = list(generate_tokens(io.StringIO(expression_str).readline))

    # Print the generated tokens
    print("Generated Tokens:")
    #all this is is putting the PROSE float encoder into it.
    res = []
    for token in tokens:
        try:            
            token_str = token.string
            val = float(token_str)
            if token_str.lstrip("-").isdigit():
                #res.extend(write_int(int(token_str)))
                res.extend(token_str)
            else:
                res.extend(encode(np.array([val])))
        except ValueError:
            res.append(token_str)
    res.append(token.string)
    
    print(res)
    
    #----------------------------------------------------------------------
    #If the transformer outputs it in list of strings form, this is a way to decode it.
    str_sympy = ""
    for tok in res:
        tok = tok.lstrip("N") # if the encoder adds the N.
        str_sympy = str_sympy + tok
    untokenized_expr = sy.sympify(str_sympy)
    print(untokenized_expr)
    #----------------------------------------------------------------------

    # Convert the tokens back into a string
    untokenized_expression = untokenize(tokens)
    #
    # # Print the untokenized expression
    print("\nUntokenized Expression:")
    print(untokenized_expression)
    # #print(u.args)

if __name__ == '__main__':
    main()




