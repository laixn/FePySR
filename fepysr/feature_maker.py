## Function - Data Generation Code
import numpy as np
import torch
from copy import deepcopy
from collections import Counter
import re
import sympy as sp
from sympy import symbols, sympify,simplify
from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import Float

## Training Code Phase
def extract_layer_expressions(extracted_symbols,symbolic_depth,feature_names,full_net):
    """
    Iterate through each layer of the trained Feature Mapping Network (FMN) to extract and record the generated symbolic expressions.
    
    This function modifies the extracted_symbols list in place, appending newly generated composite expressions 
    (excluding base variables) for each layer.

    Args:
        extracted_symbols (list of lists): Nested list to store the extraction results of each layer.
        num_layers (int): Total number of layers (depth) in the symbolic network.
        feature_names (list of str): List of original input feature names (e.g., ['X0', 'X1']).
        full_net (nn.Module): The complete, converged feature mapping network object after training.
    """
    current_expressions = deepcopy(feature_names)
    for layer_idx in range(symbolic_depth):
        current_expressions=_generate_node_expressions(current_expressions,full_net[layer_idx])
        extracted_symbols[layer_idx].extend(current_expressions[full_net.struct[layer_idx]:])
    return None

def _generate_node_expressions(current_expressions, layer):
    """
    Make hard selections via Argmax based on the current layer's network weights to generate specific mathematical symbol strings.

    Args:
        current_expressions (list of str): List of available features/expressions passed down from the previous layer.
        layer (nn.Module): Single-layer structure of the network, containing unary and binary operator nodes.

    Returns:
        list of str: List with the newly generated expressions from this layer appended.
    """
    # 1. Process all unary operator nodes (e.g., sin, cos, exp, square)
    for i in range(layer.num_single):
        weights = layer[i].w.weight
        # Find the input edge with the maximum weight via argmax, treating it as the input variable selected by the node
        selected_idx = torch.argmax(weights).item()
        
        # Call the expression method of the symbolic function, e.g., turning "X1" into "sin(X1)"
        new_unary_expr = layer[i].fun.Fun_exp(current_expressions[selected_idx])
        current_expressions.append(new_unary_expr)

    # 2. Process all binary operator nodes (e.g., add, mul)
    for i in range(layer.num_double):
        # Index of the binary node in the layer's internal list, immediately following the unary nodes
        node_idx = layer.num_single + i 
        
        weight1 = layer[node_idx].w1.weight
        weight2 = layer[node_idx].w2.weight
        
        # Find the variable indices with the maximum weight on the two input ports, respectively
        selected_idx1 = torch.argmax(weight1).item()
        selected_idx2 = torch.argmax(weight2).item()
        
        # Call the expression method of the symbolic function, e.g., turning "X1" and "X2" into "(X1 + X2)"
        new_binary_expr = layer[node_idx].fun.Fun_exp(
            current_expressions[selected_idx1], 
            current_expressions[selected_idx2]
        )
        current_expressions.append(new_binary_expr)

    return current_expressions

def merge_experiment_results(all_results):
    """
    Aggregate and merge the candidate feature lists extracted from multiple parallel experiments column by column (by network layer).

    Args:
        all_results (list of lists): Nested list of shape (num_experiments, num_layers, ...).
                                     Contains symbolic expressions returned by all parallel tasks.

    Returns:
        list of lists: Merged list of shape (num_layers, ...).
    """
    if not all_results:
        return []
    
    # Get the number of network layers (number of columns)
    num_layers = len(all_results[0])
    
    # Initialize an empty nested list of shape (num_layers, )
    merged_results = [[] for _ in range(num_layers)]

    # Iterate through the results of each individual experiment run
    for experiment_run in all_results:
        # Iterate through each layer in the network
        for layer_idx in range(num_layers):
            # Append all symbols extracted by the current experiment at this layer to the corresponding level in the global results
            merged_results[layer_idx].extend(experiment_run[layer_idx])
            
    return merged_results

## Feature Generation Phase
# ==========================================
# Phase 1: Expression Statistics and Simplification
# ==========================================
def count_all_expressions(num_layers, extracted_symbols):
    """
    Flatten the symbolic expressions extracted from all layers and count the frequency of each expression.
    
    Args:
        num_layers (int): Depth of the network.
        extracted_symbols (list of lists): Nested list of symbolic expressions categorized by layer.
        
    Returns:
        Counter: Counter object containing each expression and its occurrence count.
    """
    all_expressions = []
    for i in range(num_layers):
        all_expressions.extend(extracted_symbols[i])
    
    return Counter(all_expressions)

def _simplify_formula(formula_str):
    """
    Internal helper function: Use SymPy to algebraically expand and simplify the formula string.
    """
    expr = sp.sympify(formula_str)
    expanded_expr = sp.expand(expr)
    return str(expanded_expr)

def merge_duplicate_formulas(expression_counts): 
    """
    Algebraically simplify the frequency-counted formulas, merge mathematically equivalent formulas (including negative equivalents), and accumulate their frequencies.
    
    Args:
        expression_counts (Counter/dict): Dictionary in the format {formula string: frequency}.
        
    Returns:
        list of tuples: Tuple list sorted by frequency in descending order, format: [(simplified formula, total frequency), ...].
    """
    final_counts = {}
    processed_forms_map = {}

    for original_str, count in expression_counts.items():
        # 1. Get the "standard" expanded and simplified form of the current formula
        canonical_expr = _simplify_formula(original_str)
        
        # 2. If this simplified form has been recorded, directly accumulate the count
        if canonical_expr in processed_forms_map:
            canonical_key = processed_forms_map[canonical_expr]
            final_counts[canonical_key] = final_counts.get(canonical_key, 0) + count
            continue

        # 3. Attempt to find its "negative equivalent" (e.g., x-y and y-x are equivalent in feature extraction)
        try:
            expr_obj = sp.sympify(canonical_expr)
            negative_equivalent = str(sp.expand(-expr_obj))
        except Exception:
            # If SymPy parsing fails, retain it as an independent feature
            final_counts[canonical_expr] = final_counts.get(canonical_expr, 0) + count
            processed_forms_map[canonical_expr] = canonical_expr 
            continue 
        
        # 4. Accumulate the current formula count and establish a positive-negative equivalence mapping to prevent future duplicates
        final_counts[canonical_expr] = final_counts.get(canonical_expr, 0) + count
        processed_forms_map[canonical_expr] = canonical_expr
        processed_forms_map[negative_equivalent] = canonical_expr
            
    # Sort by occurrence frequency in descending order and return
    return sorted(final_counts.items(), key=lambda item: item[1], reverse=True)

def filter_expressions_by_variables(expression_tuples, valid_variables):
    """
    Filter the expression list, retaining only those expressions that contain at least one valid base variable.
    (Filter out pure constant terms, e.g., "2*sin(1)")

    Args:
        expression_tuples (list of tuples): Format: [(formula string, frequency), ...].
        valid_variables (list of str): List of allowed base variable names (e.g., ['X0', 'X1']).

    Returns:
        list of tuples: Filtered list of expression tuples.
    """
    if not valid_variables:
        return []
        
    # item[0] is the formula string. any() checks if this string contains any variable from valid_variables
    return [
        item for item in expression_tuples 
        if any(var in item[0] for var in valid_variables)
    ]

def count_and_sort_by_layer(num_layers, extracted_symbols):
    """Count and sort the expression frequencies independently by layer."""
    layer_sorted_results = {}
    for i in range(num_layers):
        layer_counts = Counter(extracted_symbols[i])
        layer_sorted_results[i] = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
    return layer_sorted_results

# ==========================================
# Phase 2: Expression Parsing and Evaluation
# ==========================================
def preprocess_expression(expr, variables):
    """Preprocess the expression: adapt to arbitrary variable names, supplement missing multiplications, and standardize function names"""
    # Generate variable matching regex (supports arbitrary variable names, such as x, y, a1, b2, etc.)
    var_pattern = '|'.join(re.escape(var) for var in variables)
    var_regex = rf'\b({var_pattern})\b'
    
    # 1. Replace absolute value symbols with abs()
    expr = re.sub(r'\|([^|]+)\|', r'abs(\1)', expr)
    # 2. Replace power operation ^ with **
    expr = re.sub(r'\^', '**', expr)
    # 3. Standardize function names (sin/cos/exp/log uniformly lowercase)
    expr = re.sub(r'\b(sin|cos|exp|log)\b', lambda m: m.group().lower(), expr, flags=re.IGNORECASE)
    # 4. Supplement multiplication between "number+variable" (e.g., 2x -> 2*x, 3a -> 3*a)
    expr = re.sub(r'(\d+)(?=' + var_regex + r')', r'\1*', expr)
    # 5. Supplement multiplication between "variable+variable" (e.g., xy -> x*y, a1b2 -> a1*b2)
    expr = re.sub(r'(' + var_regex + r')(?=' + var_regex + r')', r'\1*', expr)
    return expr

def parse_custom_formula(formula_str, variables_list):
    """
    Parse a custom formula string into an executable function
    Args:
        formula_str: Formula string (e.g., "x+sin(y)", "2*a+b**2")
        variables_list: List of variables provided by the user (e.g., ['x', 'y'], ['a', 'b', 'c'])
    Returns:
        func: Callable function (input order consistent with variables_list, supports NumPy arrays)
        used_variables: List of variables actually used in the formula
    """
    # Verify that the variable list is not empty
    if not variables_list:
        raise ValueError("变量列表不能为空")
    
    # 1. Preprocess the formula (supplement multiplication, standardize function names)
    processed_expr = preprocess_expression(formula_str, variables_list)
    
    # 2. Extract variables actually used in the formula (ensure it only contains user-provided variables)
    var_pattern = '|'.join(re.escape(var) for var in variables_list)
    used_variables = sorted(set(re.findall(r'\b(' + var_pattern + r')\b', processed_expr)))
    
    # 3. Verify if there are any undefined variables in the formula
    all_matched = re.findall(r'\b([a-zA-Z_]\w*)\b', processed_expr)  
    defined_vars = set(variables_list + ['sin', 'cos', 'exp', 'log', 'abs'])
    undefined_vars = [var for var in all_matched if var not in defined_vars]
    if undefined_vars:
        raise ValueError(f"公式中包含未定义的变量：{', '.join(set(undefined_vars))}")
    
    # 4. Create symbolic variables (in the order of the user-provided variable list to ensure consistent function input order)
    sym_vars = symbols(','.join(variables_list))
    
    # 5. Convert to symbolic expression and generate a callable function
    expr = sympify(processed_expr)
    callable_func = lambdify(used_variables, expr, "numpy")  # Supports NumPy array input
    
    return callable_func, used_variables

def parse_formula_list(formula_list, valid_variables):
    """Batch parse a list of formulas into a set of executable functions."""
    callable_funcs = []
    used_vars_list = []
    
    for formula_str in formula_list:
        func, used_vars = parse_custom_formula(formula_str, valid_variables)
        callable_funcs.append(func)
        used_vars_list.append(used_vars)
        
    return callable_funcs, used_vars_list

def evaluate_formulas(callable_funcs, used_vars_list, data_dict):
    """
    Pass actual data into dynamically compiled functions to compute the data tensor for new features.
    
    Args:
        callable_funcs: List of parsed functions.
        used_vars_list: List of input variable names corresponding to each function.
        data_dict (dict): Dictionary containing each variable name and its corresponding data (Tensor/Array).
    """
    results = []
    for func, variables in zip(callable_funcs, used_vars_list):
        # Extract the data columns required by the current function
        input_data = [data_dict[var] for var in variables]
        # Vectorized calculation of feature results
        result = func(*input_data)
        results.append(result)
    return results

# ==========================================
# Phase 3: Main Entry for Feature Generation
# ==========================================
def feature_maker(cfg,extracted_symbols,data_analyzer):
    """
    Main controller function for feature generation: integrates candidate expressions from all layers, extracts the Top K, computes their corresponding numerical values, and appends them to the original data.

    Args:
        cfg: Hydra global configuration object.
        extracted_symbols (list of lists): Candidate symbol set outputted by the FMN training network.
        data_analyzer (DataAnalyzer): Core data object managing features and variable names.

    Returns:
        tuple: (Updated data object, list of base + new feature names, selected feature information with frequencies)
    """
    num_layers = cfg.FMN.net.net_depth
    top_k = cfg.data_symbol.fea_num

    # 1. Count, simplify, and merge all feature expressions
    expression_counts = count_all_expressions(num_layers, extracted_symbols)
    merged_expressions = merge_duplicate_formulas(expression_counts)

    # 2. Filter out constant terms, keeping only expressions with valid variables
    valid_expressions = filter_expressions_by_variables(merged_expressions, data_analyzer.feature_names)
    layer_sorted_expressions = count_and_sort_by_layer(num_layers, extracted_symbols)

    # 3. Intercept the top_k most frequent features
    top_feature_tuples = valid_expressions[:top_k]
    new_feature_names = [item[0] for item in top_feature_tuples]

    # 4. String parsing and numerical calculation
    callable_funcs, used_vars_list = parse_formula_list(new_feature_names, data_analyzer.feature_names)
    new_feature_tensors = evaluate_formulas(callable_funcs, used_vars_list, data_analyzer.feature_dict)

    # 5. Update the data dictionary and stack it into a NumPy matrix
    new_features_dict = {name: tensor for name, tensor in zip(new_feature_names, new_feature_tensors)}
    data_analyzer.feature_dict.update(new_features_dict)
    data_analyzer.stack_features_to_numpy()

    # 6. Generate the final comprehensive feature name list (base features + generated new features)
    final_feature_names = data_analyzer.feature_names + new_feature_names
    return data_analyzer, final_feature_names, top_feature_tuples


# Output Equation Processing
def replace_pysr_variables(pysr_equation, feature_names):
    """
    Replace placeholder variables (x0, x1...) in the default formula string generated by PySR with actual physical feature names.
    
    Args:
        pysr_equation (str): Original formula string output by PySR (e.g., 'x0 + x1 * x2').
        feature_names (list of str): List of feature names generated in the first stage. 
                                     Indices strictly correspond, e.g., x0 corresponds to feature_names[0].
                                     
    Returns:
        str: Final formula string after replacing variables and algebraic simplification.
        
    Raises:
        IndexError: Triggered when a variable index appearing in the formula exceeds the length of feature_names.
    """
    # 1. Extract all variables starting with x or X in the formula (e.g., x0, X1, x2)
    extracted_vars = re.findall(r'\b([xX]\d+)\b', pysr_equation)
    if not extracted_vars:
        return pysr_equation  # No variables to replace, directly return the original formula
    
    # 2. Parse variable indices and remove duplicates (the regex \d+ ensures the substring is definitively a number, so try-except is not needed)
    var_indices = set()
    for var in extracted_vars:
        try:
            # Extract the numerical part (x0 -> 0, x1 -> 1)
            idx = int(var[1:])  # When var is 'x0', var[1:] is '0', which converts to the integer 0
            var_indices.add(idx)
        except ValueError:
            raise ValueError(f"公式中包含无效变量格式：{var}")
    
    # 3. Verify if variable indices are within the feature_names range (to avoid cases where x3 exists but feature_names only has 3 elements)
    max_idx = max(var_indices) if var_indices else -1
    if max_idx >= len(feature_names):
        raise IndexError(
            f"PySR 公式中存在越界变量 x{max_idx},"
            f"但当前特征列表仅包含 {len(feature_names)} 个变量。"
        )
    
    # 4. Create a replacement mapping (x0 -> feature_names[0], x1 -> feature_names[1]...)
    replace_map = {f'x{idx}': feature_names[idx] for idx in var_indices}
    
    # 5. Replace variables in the formula using regex (ensure full-word match to avoid mistakenly replacing x01)
    # Build regex pattern: match x0, x1, etc. (e.g., x0|x1|x2)
    var_pattern = '|'.join(re.escape(var) for var in replace_map.keys())
    replaced_formula = re.sub(
        rf'\b({var_pattern})\b',  # Full-word match for x0, x1, etc.
        lambda m: replace_map[m.group(1)],  # Replace with the corresponding name
        pysr_equation
    )

    return _simplify_final_equation(replaced_formula)

def _simplify_final_equation(equation_str, threshold=1e-5):
    """
    Simplify the mathematical formula string, merge like terms, and erase extremely small floating-point constants (treating them as 0).
    
    Note: An underscore is added to the function name for distinction, avoiding naming conflicts with the basic simplify_formula you used in the FMN feature generation phase.
    
    Args:
        equation_str (str): The formula string to be simplified.
        small_const_threshold (float): Threshold for determining extremely small constants. SymPy Floats with absolute values less than this will be replaced with the integer 0 to improve the cleanliness of the final formula.
                                       
    Returns:
        str: The simplified formula string.
    """
    # 1. Convert the formula string to a SymPy symbolic expression
    try:
        sympy_expr = sympify(equation_str)
    except Exception as e:
        raise ValueError(f"公式格式错误,SymPy 无法解析: '{equation_str}'。错误详情: {e}")
    # 2. Replace extremely small constants with 0 (avoid numbers like 9.173594e-9 affecting readability)
    def snap_to_integer(num):
        val = float(num)
        nearest_int = round(val)
        # If the distance to the nearest integer is extremely close, convert it to a perfect SymPy Integer
        if abs(val - nearest_int) < threshold:
            return sp.Integer(nearest_int)
        return num

    cleaned_expr = sympy_expr.replace(lambda x: x.is_Float, snap_to_integer)
    
    # 3. Execute simplification (merge power operations, like terms, etc.)
    simplified_expr = simplify(cleaned_expr)
    
    # 4. Convert the simplified symbolic expression back to a string
    simplified_str = str(simplified_expr)
    simplified_str = simplified_str.replace('**', '^')
    
    return simplified_str