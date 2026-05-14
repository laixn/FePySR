# Activation functions
import torch

class BaseFunction:
    """Abstract class for primitive functions"""
    def __init__(self):
        self.dim = 1
        self.name=None

    def forward(self, x):
        return None
    
    def Fun_exp(self, x):
        """Expression"""
        return None

class square(BaseFunction):

    def __init__(self):
        self.dim = 1
        self.name="square"

    def forward(self, x_data):
        return x_data**2
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"{x_name}^2"  

class Trigon_sin(BaseFunction):
    
    def __init__(self):
        self.dim = 1
        self.name="sin"
    def forward(self, x_data):
        return torch.sin(x_data)
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"sin({x_name})"

class Trigon_cos(BaseFunction):
        
    def __init__(self):
        self.dim = 1
        self.name="cos"
    def forward(self, x_data):
        return torch.cos(x_data)
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"cos({x_name})"

class fun_abs(BaseFunction):

    def __init__(self):
        self.dim = 1
        self.name="abs"
    def forward(self, x_data):
        return torch.abs(x_data)
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"|{x_name}|"
    
class exp(BaseFunction):

    def __init__(self):
        self.dim = 1
        self.name="exp"

    def forward(self, x_data):
        return torch.exp(x_data)
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"exp({x_name})"

class log(BaseFunction):
    def __init__(self):
        self.dim = 1
        self.name="log"
    def forward(self, x_data):
        return torch.log(torch.abs(x_data))
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"log({x_name})"

class sqrt(BaseFunction):
    def __init__(self):
        self.dim = 1
        self.name="sqrt"
    def forward(self, x_data):
        return torch.sqrt(torch.abs(x_data))
    
    def Fun_exp(self, x_name):
        """Expression"""
        return f"sqrt({x_name})"

class mul(BaseFunction):
    
    def __init__(self):
        self.dim = 2
        self.name="*"
    def forward(self, x1_data, x2_data):
        return x1_data*x2_data
    
    def Fun_exp(self, x1_name,x2_name):
        """Expression"""
        return f"{x1_name}*{x2_name}"

class add(BaseFunction):
    
    def __init__(self):
        self.dim = 2
        self.name="+"
    def forward(self, x1_data, x2_data):
        return x1_data+x2_data
    
    def Fun_exp(self, x1_name,x2_name):
        """Expression"""
        return f"{x1_name}+{x2_name}"
    
class div(BaseFunction):
    
    def __init__(self):
        self.dim = 2
        self.name="/"
    def forward(self, x1_data, x2_data):
        return x1_data/x2_data
    
    def Fun_exp(self, x1_name,x2_name):
        """Expression"""
        return f"{x1_name}/{x2_name}"

class sub(BaseFunction):

    def __init__(self):
        self.dim = 2
        self.name="-"
    def forward(self, x1_data, x2_data):
        return x1_data-x2_data
    
    def Fun_exp(self, x1_name,x2_name):
        """Expression"""
        return f"{x1_name}-{x2_name}"

def delete_small(w,size=0.1):
    with torch.no_grad():
            w[w < size] = 0

def L1Nor(w):
    with torch.no_grad():  # Disable gradient calculation since we are modifying parameters directly
        param_sum = w.sum()  # Calculate the sum of parameters
        if param_sum != 0:  # Ensure the sum of parameters is not zero
            w /= param_sum      

def MaxNor(w):
    with torch.no_grad():  # Disable gradient calculation since we are modifying parameters directly
    # Normalize network parameters
        param_max = torch.max(w)  # Calculate the sum of parameters#####
        if param_max != 0:  # Ensure the sum of parameters is not zero
            w/= param_max

func_mapping = {
    "square": square(),       
    "exp": exp(),             
    "sin": Trigon_sin(),      
    "cos": Trigon_cos(),      
    "mul": mul(),             
    "add": add(),             
    "sub": sub()             
}

default_func1 = [
    *[square()] * 4,
    *[exp()] * 4,
    *[Trigon_sin()] *4,
    *[Trigon_cos()] *4,
]

default_func2 = [
    *[square()] * 4,
    *[exp()] * 4,
    *[Trigon_sin()] * 4,
    *[Trigon_cos()] * 4,
    *[mul()] * 4,
    *[add()] * 4,
    *[sub()] * 4,
]