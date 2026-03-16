import torch
import torch.nn as nn
from . import functions as fun

## Network - Initialization 0.5 + Mask
# Unary element
class Symnet_single(nn.Module):

    def __init__(self,row_w,fun,expmask,Nor=fun.L1Nor):
        super().__init__()
        self.par=1
        self.w= nn.Linear(row_w, 1, bias=False)  
        self.binary_w=None
        self.expmask=expmask
        self.initialize_params()
        self.fun=fun
        self.Nor=Nor
        self.index=1
        
    def initialize_params(self):
        # Initialize weights to 1
        with torch.no_grad():  # Disable gradient calculation
            self.w.weight.data = torch.normal(mean=0.5, std=0.01, size=self.w.weight.shape)
            self.w.weight.data *= self.expmask

    def Delete_small(self):
        fun.delete_small(self.w.weight)
        return None
        
    def sym_norm(self):
        #self.Nor(self.w.weight)
        with torch.no_grad():
            self.w.weight.copy_(nn.functional.softmax(self.w.weight, dim=1))
            # self.w.weight.copy_(torch.sigmoid(self.w.weight))
        return  None
    
    def forward(self, X):
        if self.fun.name=="exp":
            return torch.matmul(X,(self.w.weight * self.expmask).T)
        else:
            return self.fun.forward(self.w(X) )
        # return self.fun.forward(self.w(X) if self.fun.name!="exp" else  torch.matmul(X,(self.w.weight.data*self.expmask).T))
    
    def sparsification_net(self):
        return torch.abs(self.w.weight.data).sum()
    
# Binary element
class Symnet_double(nn.Module):

    def __init__(self,row_w, fun):
        super().__init__()
        self.par=2
        self.w1= nn.Linear(row_w, 1, bias=False)
        self.w2= nn.Linear(row_w, 1, bias=False)  
        self.binary_w1=None
        self.binary_w2=None
        self.initialize_params()
        self.fun=fun
        self.index=2        
        
    def initialize_params(self):
        # Initialize weights to 1
        with torch.no_grad():  # Disable gradient calculation
            self.w1.weight.data = torch.normal(mean=0.5, std=0.01, size=self.w1.weight.shape)
#             self.w2.weight.data = torch.normal(mean=0.5, std=0.1, size=self.w2.weight.shape)
            self.w2.weight.data = 1- self.w1.weight.data
    
    def Delete_small(self):
        fun.delete_small(self.w1.weight)
        fun.delete_small(self.w2.weight)
        return None 
    
    # If parameters are close, retrain
    def Re_train(self):
        with torch.no_grad():
            if torch.rand(1)<0.5:
                self.w1.weight.data = torch.normal(mean=0.5, std=0.01, size=self.w1.weight.shape)
                self.w1.double()
            else:
                self.w2.weight.data = torch.normal(mean=0.5, std=0.01, size=self.w2.weight.shape)
                self.w2.double()
            
    def sym_norm(self):
        with torch.no_grad():
            self.w1.weight.copy_(nn.functional.softmax(self.w1.weight, dim=1))
            self.w2.weight.copy_(nn.functional.softmax(self.w2.weight, dim=1))
        return  None
    
    def loss_cor(self):
        loss_cor = torch.abs(torch.dot(self.w1.weight.squeeze(), self.w2.weight.squeeze()))/torch.norm(self.w1.weight, p=2)/torch.norm(self.w2.weight, p=2)
        return loss_cor

    def forward(self, X):
        return self.fun.forward(self.w1(X),self.w2(X))

    def sparsification_net(self):
        return torch.abs(self.w1.weight.data).sum()+torch.abs(self.w2.weight.data).sum()    
    
# Single-layer SymNet sequential connection
class Symnet_Sequential(nn.Module):
    
    def __init__(self, idx,funcs,input,num_single,num_double,expmask):
        super().__init__()
        self.idx=idx
        self.funcs=funcs
        self.input=input
        self._modules_dict = nn.ModuleDict()  # Initialize _modules_dict
        self.num_single=num_single
        self.num_double=num_double
        self.expmask=expmask
        for i, func in enumerate(funcs):
            # Here, the module is an instance of a Module subclass. Save it in the 'Module' class members
            if func.dim==1:
                self._modules_dict[str(i)] = Symnet_single(self.input,func,self.expmask)
            else:
                self._modules_dict[str(i)] = Symnet_double(self.input,func)
                
    def __getitem__(self, index):
        return self._modules_dict[str(index)]
    
    # Forward pass
    def forward(self, X):
        Y=X
        for block in self._modules_dict.values():
            Y=torch.cat((Y,block(X)),dim=1)
        return Y
    
    # Set small decimals to zero
    def Delete_small(self):
        for block in self._modules_dict.values():
            block.Delete_small()
        return None
    
    # Retrain identical parameters - limit 0.0001
    def Re_train(self):
        for block in self._modules_dict.values():
            if block.par==2:
                if ((block.w1.weight-block.w2.weight)**2).max()<0.0001:
                    print("wuhu")
                    block.Re_train()
    
    # Calculate loss
    def loss_cor(self):
        loss=0
        for block in self._modules_dict.values():
            if block.index==2:
                loss+=block.loss_cor()
        return loss
    # Normalization method
    def sym_norm(self):
        for block in self._modules_dict.values():
            block.sym_norm()
        return None
    # Sparsification
    def sparsification_net(self):
        loss=0
        for block in self._modules_dict.values():
            loss+=block.sparsification_net()
        return loss

# Multi-layer network composition
# Count unary operators

class Symnet_all(nn.Module):
    
    def __init__(self,nums, symbolic_depth,funcs):
        super().__init__()
        self.symbolic_depth=symbolic_depth
        self.funcs=funcs
        self._modules_dict = nn.ModuleDict()  # Initialize _modules_dict
        self.featurenums=nums
        self.struct=[self.featurenums]
        self.struct_single=[self.featurenums]
        self.struct_double=[0]
        self.expmask=[torch.ones(nums)]
        for idx in range(symbolic_depth):
            self.struct.append(self.struct[idx]+len(self.funcs[idx]))   
            self.struct_single.append(self.count_single(self.funcs[idx]) )
            self.struct_double.append(len(self.funcs[idx])-self.struct_single[idx+1])
            self.expmask.append(torch.cat((self.expmask[-1], self.exp_mask(self.funcs[idx])), dim=0))
            # In the _modules variable. The type of _module is OrderedDict
            self._modules_dict[str(idx)] = Symnet_Sequential(idx,funcs[idx],self.struct[idx],self.struct_single[idx+1],self.struct_double[idx+1],self.expmask[idx])
        self.outnum=self.struct[-1]
        self.w= nn.Linear(self.outnum, 1, bias=True)   
          
    def __getitem__(self, index):
        return self._modules_dict[str(index)]
    
    def forward(self, X):
        for block in self._modules_dict.values():
            X=block(X)
        X=self.w(X)
        return X

    def Delete_small(self):
        for block in self._modules_dict.values():
            block.Delete_small()
        return None
    
    def sym_norm(self):
        for block in self._modules_dict.values():
            block.sym_norm()
        return None

    def Re_train(self):
        for block in list(self._modules_dict.values())[:-1]:
            block.Re_train()

    def loss_cor(self):

        loss=0
        for block in self._modules_dict.values():
            loss+=block.loss_cor()
        return loss
    
    def sparsification_net(self):
        loss=0
        for block in self._modules_dict.values():
            loss+=block.sparsification_net()
        loss+=self.w.weight.data.sum()
        return loss

    def count_single(self,funcs):
        i = 0
        for func in funcs:
            if func.dim==1:
                i += 1
        return i 
    
    def exp_mask(self,funcs):
        mask_tensor=torch.ones(len(funcs))
        for i,func in enumerate(funcs):
            if func.name=="exp":
                mask_tensor[i]=0

        return mask_tensor