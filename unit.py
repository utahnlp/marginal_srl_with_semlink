import torch 

def replace_with_duplicated(mat):
 """ replace a nonzero integer by another randomly in each row

  sample 2 position indices from each row based on the sampling_weight. The first index position is going to be replaced by a duplicated element. The duplicated element is at the second index position. 
 """
 #mat = torch.tensor([[3,0,0,2,0,3], [3,0,1,0,0,5]])

 # make sure each row as at least two valid elements.
 # for those without at least two, replace it with all-one temporarily
 valid_mask = (mat>0).sum(-1, keepdim=True) > 1
 mat_legit = torch.where(valid_mask, mat, 1)

 sampling_weight = (mat_legit > 0).float() # sampling probabolity. If an element is nonzero then it has sampling weight of 1 otherwise 0.

 sampled_indices = torch.multinomial(sampling_weight, 2) 
 
 src_idx =  sampled_indices[:,1] # indices to be duplicated 
 tgt_idx = sampled_indices[:,0] # indices to be replaced by the duplicated values
 
 # replace with duplicated values
 mat_legit[torch.arange(mat.size()[0]), tgt_idx] = mat_legit[torch.arange(mat.size()[0]), src_idx]
 result = torch.where(valid_mask, mat_legit, mat)
 return result


mat = torch.tensor([[1,0,0,2,0,3], [0,4,0,1,0,5], [0,0,0,1,0,0]])
print(mat)
rs = replace_with_duplicated(mat)
print(rs)