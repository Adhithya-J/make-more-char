import torch
import matplotlib.pyplot as plt
import numpy as np

FILE_PATH = r".\data\names.txt"

def main():
    words = open(FILE_PATH, "r",encoding="utf8").read().splitlines()
    # print(words[:10])
    
    chars = sorted(list(set("".join(words)))) # unique list of chars from names # tokenization
    stoi = {s: i+1 for i,s in enumerate(chars)} # attaching a id to each character (encoding)
    stoi['.'] = 0

    n = len(stoi) # vocab size
    N = torch.zeros((n,n), dtype=torch.int32) # creating a n*n matrix
    

    itos = {i:s for s,i in stoi.items()} # reverse mapping (decoding)

    for w in words:
        chs = ["."] + list(w) + ["."]
        for chr1, chr2 in zip(chs, chs[1:]): # how can you iterate when both are of different size? # It stops at the shortest length so <E> from 1st has no pair and not added
            idx1, idx2 = stoi[chr1], stoi[chr2]
            N[idx1, idx2] += 1
            
    def display_freqmatrix(mat, itos):
        fig, ax = plt.subplots(figsize=(20,20))
        im = ax.imshow(mat,cmap="Blues")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                chstr = itos[i]+itos[j]
                plt.text(j,i, chstr, ha="center", va="bottom",color="gray",fontsize=8)
                plt.text(j,i, mat[i,j].item(), ha="center", va="top",color="gray",fontsize=6)
        plt.axis("off")
        plt.tight_layout()
        # ax.set_xlabel("Chr2")
        # ax.set_ylabel("Chr1")
        # labels = list(itos.values())
        # ax.set_xticks(np.arange(mat.shape[1]))
        # ax.set_yticks(np.arange(mat.shape[0]))
        # ax.set_xticklabels(labels)
        # ax.set_yticklabels(labels)
        # plt.colorbar(im, ax=ax)
        plt.show()

    # display_freqmatrix(N, itos=itos)
    
    g = torch.Generator().manual_seed(42)

    def sample_from_tensor(t):
        t = t.float()
        t /= t.sum() #  actually it would make sense to normalize the vector to avoid recalculations
        idx = torch.multinomial(t, 1,replacement=False, generator=g)
        return idx.item()
    
    result = ""
    idx = 0
    while True:
        result += itos[idx]
        idx = sample_from_tensor(N[idx])
        if idx == 0:
            break
        
    print(result)
    
if __name__ == "__main__":
    main()