import random
import string
import numpy 

# convenience function for saving data in the right format
def save_dat(bit_list, weight_list, matrix, outname): 
    """save parameter file in the right format

    Args:
        bit_list (lst): list of bit strings
        weight_list (lst): list of weights
        matrix (np.Array): matrix of 0, 1, -1 (just used for shape here)
        outname (lst): outname identifier
    """
    rows, cols = matrix.shape
    with open(f'{outname}', 'w') as f: 
        f.write(f'{rows}\n{cols}\n')
        for bit, weight in zip(bit_list, weight_list): 
            f.write(f'{bit} {weight}\n')
            
# convenience function for generating random identifier 
def randomword(length):
    """generate random string of letters

    Args:
        length (int): number of random letters to generate

    Returns:
        str: string of random letters   
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))