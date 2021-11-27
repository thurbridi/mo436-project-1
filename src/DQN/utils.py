import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

def make_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def OH(value, size):
    x = torch.LongTensor([[value]])
    one_hot = torch.FloatTensor(1, size)
    return one_hot.zero_().scatter_(1, x, 1)


def plot_results(show, save, folder, scale, comb, **kwargs):
    window = int(scale / 10)

    plt.figure(figsize=[5, 10])
    j = 1
    for i in kwargs.keys():
        if i != "Epsilon":
            plt.subplot(len(kwargs.keys()), 1, j)
            j += 1
            plt.plot(pd.Series(kwargs[i]).rolling(window).mean())
            plt.title('{} Moving Average ({}-episode window)'.format(i, window))
            plt.ylabel('Moves')
            plt.xlabel('Episode')
        else:
            plt.subplot(len(kwargs.keys()), 1, j)
            plt.plot(kwargs[i])
            plt.title('Random Action Parameter')
            plt.ylabel('Chance Random Action')
            plt.xlabel('Episode')
        
        if ( i=='Rewards'):
            plt.ylim(-1,1)

        if ( i=='Loss'):
            plt.ylim(0,0.05)

    plt.tight_layout(pad=2)
    
    if (save==True):
       plt.savefig(f'{folder}/plot_results_{comb}') 

    if (show==True):
        plt.show()
    

def plot_result_lists(show, save, folder, comb, **kwargs ):
    plt.figure(figsize=[5, 10])
    j = 1
    for i in kwargs.keys():
        plt.subplot(len(kwargs.keys()), 1, j)
        j += 1
        plt.plot(kwargs[i])      
        plt.title('Total {}'.format(i))
        plt.ylabel(str(i))
        if ( i=='Rewards'):
            plt.ylim(-1,1)

        if ( i=='Loss'):
            plt.ylim(0,0.05)
            
        plt.xlabel('Episode')
    
    plt.tight_layout(pad=2) 

    if (save==True):
       plt.savefig(f'{folder}/plot_{comb}') 

    if (show==True):
        plt.show()

