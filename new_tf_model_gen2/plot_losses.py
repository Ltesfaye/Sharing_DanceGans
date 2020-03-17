from matplotlib import pyplot as plt
import datetime 

def plot_errors(generator, disriminator, display):
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.png'

    # plt.title("Final Model Training Losses vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    plt.plot(generator,'b',label='Generator Loss')
    plt.plot(disriminator,'r',label = 'Discriminator Loss')
    plt.legend(loc='best')
    plt.savefig(file_name)
    if display:
        plt.show()


