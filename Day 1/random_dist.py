#                                   OBJECTIVE


#   Generate Random numbers between [1-100] for {1K, 10K, 100K, 1M, 10M} times and
#   verify if the Distribution is uniform.

#   Measure and plot the MAE to check if the error decreases with increase in sample size


# -----------------------------------------------------------------------------------------------


#   Importing required libraries
import random
import matplotlib.pyplot as plt


# Random number generator for perticular size
def random_num_generator(limit):
    batch = []
    for i in range(limit):
        batch.append(random.randint(0, 100))
    return batch


def main(limit, new_dict={}):                # Creating the freq dict [ 0-100 ]
    random_batch = random_num_generator(limit)

    for i in range(101):                       # Iterating values from 0 - 100
        # Counting the freq of each numbers
        new_dict[i] = random_batch.count(i)

    diff = [abs(1-(i/(limit/100)))
            for i in new_dict.values()]  # Calculating the MAE
    mae = sum(diff)/100
    return mae, new_dict


if __name__ == '__main__':

    # Sample size in range {1K, 10K, 100K, 1M, 10M}
    sample_size = [10**3, 10**4, 10**5, 10**6, 10**7]

    labels = ['1K', '10K', '100K', '1M', '10M']

    mae_list = []
    # Plotting the dist for all the sample size
    for s in range(len(sample_size)):
        mae, freq_count = main(sample_size[s])
        mae_list.append(mae)
        plt.bar(freq_count.keys(), freq_count.values())
        plt.xlabel('Values [ 0 - 100 ]')
        plt.ylabel('Sample Size')
        plt.title(f'Random number Distribution for {labels[s]} samples')
        plt.show()

    # Plotting MAE for the samples size
    plt.plot(labels, mae_list)
    plt.xlabel('Sample Size')
    plt.ylabel('MAE values')
    plt.title('MAE decreases with increase in sample size')
    plt.show()
