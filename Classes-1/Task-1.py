import numpy as np
import matplotlib.pyplot as plt

#----- Our model: Kicked rotator
def set_model(x0, p0, K, steps):
    """
    Kicked rotator model (lecture)

    x0: initall phase [rad]
    p0: initall momentum []
    K: strength of kicking []
    steps: how many kikc we want - see lecture

    return: [x_list, p_list, kick_list], where:
             x_list: phase value after i-th kick
             p_list: momentum value after i-th kick
             kick_list: the kick index - for which kick we calculated the phase and momentum
    """
    # list containg all data what we need
    x_list = [x0]
    p_list = [p0]
    kick_list = [0]
    # initall values
    x = x0
    p = p0

    for i in range(1, steps):
        p = p + K * np.sin(x)
        x = (x + p) % (2 * np.pi)
        # appending
        x_list.append(x)
        p_list.append(p)
        kick_list.append(i)

    return [x_list, p_list, kick_list]

#----- Initall conditions
K = 1.2 # the strength of kicking, given in task

# first initall condition
x0_1 = 3
p0_1 = 1.9
kicks = 40

first_case = set_model(x0_1, p0_1, K, kicks)
x_list_1 = first_case[0]
p_list_1 = first_case[1]
kick_list_1 = first_case[2]

# second intall condition
x0_2 = 3
p0_2 = 1.8999
kicks = 40

second_case = set_model(x0_2, p0_2, K, kicks)
x_list_2 = second_case[0]
p_list_2 = second_case[1]
kick_list_2 = second_case[2]

#------ Plotting
# Plotting first initall condition
fig, ax = plt.subplots(2, 1, figsize=(6.0, 10.0))
ax[0].set_title('Kicked rotator model: first initall condition', fontsize=18)
ax[0].plot(kick_list_1, x_list_1, label="x")
ax[0].plot(kick_list_1, p_list_1, label="p")
# describe plot
ax[0].set_xlabel(r"number of kicks", fontsize=18)
ax[0].set_ylabel(r"value", fontsize=18)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].grid(True)
ax[0].legend()

# Plotting second initall condition
ax[1].set_title('Kicked rotator model: second initall condition', fontsize=18)
ax[1].plot(kick_list_2, x_list_2, label="x: phase")
ax[1].plot(kick_list_2, p_list_2, label="p: momentum")
# describe plot
ax[1].set_xlabel(r"number of kicks", fontsize=18)
ax[1].set_ylabel(r"value", fontsize=18)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].grid(True)
ax[1].legend()
# additional sets
fig.tight_layout()
# plt.savefig('Task-1-plot.pdf')
plt.show()


# plt.close(fig)
