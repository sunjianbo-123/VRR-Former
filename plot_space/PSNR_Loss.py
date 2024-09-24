import re
import matplotlib.pyplot as plt

# Load the data from file
with open("/home/sunjianbo/PycharmProjects/MyProjects/ReflectionRemoval/Uformer-main/logs/reflectionremoval/my_syn/Uformer_B_wo_perceptualloss/2024-04-30T01:36:15.242402.txt", "r") as file:
    data = file.readlines()

# Initialize lists to store loss values and epoch numbers
loss_values = []
epoch_numbers = []

# Initialize dictionary to store the final PSNR values for each epoch
psnr_per_epoch = {}

# Process the file data
for line in data:
    # Extract epoch and loss values
    loss_match = re.search(r"Epoch: (\d+)\s+Time: [\d.]+\s+Loss: ([\d.]+)", line)
    if loss_match:
        epoch_numbers.append(int(loss_match.group(1)))
        loss_values.append(float(loss_match.group(2)))

    # Extract PSNR values and update the dictionary to only store the last PSNR per epoch
    epoch_match = re.search(r"Ep (\d+)", line)
    psnr_match = re.search(r"PSNR SIDD: ([\d.]+)", line)
    if epoch_match and psnr_match:
        epoch = int(epoch_match.group(1))
        psnr_per_epoch[epoch] = float(psnr_match.group(1))

# Extract the final PSNR values in the order of epochs
final_psnr_values = [psnr_per_epoch[epoch] for epoch in epoch_numbers if epoch in psnr_per_epoch]

# Create a plot
fig, ax1 = plt.subplots()

# Plot PSNR on left y-axis
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('PSNR', color=color)
ax1.plot(epoch_numbers, final_psnr_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for the loss values
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)
ax2.plot(epoch_numbers, loss_values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Title and show plot
plt.title('PSNR and Loss per Epoch')
fig.tight_layout()
plt.show()


