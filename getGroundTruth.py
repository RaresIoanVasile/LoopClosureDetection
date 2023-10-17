import matplotlib.pyplot as plt

matrix_size = (388, 388)

binary_matrix = [[0] * matrix_size[1] for _ in range(matrix_size[0])]

with open("results.txt", "r") as file:
    for line in file:
        i, j = map(int, line.strip().split())
        binary_matrix[i][j] = 1

fig, ax = plt.subplots()

ax.imshow(binary_matrix, cmap="binary_r")

ax.set_xticks(range(len(binary_matrix[0])))
ax.set_yticks(range(len(binary_matrix)))
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.show()