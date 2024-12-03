layers = [(16, 'linear')]
for i in range(len(layers)-1):
    print(layers[i][1])
print(layers[-1])