import csv

a = []
csvfile = open('2.csv')
reader = csv.reader(csvfile)
for row in reader:
    a.append(row)
    print(row)

num_attributes = len(a[0]) - 1

print("\nInitial Hypothesis: \n")
s = ['0'] * num_attributes
g = ['?'] * num_attributes
print("The most Specific is :", s)
print("The most General: ", g)

for j in range(0, num_attributes):    
    s[j] = a[0][j]

temp = []

print("\nCandidate Algorithm: ")
for i in range(0, len(a)):  
    if a[i][num_attributes] == 'yes':
        for j in range(0, num_attributes):
            if a[i][j] != s[j]:
                s[j] = '?'

        temp = [h for h in temp if all(h[j] == '?' or h[j] == s[j] for j in range(num_attributes))]

        print("For instance {0} Hypo S{0}:".format(i+1), s)
        if len(temp) == 0:
            print("For instance {0} Hypo G{0}:".format(i+1), g)
        else:
            print("For instance {0} Hypo G{0}:".format(i+1), temp)

    elif a[i][num_attributes] == 'no':
        for j in range(0, num_attributes):
            if s[j] != a[i][j] and s[j] != '?':
                new_g = ['?'] * num_attributes
                new_g[j] = s[j]
                temp.append(new_g)

        print("For instance {0} Hypo S{0}:".format(i+1), s)
        print("For instance {0} Hypo G{0}:".format(i+1), temp)
