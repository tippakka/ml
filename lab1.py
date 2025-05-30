import csv
num_attributes = 6
a = []
print("Given Data is: ")
csvfile = open('1.csv')
reader = csv.reader(csvfile)
for row in reader:
    a.append(row)
    print(row)  
    
print("The initial Values is: ")
hypo = ['0'] * num_attributes
print(hypo)

for j in range(0, num_attributes):
    hypo[j] = a[0][j]

for i in range(0, len(a)):
    if a[i][num_attributes] == 'yes':
        for j in range(0, num_attributes):
            if a[i][j] == hypo[j]:
                hypo[j] = '?'
            else:
                hypo[j] = a[i][j]
                
    print("For training Instance no :", i, "Hypo is", hypo)
print("The max specific hypo is:", hypo)