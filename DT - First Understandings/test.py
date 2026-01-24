import numpy
import pandas as pd
from Tree import Node
from Filter import Filter

# Load Data
df = pd.read_csv('data.csv')

# Convert categorical data to numerical
countryReference = {"UK": 0, "USA": 1, "N": 2}
df['Nationality'] = df['Nationality'].map(countryReference)

goReference = {"YES": 1, "NO": 0}
df['Go'] = df['Go'].map(goReference)

x = 'Experience'
y = 'Age'

def doSplitLearn(df):
    for x_threshold in df[x].unique():
        for y_threshold in df[y].unique():
            print('x: ', x_threshold, 'y: ', y_threshold)
            tree = Node()
            tree.data = df

            filter_1 = Filter(x, float(x_threshold))
            filter_2 = Filter(y, float(y_threshold))
            filters = [filter_1, filter_2]
            tree = buildRootState(tree, filters, 0)

            left = tree.left
            right = tree.right

            left.entropy = getEntropy(left.sampleData)
            right.entropy = getEntropy(right.sampleData)
            tree.childIG = getInformationGain(tree, left, right)
            left.left.entropy = getEntropy(left.left.sampleData)
            right.right.entropy = getEntropy(right.right.sampleData)
            left.childIG = getInformationGain(tree, left.left, right.right)

            leftGo = " ".join(left.sampleData['Go'].astype(str))
            rightGo = " ".join(right.sampleData['Go'].astype(str))
            llGo = " ".join(left.left.sampleData['Go'].astype(str))
            rrGo = " ".join(right.right.sampleData['Go'].astype(str))

            print("Left Node Data: ", leftGo, "\nE: ", left.entropy)
            print("Right Node Data: ", rightGo, "\nE: ", right.entropy)
            print("Information Gain:", tree.childIG)
            print("Left-Left Node Data: ", llGo, "\nE: ", left.left.entropy if left.left else "No further split")
            print("Right-Right Node Data: ", rrGo, "\nE: ", right.right.entropy if right.right else "No further split")
            print("Information Gain:" , left.childIG)

def getEntropy(data):
    size = len(data)
    if size == 0:
        return 0
    x_prob = len(data[data['Go'] == 1])/size
    y_prob = len(data[data['Go'] == 0])/size
    if x_prob == 0:
        x_val = 0
    else:
        x_val = -x_prob * numpy.log2(x_prob)

    if y_prob == 0:
        y_val = 0
    else:
        y_val = -y_prob * numpy.log2(y_prob)
    return  x_val + y_val

def getInformationGain(parent, left, right):
    Ep = getEntropy(parent.sampleData)
    size = len(parent.sampleData)
    wLeft = len(left.sampleData['Go'].unique())
    wRight = len(right.sampleData['Go'].unique())

    return Ep - wLeft/size * left.entropy - wRight/size * right.entropy


# # Define thresholds for decision tree splitting
# age_threshold = 30
# experience_value = 10
#
# # Initialize root node
# tree = Node()
# tree.data = df  # Assign full dataset to root
#
# # Define filters
# filter_1 = Filter('Age', age_threshold)
# filter_2 = Filter('Experience', experience_value)
# filters = [filter_1, filter_2]


# Recursive function to build decision tree
def buildRootState(tree: Node, filters, count):
    if count < len(filters):  # Prevent index out-of-bounds
        tree.left = Node()
        tree.right = Node()

        tree.left.data = filterExperience(tree.data, filters[count], False)
        tree.right.data = filterExperience(tree.data, filters[count], True)

        # Recursively split the left and right subtrees
        buildRootState(tree.left, filters, count + 1)
        buildRootState(tree.right, filters, count + 1)

    return tree  # Return the tree after building


# Function to filter the dataset
def filterExperience(df, filter_obj: Filter, moreOrLess):
    if moreOrLess:
        return df[df[filter_obj.column] >= filter_obj.threshold]
    return df[df[filter_obj.column] <= filter_obj.threshold]


# Build the decision tree
#tree = buildRootState(tree, filters, 0)

doSplitLearn(df)




# ---- Visualization ----
# plt.figure(figsize=(8, 6))
#
# # Scatter plot of the data
# plt.scatter(df[df['Go'] == 1]['Age'], df[df['Go'] == 1]['Experience'], color='green', label="Go = YES")
# plt.scatter(df[df['Go'] == 0]['Age'], df[df['Go'] == 0]['Experience'], color='red', label="Go = NO")
#
# # Plot decision boundary for Age
# plt.axvline(x=age_threshold, color='blue', linestyle='--', label=f"Age ≤ {age_threshold}")
#
# # Plot decision boundary for Experience
# plt.axhline(y=experience_value, color='purple', linestyle='--', label=f"Experience ≤ {experience_value}")
#
# # Annotate decision nodes
# plt.text(age_threshold - 2, max(df['Experience']), f'Age ≤ {age_threshold}', fontsize=12, color='blue')
# plt.text(min(df['Age']), experience_value + 1, f'Experience ≤ {experience_value}', fontsize=12, color='purple')
#
# # Customize the plot
# plt.xlabel("Age")
# plt.ylabel("Experience")
# plt.title("DT - First Understandings Split Visualization")
# plt.legend()
# plt.grid(True)

# Show plot
#plt.show()
