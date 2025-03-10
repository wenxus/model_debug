

#%%
import goodfire
import os
GOODFIRE_API_KEY = os.environ.get("GOODFIRE_API_KEY")
client = goodfire.Client(api_key=GOODFIRE_API_KEY)
# Instantiate a model variant
#%%
variant = goodfire.Variant("meta-llama/Meta-Llama-3.1-8B-Instruct") #meta-llama/Meta-Llama-3.1-8B-Instruct") # meta-llama/Llama-3.3-70B-Instruct
#%%
prompt = "How many r are in berry?"
prompt = "how did you tokenize this: '"+prompt+"'"
print(prompt)
for token in client.chat.completions.create(
    [{"role": "user", "content": prompt}],
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    stream=True,
    max_completion_tokens=100,
):
    print(token.choices[0].delta.content, end="")
print()
prompt2 = "How many R are in BERRY?"
prompt2 ="how did you tokenize this: '"+prompt2+"'"
print(prompt2)
for token in client.chat.completions.create(
    [{"role": "user", "content": prompt2}],
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    stream=True,
    max_completion_tokens=100,
):
    print(token.choices[0].delta.content, end="")


#%%
# edits = client.features.AutoSteer( # 'FeaturesAPI' object has no attribute 'AutoSteer'
#     specification="be funny",  # or your desired behavior
#     model=variant,
# )
# variant.set(edits)
# print(edits)

# for token in client.chat.completions.create(
#     [{"role": "user", "content": "How many Rs are in STRAWBERRY?"}],
#     model=variant,
#     stream=True,
#     max_completion_tokens=100,
# ):
#     print(token.choices[0].delta.content, end="")
#%%
# for token in client.chat.completions.create(
#     [{"role": "user", "content": "How many Rs are in STRAWBERRY?"}],
#     model=variant,
#     stream=True,
#     max_completion_tokens=100,
# ):
#     print(token.choices[0].delta.content, end="")
# #%%
# cntletter_conversation = [
#     [
#         {
#             "role": "user",
#             "content": "How many Rs are in STRAWBERRY?"
#         },
#         {
#             "role": "assistant",
#             "content": "There are 2 Rs in the word STRAWBERRY."
#         }
#     ]
# ]
# # variant.reset()
# context = client.features.inspect( # does not work
#     messages=cntletter_conversation[0],
#     model=variant,
# )
# context
# %%
dataset_1 = [[
    {"role": "user", "content": "How many Rs are in BERRY?"},
    {"role": "assistant", "content": "There's 2 R's in BERRY."}
]]
dataset_2 = [[
    {"role": "user", "content": "How many rs are in berry?"},
    {"role": "assistant", "content": "I'm not sure what you mean by \"rs in berry.\""}
]]

formal_features, informal_features =  client.features.contrast(
    dataset_1=dataset_1,
    dataset_2=dataset_2,
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    top_k=5
)

print(f"{formal_features=}")
print(f"{informal_features=}")
# %%
countl_features = client.features.search(
    "Discussions about counting or verifying the number of letters in words.", #"Counting letters or characters in words and strings",
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    top_k=3
)
print(countl_features)

#%%
countl_features2 = client.features.search(
    "Counting or verifying quantities, especially letters in words",
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    top_k=3
)
print(countl_features2)
#%%
confused_features = client.features.search(
    "The assistant needs clarification",
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    top_k=3
)
print(confused_features)
# %%
variant.reset()
# %%
 #.5y .65n
variant.reset()
variant.set(countl_features[0], 1)
variant.set(countl_features2[0], 1)
variant.set(confused_features[0], -1)
for token in client.chat.completions.create(
    [{"role": "user", "content": "How many Rs are in BERRY?"}],
    model=variant,
    stream=True,
    max_completion_tokens=100,
):
    print(token.choices[0].delta.content, end="")

print(";")
for token in client.chat.completions.create(
    [{"role": "user", "content": "How many rs are in berry?"}],
    model=variant,
    stream=True,
    max_completion_tokens=100,
):
    print(token.choices[0].delta.content, end="")
#%%

# %%
# variant.reset()
# for token in client.chat.completions.create(
#     [{"role": "user", "content": "How many Rs are in BERRY?"}],
#     model=variant,
#     stream=True,
#     max_completion_tokens=100,
# ):
#     print(token.choices[0].delta.content, end="")
# %% # steering makes originally correct answer wrong
# variant.set(countl_features[1], 9)
# for token in client.chat.completions.create(
#     [{"role": "user", "content": "How many Rs are in BERRY?"}],
#     model=variant,
#     stream=True,
#     max_completion_tokens=100,
# ):
#     print(token.choices[0].delta.content, end="")
# %%
def contrast(a1, a2, model=variant, q1="How many Rs are in BERRY?", q2="How many rs are in berry?"):
    dataset_1 = [[
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1}
    ]]
    dataset_2 = [[
        {"role": "user", "content": q2},
        {"role": "assistant", "content": a2}
    ]]

    formal_features, informal_features =  client.features.contrast(
        dataset_1=dataset_1,
        dataset_2=dataset_2,
        model=model,
        top_k=5
    )
    print(f"{formal_features=}")
    print(f"{informal_features=}")
# %%
q1="How many Rs are in BERRY?"
q2="How many rs are in berry?"
a1 = "The word BERRY has 3 Rs in it."
a2 = "That's a clever play on words. \"BerrY\" has 4 Rs."
def ds(q, a):
    return [[
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}
    ]]
contrast(a1, a2)
# %%
dataset_1

# %%
# Get activation matrix for a conversation
# matrix =  client.features.activations(
#     messages=dataset_1[0],
#     model="meta-llama/Llama-3.3-70B-Instruct"
# )
# print(matrix)
# Analyze how features activate in text
inspector = client.features.inspect(
    dataset_1[0],
    model="meta-llama/Llama-3.3-70B-Instruct"
)

# Get top activated features
for activation in inspector.top(k=5):
    print(f"{activation.feature.label}: {activation.activation}")# %%

# %%

# %%
def print_top_activations(convo, model=variant, k=5):
    inspector = client.features.inspect(
        convo,
        model=model
    )

    # Get top activated features
    for activation in inspector.top(k=5):
        print(f"{activation.feature.label}: {activation.activation}")# %%
#%%
print_top_activations(ds(q1,a1)[0])
# %%
print_top_activations(dataset_2[0], model='meta-llama/Meta-Llama-3.1-8B-Instruct' )
# %%
print_top_activations(ds(q2,a2)[0])
# %%
print_top_activations(dataset_1[0], model='meta-llama/Meta-Llama-3.1-8B-Instruct')

# %%
f1l = list(range(0,105,5))+[200]
f2l = list(range(0,105,5))+[200]
# r = ""
r = "f0;f1;f2;answer_to_q1;answer_to_q1_is_correct;answer_to_q2;answer_to_q2_is_correct\n"
for f0 in [-1, 2]: #confused
    for f1 in f1l:
        for f2 in f2l:
            variant.reset()
            if f0 == -1:
                variant.set(confused_features[0], -1)
            if f1/100 < 1.1:
                variant.set(countl_features[0], f1/100)
            if f2/100 < 1.1:
                variant.set(countl_features[0], f2/100)
            line=f"{f0}; {f1/100}; {f2/100};"
            # print("answer to q1= ")
            answer1=""
            for token in client.chat.completions.create(
                [{"role": "user", "content": "How many Rs are in BERRY?"}],
                model=variant,
                stream=True,
                max_completion_tokens=20,
            ):
                answer1+=token.choices[0].delta.content
                # print(token.choices[0].delta.content, end="")
            
            line+=answer1+";"+ ("1" if ("2" in answer1 or "two" in answer1) else "0")+";"
            answer2=""
            for token in client.chat.completions.create(
                [{"role": "user", "content": "How many rs are in berry?"}],
                model=variant,
                stream=True,
                max_completion_tokens=20,
            ):
                answer2+=token.choices[0].delta.content
            # print(answer2)
            line += answer2+";"
            line += "1" if ("2" in answer2 or "two" in answer2) else "0"
                # print(token.choices[0].delta.content, end="")
            # print(line)
            line.replace("\n", " ").replace("\r", " ").replace("\r\n", " ")
            r+=f"{line}\n"
    # r.replace("\n", " ")
    # r.replace("\t", " ")

    # create file if doesnt exist and wrie to it 

    with open("/Users/wenx/Downloads/feature_steer_sweep_all.csv", "w") as f:
        f.write(r + "\n")
#%%
# Open the source file for reading and the destination file for writing
with open("/Users/wenx/Downloads/feature_steer_sweep_all.csv", "r") as infile, open("/Users/wenx/Downloads/feature_steer_sweep_all_processed.csv", "w") as outfile:
    for line in infile:
        if line.startswith("-1") or line.startswith("2") or line.startswith("f0"):
            # line.replace("2;", "No Steering;")
            outfile.write(line)  # Write each line to the output file


# %%
# print(r)
# # %%
# from transformers import AutoTokenizer

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# # Example text
# text = "Hello, how are you today?"

# # Tokenize the text
# tokens = tokenizer(text, return_tensors="pt")

# # Print tokenized output
# print("Tokenized IDs:", tokens['input_ids'])
# print("Decoded Tokens:", tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Read the CSV file
df = pd.read_csv('/Users/wenx/Downloads/feature_steer_sweep_all_processed.csv', sep=';')

# Convert correctness columns to boolean/numeric
df['answer_to_q1_is_correct'] = df['answer_to_q1_is_correct'].map({'True': 1, 'true': 1, True: 1, 1: 1, 'False': 0, 'false': 0, False: 0, 0: 0})
df['answer_to_q2_is_correct'] = df['answer_to_q2_is_correct'].map({'True': 1, 'true': 1, True: 1, 1: 1, 'False': 0, 'false': 0, False: 0, 0: 0})

# Fill NaN values with 0 (incorrect)
df['answer_to_q1_is_correct'] = df['answer_to_q1_is_correct'].fillna(0)
df['answer_to_q2_is_correct'] = df['answer_to_q2_is_correct'].fillna(0)

# Group by f1 and f2, calculate accuracy for Q1 and Q2
grouped = df.groupby(['f1', 'f2']).agg({
    'answer_to_q1_is_correct': 'mean',
    'answer_to_q2_is_correct': 'mean',
    'f0': 'first'  # Since f0 is constant (-1) across all data
}).reset_index()

# Rename columns for clarity
grouped = grouped.rename(columns={
    'answer_to_q1_is_correct': 'q1_accuracy',
    'answer_to_q2_is_correct': 'q2_accuracy'
})

# Set up the figure for two subplots
plt.figure(figsize=(15, 8))

# Create a custom colormap (white to blue)
cmap = LinearSegmentedColormap.from_list('white_to_blue', ['#FFFFFF', '#0000FF'])

# Get unique values for f1 and f2
f1_values = sorted(grouped['f1'].unique())
f2_values = sorted(grouped['f2'].unique())

# Create meshgrid for heatmap
f1_mesh, f2_mesh = np.meshgrid(f1_values, f2_values)

# Create empty matrices for accuracies
q1_accuracy_matrix = np.zeros((len(f2_values), len(f1_values)))
q2_accuracy_matrix = np.zeros((len(f2_values), len(f1_values)))

# Fill the matrices with data where available
for i, f1 in enumerate(f1_values):
    for j, f2 in enumerate(f2_values):
        data_point = grouped[(grouped['f1'] == f1) & (grouped['f2'] == f2)]
        if not data_point.empty:
            q1_accuracy_matrix[j, i] = data_point['q1_accuracy'].values[0]
            q2_accuracy_matrix[j, i] = data_point['q2_accuracy'].values[0]
        else:
            # Use NaN for missing data points
            q1_accuracy_matrix[j, i] = np.nan
            q2_accuracy_matrix[j, i] = np.nan

# Plot Q1 accuracy
plt.subplot(1, 2, 1)
im1 = plt.pcolormesh(f1_mesh, f2_mesh, q1_accuracy_matrix, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(im1, label='Accuracy')
plt.title('Q1 Answer Correctness (f1 vs f2)')
plt.xlabel('f1 Value')
plt.ylabel('f2 Value')

# Annotate the Q1 plot with accuracy values
for i, f1 in enumerate(f1_values):
    for j, f2 in enumerate(f2_values):
        if not np.isnan(q1_accuracy_matrix[j, i]):
            text_color = 'white' if q1_accuracy_matrix[j, i] > 0.5 else 'black'
            plt.text(f1, f2, f'{q1_accuracy_matrix[j, i]:.2f}', 
                     ha='center', va='center', color=text_color, fontsize=7)

# Plot Q2 accuracy
plt.subplot(1, 2, 2)
im2 = plt.pcolormesh(f1_mesh, f2_mesh, q2_accuracy_matrix, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(im2, label='Accuracy')
plt.title('Q2 Answer Correctness (f1 vs f2)')
plt.xlabel('f1 Value')
plt.ylabel('f2 Value')

# Annotate the Q2 plot with accuracy values
for i, f1 in enumerate(f1_values):
    for j, f2 in enumerate(f2_values):
        if not np.isnan(q2_accuracy_matrix[j, i]):
            text_color = 'white' if q2_accuracy_matrix[j, i] > 0.5 else 'black'
            plt.text(f1, f2, f'{q2_accuracy_matrix[j, i]:.2f}', 
                     ha='center', va='center', color=text_color, fontsize=7)

# Add a note about f0 value
plt.figtext(0.5, 0.01, f'Note: f0 value is constant at {grouped["f0"].iloc[0]} for all data points', 
            ha='center', fontsize=12, bbox=dict(facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Add space at the bottom for the note
plt.savefig('feature_matrix_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
