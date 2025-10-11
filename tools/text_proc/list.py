# json.load idea400_light_GPT4V_v0.json
import json

# Specify the file path
text_path =  "inputs/wham_data/idea400_light_GT4V_v0.json"
# Open the file in read mode
with open(text_path, 'r') as file:
    # Load the JSON data
    text_data = json.load(file)
word_dict = {}
total_words = 0
for key in text_data.keys():
    #print(key)
    #print(text_data[key])
    text = text_data[key]
    for word in text.split():
        total_words += 1
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
            print(f"new word: {word}")

#  Sort the dictionary by value
sorted_word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
# Print the sorted dictionary one word one line
for key in sorted_word_dict.keys():
    #print(f"{key}: {sorted_word_dict[key]}")
    if sorted_word_dict[key] < 3:
        break
# Break down the statistic in dictionary
print(f"Total Vocabulary: {len(sorted_word_dict)}")
print(f"Total Words: {total_words}")
# Print the top 10 words
print("Top 10 words:")
for key in list(sorted_word_dict.keys())[:10]:
    print(f"{key}: {sorted_word_dict[key]}")
# Print all the activities
print("All activities:")
# Find some words in the dictionary




    
    