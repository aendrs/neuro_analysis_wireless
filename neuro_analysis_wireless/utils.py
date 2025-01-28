


def unify_categories(new_categories_map, labelsdict, labels, labelskeys, palette_dict=None):
    # Step 1: Create a reverse lookup from labelsdict (numeric -> key)
    reverse_labelsdict = {v: k for k, v in labelsdict.items()}
    
    # Step 2: Initialize the new labelsdict, labelskeys, palette_dict, and a mapping for labels
    new_labelsdict = {}
    label_mapping = {}
    new_palette_dict = {}
    
    # Step 3: Iterate over the new categories and create mappings based on the old categories and the new color
    for i, (new_category, category_info) in enumerate(new_categories_map.items()):
        # Extract old categories and color from category_info
        old_category_names = category_info['old_categories']
        new_color = category_info.get('color')  # Color specified for this new category
        
        # Add new category to labelsdict with new label (index i)
        new_labelsdict[new_category] = i
        
        # Map the old category names (strings) to the new category's label
        for old_category_name in old_category_names:
            # Safeguard: Ensure that the old category name exists in labelsdict
            if old_category_name in labelsdict:
                old_category_num = labelsdict[old_category_name]
                label_mapping[old_category_num] = i
        
        # Assign the specified color to the new category in new_palette_dict
        if new_color:
            new_palette_dict[new_category] = new_color
    
    # Step 4: Update the labels list by remapping according to label_mapping
    new_labels = [label_mapping[label] for label in labels]
    
    # Step 5: Update the labelskeys list with the new unified category names
    new_labelskeys = [reverse_labelsdict[label] for label in labels]  # Get old labels first
    new_labelskeys = [list(new_categories_map.keys())[new_labels[i]] for i in range(len(new_labels))]
    
    return new_labelsdict, new_labels, new_labelskeys, new_palette_dict

'''
# Example usage
labelsdict = {'Grasping_foraging_-500_500': 0,
              'Grasping_grooming_-500_500': 1,
              'Grasping_self_grooming_-500_500': 2,
              'Mouth_foraging_-500_500': 3,
              'Mouth_grooming_-500_500': 4,
              'Spread_foraging_-500_500': 5,
              'Spread_grooming_-500_500': 6,
              'Spread_self_grooming_-500_500': 7}

labels = [0, 1, 2, 3, 0, 4, 5, 6, 2]
labelskeys = ['Grasping_foraging_-500_500', 'Grasping_grooming_-500_500', 
              'Grasping_self_grooming_-500_500', 'Mouth_foraging_-500_500',
              'Grasping_foraging_-500_500', 'Mouth_grooming_-500_500',
              'Spread_foraging_-500_500', 'Spread_grooming_-500_500', 
              'Grasping_self_grooming_-500_500']

# Define new categories (with old categories and custom colors)
new_categories_map = {
    'A': {
        'old_categories': ['Grasping_foraging_-500_500', 'Grasping_grooming_-500_500'],
        'color': '#ff0000'  # Red color
    },
    'B': {
        'old_categories': ['Grasping_self_grooming_-500_500', 'Mouth_foraging_-500_500'],
        'color': '#00ff00'  # Green color
    },
    'C': {
        'old_categories': ['Mouth_grooming_-500_500', 'Spread_foraging_-500_500', 'Spread_grooming_-500_500', 'Spread_self_grooming_-500_500'],
        'color': '#0000ff'  # Blue color
    }
}

# Unify the categories and update the mappings
new_labelsdict, new_labels, new_labelskeys, new_palette_dict = unify_categories(
    new_categories_map, labelsdict, labels, labelskeys
)

# Output the updated structures
print("New labelsdict:", new_labelsdict)
print("New labels:", new_labels)
print("New labelskeys:", new_labelskeys)
print("New palette_dict:", new_palette_dict)
'''



def remove_substring(string_list, substring):
    return [s.replace(substring, '') for s in string_list]



def add_suffix_to_keys(dictionary, suffix):
    return {f"{key}{suffix}": value for key, value in dictionary.items()}



def get_labelsdict_palette_SamovarJanuary2025():
    labelsdict = {
        'Grasping_foraging': 0,
        'Grasping_grooming': 1,
        'Grasping_self_grooming': 2,
        'Mouth_foraging': 3,
        'Mouth_grooming': 4,
        'Mouth_self_grooming':5,
        'Spread_foraging': 6,
        'Spread_grooming': 7,
        'Spread_self_grooming': 8#,'Resting_state':9
    } 
   
    palette_dict = {
        'Grasping_foraging': '#0d44d1',       # Blue
        'Grasping_grooming': '#a4ff78',       # Light neon green
        'Grasping_self_grooming': '#ff7f0e',  # Orange
        'Mouth_foraging': '#d62728',          # Red
        'Mouth_grooming': '#2ca02c',          # Green
        'Mouth_self_grooming': '#00d0ff',     #baby blue
        'Spread_foraging': '#9b54d1',         # Purple
        'Spread_grooming': '#ffd500',         # Yellow
        'Spread_self_grooming': '#ef3e9c',     # Pink
        'Resting_state':'#757575'             # gray
    }
    return labelsdict, palette_dict



def get_array_folders_january2025(arrayselection, monkey):
    
    if arrayselection==1:
        #  All arrays ----------------------------------------------------------------
        arrayname='All Arrays'
        folders=[f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\45a",
                 f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\46v",
                 f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\F5hand",
                 f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\F5mouth"]
        
    elif arrayselection==2:
        #  PreFrontal  ----------------------------------------------------------------
        arrayname='Pre-Frontal'
        folders=[f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\45a",
                 f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\46v"]
        
    elif arrayselection==3:
        #  PreMotor   ----------------------------------------------------------------
        arrayname='Pre-Motor'
        folders=[f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\F5hand",
                 f"C:\\Users\\amendez\\Documents\\Jacopo\\data_jan2025\\Singleunit\\{monkey}\\pooled_activity\\F5mouth"]
        
    else:
        raise ValueError(f"Invalid arrayselection value: {arrayselection}. Please choose 1 (All Arrays), 2 (Pre-Frontal), or 3 (Pre-Motor).")
    
    return arrayname, folders
