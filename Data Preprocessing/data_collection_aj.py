
# this code read files from multiple folders
# write file name in excel file in 'path'
# write 'condition' label in the excel file

#read files from multiple folders
# if the file name have 'cropped' in it, then check it the string has particular sow or gilt,  then write it in the excel file within the column 'Path' and 'Condition'
#for example, if 'the file name has 'YF12595' then in condition column write 1
 
import os
import glob
import pandas as pd

root_dir = r'./videos/'  

# labels according to sow and gilt
keywords_conditions_sow_gilt = {
    #B1
    'GF173':0, 'GF187':0, 'OF143':1, 'GF176':0,
    'RF28':0, 'RF33':1, 'RF34':0, 'RF36':0, 'RF40':1, 'RF50':1, 
    'RF52':1, 'RF53':1, 'RF56':0, 'RF59':0, 'RF60':0, 'RF61':1,
    #B2
    'YF12595': 1, 'YF12749': 0, 'YF12706': 0,
    'RF72' : 1, 'RF73' : 1, 'RF74' : 1, 'RF75' : 1, 'RF78' : 1, 'RF79' : 1, 
    'RF83' : 0, 'RF85' : 0, 'RF86' : 0, 'RF88' : 0, 'RF91' : 0, 'RF92' : 0,
    #B3
    'YF12811' : 1, 'YF12825' : 0, 'YF12876' : 0, 'YF12892' : 0, 
    'RF93' : 0, 'RF96' : 1, 'RF98' : 0,  'RF103' : 1, 'RF104' : 1, 'RF105' : 1, 
    'RF109' : 0, 'RF112' : 1,  'RF114' : 0, 'RF118' : 0, 'RF119' : 0, 'RF120' : 1,
    #B4    
    'OF62' : 1, 'OF174' : 1,
    'RF123':1, 'RF124':1, 'RF125':0, 'RF126':1, 'RF127':0, 'RF128':0, 
    'RF129':0, 'RF131':1, 'RF132':0,
    # B5
    'YF12825':0, 'YF12876':0,
    'RF133':1, 'RF135':1, 'RF136':1, 'RF138':0, 'RF139':0, 'RF140':0, 
    #B6
    'YF12447':0, 'YF12612':0, 'YF12666':0, 'YF13921':1, 'YF13922':1,
    'RF147':0, 'RF154':0, 'RF160':0, 'RF167':0, 'RF169':1, 'RF173':0, 'RF142':1, 'RF143':0, 'RF151':0,
    'RF152':0, 'RF155':0, 'RF163':1, 'RF158':1, 'RF165':1, 'RF170':0, 'RF174':1, 'RF175':1, 'RF177':1,
    #B7
    'YF12752':0, 'YF12750':1, 'YF12746':1,
    #'RF185':0,'RF186':1, 'RF187':1, 'RF189':0, 'RF198':1, 'RF202':1, 'RF188':1, 'RF191':1, 'RF196':1, 'RF197':0, 'RF203':0, 'RF205':0,
    'RF185':0,'RF186':1, 'RF187':1, 'RF189':1, 'RF198':1, 'RF202':0, 'RF188':0, 'RF191':1, 'RF196':1, 'RF197':1, 'RF203':0, 'RF205':0,

    #B8
    'YF13835':1, 'YF13730':0, 'YF13770':1, 'PF27':1,
    'RF218':0, 'RF219':0,'RF220':1, 'RF221':1, 'RF222':0, 'RF224':0, 'RF227':1, 'RF228':1, 'RF229':0, 'RF233':1, 'RF234':1, 'RF235':1
}


# labels according to sow (parent)
keywords_conditions_sow_sow = {
    #B1
    'GF173':0, 'GF187':0, 'OF143':1, 'GF176':0,
    'RF28':0, 'RF33':0, 'RF36':0, 'RF50':1, 'RF52':1, 'RF59':0, 'RF34':0, 'RF40':1, 'RF53':1, 'RF56':0, 'RF60':0, 'RF61':0,
    #B2
    'YF12595': 1, 'YF12749': 0, 'YF12706': 0,
    'RF72' : 1, 'RF73' : 1, 'RF74' : 1, 'RF75' : 1, 'RF78' : 1, 'RF79' : 1, 'RF83' : 0,
    'RF85' : 0, 'RF86' : 0, 'RF88' : 0, 'RF91' : 0, 'RF92' : 0,
    #B3
    'YF12811' : 1, 'YF12825' : 0, 'YF12876' : 0, 'YF12892' : 0, 
    'RF96' : 1,  'RF103' : 0, 'RF104' : 0, 'RF105' : 0, 'RF112' : 0, 'RF93' : 1,  'RF98' : 1, 'RF109' : 0,  'RF114' : 0, 'RF118' : 0,
    'RF119' : 0, 'RF120' : 0,
    #B4    
    'OF62' : 1, 'OF174' : 1,
    'RF123':1, 'RF124':1, 'RF125':1, 'RF126':1, 'RF127':1, 'RF128':1, 'RF129':1, 'RF131':1, 'RF132':1,
    # B5
    'YF12825':0, 'YF12876':0,
    'RF138':0, 'RF139':0, 'RF140':0, 'RF133':0, 'RF135':0, 'RF136':0,
    #B6
    'YF12447':0, 'YF12612':0, 'YF12666':0, 'YF13921':1, 'YF13922':1,
    'RF147':0, 'RF154':0, 'RF160':1, 'RF167':1, 'RF169':1, 'RF173':1, 'RF142':0, 'RF143':0, 'RF151':0,
    'RF152':0, 'RF155':0, 'RF163':1, 'RF158':1, 'RF165':1, 'RF170':1, 'RF174':0, 'RF175':0, 'RF177':0,
    #B7
    'YF12752':0, 'YF12750':1, 'YF12746':1,
    'RF185':0,'RF186':0, 'RF187':0, 'RF189':0, 'RF198':1, 'RF202':1, 'RF188':0, 'RF191':1, 'RF196':1, 'RF197':1, 'RF203':1, 'RF205':1,
    #B8
    'YF13835':1, 'YF13730':0, 'YF13770':1, 'PF27':1,
    'RF218':1, 'RF219':1,'RF220':1, 'RF221':0, 'RF222':0, 'RF224':0, 'RF227':1, 'RF228':1, 'RF229':1, 'RF233':1, 'RF234':1, 'RF235':1
}


# Initialize lists to hold the data for the DataFrames
data_gilt = []
data_sow = []
keyword_counts = {key: 0 for key in keywords_conditions_sow_sow.keys()}

# Walk through the directory and subdirectories to find .jpg files
for folder, subfolders, filenames in os.walk(root_dir):
    if 'piglet' in folder:
        continue  # Skip the folder if it is 'piglet'

    #if 'sow' in folder:
    #    continue  # Skip the folder if it is 'sow'

    for filename in filenames:
        if filename.endswith('.jpg') and 'cropped' in filename:
            file_path = os.path.join(folder, filename)
            condition = None
            # Check for the keywords in the filename
            for keyword, cond in keywords_conditions_sow_sow.items():
                if keyword in filename:
                    condition = cond
                    keyword_counts[keyword] += 1
                    break
            # Append the file path and condition to the appropriate list
            if condition is not None:
                if any(keyword.startswith('RF') for keyword in keywords_conditions_sow_sow if keyword in filename):
                    data_gilt.append({'Path': file_path, 'Condition': condition})
                else:
                    data_sow.append({'Path': file_path, 'Condition': condition})

# Create DataFrames from the collected data
df_gilt = pd.DataFrame(data_gilt, columns=['Path', 'Condition'])
df_sow = pd.DataFrame(data_sow, columns=['Path', 'Condition'])
df_keyword_counts = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Count'])

# Save the DataFrames to Excel files
output_file_gilt = 'data_gilt_sow_all.xlsx'
output_file_sow = 'data_sow_sow_all.xlsx'
output_file_keyword_counts = 'keyword_counts_sow_all.xlsx'

df_gilt.to_excel(output_file_gilt, index=False)
df_sow.to_excel(output_file_sow, index=False)
df_keyword_counts.to_excel(output_file_keyword_counts, index=False)

print(f"Excel files '{output_file_gilt}', '{output_file_sow}', and '{output_file_keyword_counts}' have been created successfully.")