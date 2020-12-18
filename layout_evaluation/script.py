#!/usr/bin/python3

import os
import re
import sys
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt


def main():
    # asks user
    print("Welcome to the Keyboard Layout Evaluator & Generator:")
    eval_or_gen = input("Do you want to [e]valuate or [g]enerate? [e/g]: ")

    if eval_or_gen == "e":
        app_evaluate()
    elif eval_or_gen == "g":
        app_generate()
    else:
        print("Bibop! I don't know what to do! Closing...")
        quit()


def app_evaluate():
    """
    Evaluates layout in config.txt.
    This was implemented in github.com/bclnr/kb-layout-evaluation
    """
    # load the blocks from config.txt and parse them into dataframes
    df_layouts, df_keys, df_bigrams, df_penalties, keys, blocks = parse_config(load_config(), "e")

    # as some letters are not present in all layouts, they can be manually removed from the bigrams list
    letters_to_ignore = 'êàçâîôñäöüß/'
    # iterate over the dataframe to remove the letters
    for row in df_bigrams.itertuples():
        drop = False
        for c in letters_to_ignore:
            if str(c) in row.Index:
                drop = True
        if drop:
            df_bigrams = df_bigrams.drop(row.Index)

    # modify languages from theory to add the punctuation frequency from personal corpus
    # copy the "theory" numbers to a "no punctuation" column first
    df_bigrams['en_nopunctuation'] = df_bigrams['en']
    df_bigrams['fr_nopunctuation'] = df_bigrams['fr']
    punctuation = ".,-'/"
    for row in df_bigrams.itertuples():
        for c in punctuation:
            if str(c) in row.Index:
                df_bigrams.at[row.Index, 'en'] = df_bigrams.at[row.Index, 'en_perso']
                df_bigrams.at[row.Index, 'fr'] = df_bigrams.at[row.Index, 'fr_perso']

    # normalize df_bigrams to get 100% on each column
    df_bigrams = df_bigrams * 100 / df_bigrams.sum(axis=0)

    # this prints letters present in bigrams but not in a layout
    # letters absent from a layout do not count in the grade
    # differences between layouts skew the results
    df_missing_letters = check_missing_letters(df_layouts, df_bigrams)
    if 'Missing letters' in df_missing_letters:
        print('Some letters are missing from some layouts, skewing the results:')
        print(df_missing_letters)

    # generate a dataframe of the weights per bigram per layout
    df_bigram_weight = bigram_weight(df_layouts, df_keys, df_bigrams, df_penalties)

    # get the results
    df_results = layout_results(df_bigrams, df_bigram_weight)

    # add average column with arbitrary coefs per language
    # my average of 30/70 en fr
    df_results['Personal average'] = df_results.en * 0.6 + df_results.fr * 0.4

    # normalize the results based of Qwerty Personnal average
    df_results = df_results.applymap(lambda x: round(x/df_results.at['Qwerty', 'Personal average'] * 100, 2))
    # df_results = df_results.applymap(lambda x: round(x/df_results.at['Qwerty', 'en'] * 100, 2))

    # sort the results
    # df_results = df_results.sort_values(by=['en'], ascending=True)
    df_results = df_results.sort_values(by=['Personal average'], ascending=True)

    # filter/reorder the results
    df_results = df_results[['Personal average', 'en', 'en_perso', 'fr', 'fr_perso', 'es', 'de']] # results_full.png

    print(df_results)


def layout_list_to_str(name, layout_list):
    layout_str = f">>{name}\n"
    index = 1
    for symbol in layout_list:
        if index % 12 == 0:
            layout_str += f"{symbol}\n"
        else:
            layout_str += f"{symbol} "
        index += 1

    layout_str += "\n"
    return layout_str


def app_generate():
    """
    Generates layout based on swapping symbols
    from the first layout in config.txt until
    no swap can make the ergonomic score lower.
    """
    # load data
    df_layouts, df_keys, df_bigrams, df_penalties, keys, blocks = parse_config(load_config(), "g")

    # as some letters are not present in all layouts, they can be manually removed from the bigrams list
    letters_to_ignore = 'êàçâîôñäöüß/'
    # iterate over the dataframe to remove the letters
    for row in df_bigrams.itertuples():
        drop = False
        for c in letters_to_ignore:
            if str(c) in row.Index:
                drop = True
        if drop:
            df_bigrams = df_bigrams.drop(row.Index)

    # modify languages from theory to add the punctuation frequency from personal corpus
    # copy the "theory" numbers to a "no punctuation" column first
    df_bigrams['en_nopunctuation'] = df_bigrams['en']
    df_bigrams['fr_nopunctuation'] = df_bigrams['fr']
    punctuation = ".,-'/"
    for row in df_bigrams.itertuples():
        for c in punctuation:
            if str(c) in row.Index:
                df_bigrams.at[row.Index, 'en'] = df_bigrams.at[row.Index, 'en_perso']
                df_bigrams.at[row.Index, 'fr'] = df_bigrams.at[row.Index, 'fr_perso']

    # normalize df_bigrams to get 100% on each column
    df_bigrams = df_bigrams * 100 / df_bigrams.sum(axis=0)

    # bests
    best_score = 1000
    current_score = 100
    best_str = ""
    current_str = ""
    best_name = ""
    current_name = "BEAKL 19bis"
    times = 0

    while(current_score < best_score and times < 8):
        times += 1

        # if enter the loop then best score is current_score
        best_score = current_score
        best_str = current_str
        best_name = current_name

        # if not first iteration
        if current_score != 100:
            df_layouts = create_df_layouts(keys, create_layouts_str_for_swap(best_name, blocks))

        # this prints letters present in bigrams but not in a layout
        # letters absent from a layout do not count in the grade
        # differences between layouts skew the results
        df_missing_letters = check_missing_letters(df_layouts, df_bigrams)
        if 'Missing letters' in df_missing_letters:
            print('Some letters are missing from some layouts, skewing the results:')
            print(df_missing_letters)

        # generate a dataframe of the weights per bigram per layout
        df_bigram_weight = bigram_weight(df_layouts, df_keys, df_bigrams, df_penalties)

        # get the results
        df_results = layout_results(df_bigrams, df_bigram_weight)

        # add average column with arbitrary coefs per language
        # my average of 30/70 en fr
        df_results['Personal average'] = df_results.en * 0.4 + df_results.fr * 0.6

        # normalize the results based of Qwerty Personnal average
        df_results = df_results.applymap(lambda x: round(x/df_results.at['Qwerty', 'Personal average'] * 100, 2))

        # sort the results
        df_results = df_results.sort_values(by=['Personal average'], ascending=True)

        # filter/reorder the results
        df_results = df_results[['Personal average']]

        best_layout_name = df_results.index[0]
        best_layout_score = df_results.to_numpy()[0][0]
        best_layout_layout_list = df_layouts.get(best_layout_name).to_numpy()
        best_layout_as_str = layout_list_to_str(best_layout_name, best_layout_layout_list)

        current_score = best_layout_score
        current_str = best_layout_as_str
        current_name = best_layout_name
        blocks = best_layout_as_str
        print(blocks)

    print(best_score)
    print(best_str)


def load_config():
    """
    Load the config file and outputs its content as a list of tuples (title, block)
    """
    
    # load the whole file into a str
    filepath = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'config.txt')
    filehandle = open(filepath)
    filetext = filehandle.read()
    
    # remove the comments from the file
    filetext = removeComments(filetext)
    
    # find all the paragraphs
    parts = re.findall(r'\[(.*?)\]', filetext)
    
    # to cut into blocks
    parts_location = []
    blocks = []
    # find the paragraphs location, put it in a tuple (start, end) of title
    for t in parts:
        part_location = filetext.find('[' + t + ']')
        parts_location.append((part_location, filetext.find('\n', part_location) + 1))
    # put paragraphs into blocks list
    for i in range(len(parts_location)):
        if i < len(parts_location) - 1:
            blocks.append(filetext[parts_location[i][1]:parts_location[i + 1][0]])
        else:
            blocks.append(filetext[parts_location[i][1]:])
    
    # remove blank lines from the blocks
    for i in range(len(blocks)):
        blocks[i] = os.linesep.join([s for s in blocks[i].splitlines() if s])
    
    # return the blocks
    output = []
    for i in range(len(blocks)):
        output.append((parts[i], blocks[i]))
    return output


def parse_config(blocks, eval_or_gen):
    """
    Takes the list of tuples (title, block), and outputs the dataframes
    """
    
    # check that all the needed configuration is present
    blocks_needed = ['keys', 'weights', 'penalties', 'layouts']
    blocks_available = []
    for t in blocks:
        blocks_available.append(t[0])
    for t in blocks_needed:
        if t not in blocks_available:
            print('Missing block from config file: ' + t)
            sys.exit()
    
    # find the location of the keys in the blocks list
    for i in range(len(blocks)):
        if blocks[i][0] == 'keys':
            break
    # create list of keys
    keys = blocks[i][1].split()
    
    # find the location of the weights in the blocks list
    for i in range(len(blocks)):
        if blocks[i][0] == 'weights':
            break
    # convert weights to float
    weights_list = blocks[i][1].split()
    weights_list = [float(num) for num in weights_list]
    # create DataFrame of key weights
    df_keys = pd.DataFrame(weights_list, index=keys, columns=['weights'])
    # add columns finger and row
    df_keys['finger'] = None
    df_keys['keyrow'] = None
    # assign a letter per key corresponding to which finger is used
    for row in df_keys.itertuples():
        # pinky
        if int(row.Index[2:]) <= 2:
            df_keys.at[row.Index, 'finger'] = 'p'
        # ring
        elif int(row.Index[2:]) == 3:
            df_keys.at[row.Index, 'finger'] = 'r'
        # middle
        elif int(row.Index[2:]) == 4:
            df_keys.at[row.Index, 'finger'] = 'm'
        # index
        elif int(row.Index[2:]) >= 5:
            df_keys.at[row.Index, 'finger'] = 'i'
    # assign a number per key corresponding to which row it is
    for row in df_keys.itertuples():
        df_keys.at[row.Index, 'keyrow'] = int(row.Index[1:2])
    # df_keys is a dataframe of the keys definition (base weight, finger, and row)

    # find the location of the layouts in the blocks list
    for i in range(len(blocks)):
        if blocks[i][0] == 'layouts':
            break

    # name of first in block
    first_layout = blocks[i][1].split(">>")[1]
    name_block = first_layout.split("\n")[0]

    # create DataFrame of layouts
    if eval_or_gen == "e":
        df_layouts = create_df_layouts(keys, blocks[i][1])
    else:
        df_layouts = create_df_layouts(keys, create_layouts_str_for_swap(name_block, blocks[i][1]))
    # df_layouts is a dataframe of all predefined layouts to evaluate

    # dataframe of bigrams by language
    df_bigrams = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'stats.csv'), header=0, sep=',', index_col=0)
    # df_bigrams is a dataframe of all possible bigrams (aa, ab…) with their probability per language

    # find the location of the penalties in the blocks list
    for i in range(len(blocks)):
        if blocks[i][0] == 'penalties':
            break
    df_penalties = pd.read_csv(StringIO(blocks[i][1]), sep=',', header=0, index_col=0, skipinitialspace=True)
    # df_penalties is a dataframe of the penalties to add to bigrams, by finger and row jump

    return df_layouts, df_keys, df_bigrams, df_penalties, keys, blocks[i][1]


def check_missing_letters(df_layouts, df_bigrams):
    """
    Return a list of letters present in bigrams but missing from each layout
    """
    # create empty dataframe
    df_missing_letters = pd.DataFrame(index=df_layouts.keys())
    
    # get the list of bigrams
    bigram_list = list(df_bigrams.index.values)
    # iterate to compile list of letters
    letters_list = []
    for i in bigram_list:
        if i[0] not in letters_list:
            letters_list.append(i[0])
        if i[1] not in letters_list:
            letters_list.append(i[1])
    
    # iterate over the layouts
    for layout in df_missing_letters.itertuples():
        # iterate over letters
        missing_letters = ""
        for i in letters_list:
            if i not in df_layouts[layout.Index].values:
                missing_letters = missing_letters + i
        if missing_letters != '':
            df_missing_letters.at[layout.Index, 'Missing letters'] = missing_letters
    
    return df_missing_letters


def create_df_layouts(keys_list, layouts_str):
    """ Takes keys and layouts string and outputs DataFrame of layouts"""
    # cuts the text into blocks by >>
    data = list(filter(None, layouts_str.split('>>')))
    # puts names and layouts in 2 lists
    layouts_names = []
    layouts_list = []
    for t in data:
        splitted = t.split('\n', maxsplit=1)
        layouts_names.append(splitted[0])
        layouts_list.append(splitted[1].split())
    # create the dataframe
    return pd.DataFrame(list(zip(*layouts_list)), index=keys_list, columns=layouts_names)


def create_layouts_str_for_swap(base_name, layouts_str):
    """
    Takes a layout and generates one string
    containing all swap layout for create_df_layouts
    """
    # cuts the text into blocks by >>
    first_layout = list(filter(None, layouts_str.split('>>')))[0]
    # makes all swap possible
    swap_layouts_str = make_swap_layouts(base_name, first_layout)
    # add qwerty for comparaison
    swap_layouts_str += """>>Qwerty
# q w e r t y u i o p #
é a s d f g h j k l ; '
è z x c v b n m , . / -

"""
    return swap_layouts_str


def make_swap_layouts(base_name, base_layout):
    """
    Makes all possile one swap layout from base_layout.
    """
    # symbols to swap
    symbols_to_swap = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
                       "a", "s", "d", "f", "g", "h", "j", "k", "l", "z",
                       "x", "c", "v", "b", "n", "m", ",", ".", "'", "é"]

    # base_layout extraction
    base_layout = base_layout.split("\n")
    base_layout = base_layout[1] + "\n" + base_layout[2] + "\n" + base_layout[3]

    all_layouts = ""
    combination_tried = []

    # loop through possibility
    for symbol_from in symbols_to_swap:
        for symbol_to in symbols_to_swap:
            # checks if same symbol or already swapped
            same_symbol = symbol_from == symbol_to
            tried = (symbol_from, symbol_to) in combination_tried or (symbol_to, symbol_from) in combination_tried

            if not same_symbol and not tried:
                # adds name
                swap_name = f">>{base_name}, ({symbol_from}, {symbol_to})\n"
                # adds layout
                swap_layout = ""
                for symbol_original in base_layout:
                    if symbol_original == symbol_from:
                        swap_layout += symbol_to
                    elif symbol_original == symbol_to:
                        swap_layout += symbol_from
                    else:
                        swap_layout += symbol_original

                all_layouts += swap_name + swap_layout + "\n\n\n"
                combination_tried.append((symbol_from, symbol_to))

    # return all layouts found
    return all_layouts


def weight(bigram, layout, df_layouts, df_keys, df_penalties):
    """
    Returns the calculated weight of a bigram, for a given layout
    """
    
    # check that the bigram keys exist in the layout, otherwise return 0
    if (bigram[0] not in df_layouts[layout].values) or (bigram[1] not in df_layouts[layout].values):
        return 0.0

    # find the keys of each letter
    key1 = df_layouts.loc[df_layouts[layout] == bigram[0]].index[0]
    key2 = df_layouts.loc[df_layouts[layout] == bigram[1]].index[0]
    # get the weights, fingers, and keyrows
    weight1 = df_keys.at[key1, 'weights']
    finger1 = df_keys.at[key1, 'finger']
    keyrow1 = df_keys.at[key1, 'keyrow']
    weight2 = df_keys.at[key2, 'weights']
    finger2 = df_keys.at[key2, 'finger']
    keyrow2 = df_keys.at[key2, 'keyrow']

    # penalty exists only if same hand, and not same letter
    penalty = 0.0
    if (key1[0] == key2[0]) and (bigram[0] != bigram[1]):
        # define the row jump (column name in df_penalties)
        if(abs(keyrow1 - keyrow2) == 0):
            rowjump = "same_row"
        elif(abs(keyrow1 - keyrow2) == 1):
            rowjump = "row_jump1"
        elif(abs(keyrow1 - keyrow2) == 2):
            rowjump = "row_jump2"
        else:
            sys.exit('Penalty for line jump not defined')
        
        penalty = df_penalties.at[finger1 + finger2, rowjump]
    
    return weight1 + weight2 + penalty


def bigram_weight(df_layouts, df_keys, df_bigrams, df_penalties):
    """
    Create a dafaframe of the weight per bigram and layout
    """
    # create the empty dataframe
    df_bigram_weight = pd.DataFrame(index=df_bigrams.index.values, columns=df_layouts.keys())
    
    # iterate over the whole dataframe to compute the weights
    # iterating isn't efficient but I don't know better
    for column in df_bigram_weight:
        for row in df_bigram_weight.itertuples():
            df_bigram_weight.at[row.Index, column] = weight(row.Index, column, df_layouts, df_keys, df_penalties)

    return df_bigram_weight


def layout_results(df_bigrams, df_bigram_weight):
    """
    Generate dataframe of results (grade per layout per language)
    """
    # create the empty dataframe
    df_results = pd.DataFrame(index=df_bigram_weight.keys(), columns=df_bigrams.keys())
    
    # iterate over the dataframe to compute the grades
    # iterating isn't efficient but this is small enough
    for column in df_results: # language
        for row in df_results.itertuples(): # layout
            # sum of (probability of bigram (from df_bigram, is a percentage) times its weight)
            df_results.at[row.Index, column] = (df_bigrams[column]/100 * df_bigram_weight[row.Index]).sum()

    return df_results


def removeComments(string):
    """
    Remove the comments from the config file passed as argument. From:
    https://stackoverflow.com/questions/2319019/using-regex-to-remove-comments-from-source-files
    """
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
    
    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)
    
    return regex.sub(_replacer, string)


if __name__ == '__main__':
    main()
