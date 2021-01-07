# florianfmmartin's fork
The main goal I want to acheive with this fork is to use the layout evaluator to generate more ergonomic layouts based on existing ones by swapping characters in layout until the ergonomic score isn't getting lower.

## Things I added
Here is a list of features I added with it's description and way to modify it.

- Prompts for Personal Average:
    - Enter a list of language and then their corresponding weights to make your PA
- Prompts for layout weights to use:
    - Enter the name corresponding to the weights to use it
    - You can add other weights to the config by using a "[w_MINE]" tag and enter "MINE" in prompt to use it
- Added some layout (more info in the layouts section)
- Added layout generation:
    - You can ask to generate a layout
    - To modify from what layout the generation will happen you need to do so (for now, I hope to make the layout promptable in the near future):
        - Put the layout first in the config
        - Change the name of the variable _current\_name_ before the while loop in the _app\_generate_ function
- Added thumb row evaluation thanks to u/sdothum on reddit
    - Thumb evaluation is done with a different config_thumb.txt
    - Generation in yet to be tested/ameliorated
        - Base layout that do not contain any letters on the thumb row won't make one appear

## The Generation
The generation is quite simple and really not optimal, but you can still make some great layouts with it. It starts from the base layout and makes a big list of layout one swap away from itself and takes the one with the best score to continue iterating until the best layout from epoch _n+1_ can't beat the one from epocn _n_.

## Layouts
I added a few layout from the internet and some that I generated. For the ones I generated DARN, COLE and ROLL refer to the type of weights chose for the generation. ROLL means no weights and only penalties. This was done by changing the return value of the _weights_ function for it to return only the variable _penalty_. They were for the most part generated for 50/50 en/fr so it might not fit your needs.

## Contribution
I am fully open to contribution. Modification to the script and to the config by adding layouts or weights are more than welcomed :)

## More information
For more information go checkout the orginial [repo](https://github.com/bclnr/kb-layout-evaluation).
