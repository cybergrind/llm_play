#!/usr/bin/env python3
import argparse
import logging

from ctransformers import AutoModelForCausalLM


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('run_infer')
WINE_OUT_FIELDS = 13


prompt_template = f'''
### Instruction:
Given the CSV file with content:

{{products}}

Infer the new CSV with following columns:

sku - from original file
name - from original file
varietal
color
vintage
country
region
sub-region
appellation
alchohol
is_sparkling - int
is_dessert - int
is_fortified - int

Output should contain the same number of items as input.
Output MUST contain header.
Output columns MUST be pipe-separated.
Output MUST have {WINE_OUT_FIELDS} columns

### Response:
'''

products = '''
sku|type|name|typ.name|cat|ml|pack|proof
1004|4M1J|STONELEIGH SAUVIGNON BLANC|AUSTRALIAN|140|750|12|0.0
1023|4BSJ|CHATEAU BRANAIRE DUCRU 2009|BORDEAUX, ST. JULIEN|140|750|12|0.0
2017|4AAA|DEFUSSIGNY PIN CHARENTES BLANC|CLOSEOUT WINES|140|750|12|80.0
6580|4LCL|BERNIE APRICOT COOLER (CHILE)|WINE COOLERS|140|355|1|0.0
6581|4LCL|BERNIE ASSORTED COOLERS (CHILE)|WINE COOLERS|140|355|1|0.0
6582|4LCL|BERNIE RASPBERRY COOLER (CHILE)|WINE COOLERS|140|355|1|0.0
6583|4LCL|CANCUN COOLER|WINE COOLERS|140|355|1|0.0
6584|4LCL|CAPT. TOM MARGARITA|WINE COOLERS|140|237|45|0.0
6585|4LCL|CAPT. TOM PINA COLADA|WINE COOLERS|140|237|45|0.0
6586|4LCL|CAPT. TOM STRAWBERRY DAQUIRI|WINE COOLERS|140|237|45|0.0
6587|4LCL|COZUMEL WINE REFRESHER|WINE COOLERS|140|325|1|0.0
6588|4LCL|COZUMEL WINE REFRESHER|WINE COOLERS|140|325|1|0.0
6589|4LCL|Z FRANZIA CHARDONNAY COOLER|WINE COOLERS|140|355|1|0.0
6590|4LCL|Z FRANZIA WHITE GRENACHE COOLER|WINE COOLERS|140|355|1|0.0
6594|4LCL|ICE BREAKER FROSTY KAMIKAZE|WINE COOLERS|140|828|45|0.0
6595|4LCL|ICE BREAKER FROSTY NEW ORLEANS HURRICANE|WINE COOLERS|140|828|45|0.0
6596|4LCL|ICE BREAKER FROSTY SCREWDRIVER|WINE COOLERS|140|828|45|0.0
6597|4LCL|ICE BREAKERS FROSTY SEA BREEZE N/A|WINE COOLERS|140|828|45|0.0
6598|4LCL|TROPICAL FREEZE CALYPSO 15/3/8|WINE COOLERS|140|237|45|0.0
6599|4LCL|TROPICAL FREEZE MARGARITA 15/3/8 11.8O|WINE COOLERS|140|237|45|11.8
6600|4LCL|TROPICAL FREEZE PEACH DAIQUIRI 15/3/8 11.8O|WINE COOLERS|140|237|45|11.8
6601|4LCL|TROPICAL FREEZE PINA COLADA 15/3/8|WINE COOLERS|140|237|45|0.0
6602|4LCL|TROPICAL FREEZE PINEAPLE DAIQUIRI 15/3/8 11.8O|WINE COOLERS|140|237|45|11.8
6603|4LCL|TROPICAL FREEZE PUNCH 15/3/8 11.8O|WINE COOLERS|140|237|45|11.8
6604|4LCL|TROPICAL FREEZE STRAWBERRY DAIQU 15/3/8 11.8O|WINE COOLERS|140|237|45|11.8
6605|4LCL|TROPICAL FREEZE WHATAMELON 15/3/8|WINE COOLERS|140|237|45|0.0
6606|4M01|ABUELOS SANGRIA BLUSH||140|1500|1|0.0
'''


def parse_args():
    parser = argparse.ArgumentParser(description='DESCRIPTION')
    parser.add_argument('-m', '--model', default='../guanaco-33B.ggmlv3.q4_1.bin')
    # parser.add_argument('-m', '--mode', default='auto', choices=['auto', 'manual'])
    # parser.add_argument('-l', '--ll', dest='ll', action='store_true', help='help')
    return parser.parse_args()


def main():
    args = parse_args()
    llm = AutoModelForCausalLM.from_pretrained(
        args.model, model_type='llama', stream=True, temperature=0, max_new_tokens=2048
    )
    prompt = prompt_template.format(products=products)
    print(prompt)
    print('=' * 80)

    last_n = False
    for line in llm(prompt, stream=True, temperature=0, max_new_tokens=4096):
        if not line:
            continue
        if line == '\n':
            if last_n:
                continue
            print(line, end='')
            last_n = True
            continue
        last_n = False
        print(line, end='')


if __name__ == '__main__':
    main()
