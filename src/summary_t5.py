#!/usr/bin/env python3
import argparse
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger('summary_t5')


def t5_summarize(txt):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    input_ids = tokenizer.encode("summarize: " + txt, return_tensors="pt", max_length=1024).input_ids.to("cuda")
    outputs = model.generate(inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0])

TEXT1 = '''
1. A man hears a voice roar "Take it!" and removes his helmet; each contestant gets one triangle shape, whoever sticks their first wins the Super-Show. Senator Hull-Mendoza attacks John Kingsley-Schultz over bluenose meetings while denying involvement in fly casting at Oregon or getting a PhD from Florida after being asked about them during an interview on TV; twelve lives are lost when cars pile up, and all 94 people aboard die as rockets crash.
2. The roadside jingle has an alert system with each word read before the next one is shown while Barlow escapes police who seem to have mind-reading machines & TV eyes; he draws carbonated orange drink from another glassy wrapper but spills all over himself, noticing people still wear clothes and smoke/eat in shops near movie theaters showing _Babies Are Terrible_, Don't Have Children_.
3. A man who has been living underground for a long time comes out finding everything changed - horse races faster than ever without sense or reason, he can't understand any horses in his second race which seems like pigs one day and mudders another; they don't wear clothes/smoke tobacco anymore but eat food from shops near movie theaters showing _Babies Are Terrible_, Don't Have Children_.
4. A psychiatrist, an African man named Ryan-Ngana discuss their plan with Barlow for world domination; he doesn't take due to prejudice but agrees on seeing what arrangement can be made between them despite that. They want Congress create emergency act allowing dictatorship where they would rule as the dictator, Poprob insists using him in their plan and Sam Immerman swindles Verna out of money so she isn't part; others visiting investments lead Barlow demanding letter from Polar President plus session empowering Congress giving temporary emergency powers making him dictator or no other offers will be considered.
5. Barlow wins unanimously as new Polar president dismisses old one completely taking control over world finances and starting program building his own palace with painters working on portraits/statues; TV shows ad for _Parfum Assault Criminale_ but Mrs. Garvy thinks it's impossible because she thought only rocket that went there crashed onto Moon, while Buzz Rentshaw makes fun of her hazy convictions about space travel on _Henry's Other Mistress_.
6. A woman named Mrs. Garvy goes to a freud with amnesia and can't remember things like everyone else did so he suggests she take vacation on Venus for its soap root; later family finds an article in his journal about going there after disciplining herself out of rocket-ship obsession, enjoying the trip despite not seeing anything due to meteorite season exploring tropical island with free soap root/delicious fruits from Earth transplants playing cards or craps during takeoff & landing. Nations envy Columbia's freedom so they discuss an ambitious plan called "The Venusian Way" - sending one city at time wrecking empty cities for steel, Black-Kupperman reads out loud but another responds feebly saying we have no time to waste because it must be American!
'''

def main():
    print(t5_summarize(TEXT1))


if __name__ == '__main__':
    main()