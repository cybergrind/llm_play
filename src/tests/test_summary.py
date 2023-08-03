from hashlib import sha256
from unittest.mock import MagicMock

import pytest

from summary_generator import ASSISTANT_PREFIX, do_summary, squeeze, summary_query
from utils.query import DIVINE, query_api


TEXT0 = '''1. Barlow meets with a psychiatrist and an African-American man named Ryan Ngana to discuss their plan for world domination through creating an emergency act in Congress that would allow them to have a dictatorship, but doesn't take
 his hand because he thinks working with someone who looks African might not be best.
2. They want to create it by putting some lemming urge into people which they will reveal later when getting the right signatures on the deal; however, Barlow is prejudiced against lemmings and Poprob insists on using him for their
plan because he thinks Barlow might have an irrational approach.
3. Ryan Ngana mentions that he was called away from working on his theorem before finishing it so now needs to go back while running the San Francisco subway system, agreeing with them seeing what arrangement can be made between the
m.
4. The character plans to sell tundra building lots which would result in lemmings committing suicide by purchasing them because he thinks this will solve their problem and make him rich quickly without working hard or having any id
eas himself - disastrous consequences follow when people start buying the land, causing many deaths.
5. Sam Immerman swindled Verna; she wasn't on that deal since too old to be roped anymore but told visitors he wouldn't give info until they gave him a letter of intent from Polar President and session empowered by Congress for maki
ng dictator, which Barlow got unanimously.
6. He becomes the new Polar President with complete control over world finances under temporary emergency powers needing approval from Congress; he demands title World Dictator plus publicity campaign historical writeup but refuses
modifying his demands or knock off even 10%. If refused, won't accept other offer.
7. Barlow builds a palace with painters and sculptors working on portraits/statues while TV shows an ad for _Parfum Assault Criminale_ saying "easy as trip to Venus" but Mrs Garvy thinks it impossible because she thought only rocket
 went there crashed Moon; Buzz Rentshaw, Master Rocket Pilot of the Venus run makes fun at her hazy convictions about space travel.
'''  # noqa

TEXT1 = '''Context: A chat between a user and an assistant. The assistant provides concise summaries of the user's text, distilling it down to the most important details in approximately 30 words.

USER: Please summarize the following story in roughly 30 words, maintaining only the key points:
1. A man hears a voice roar "Take it!" and removes his helmet; each contestant gets one triangle shape, whoever sticks their first wins the Super-Show. Senator Hull-Mendoza attacks John Kingsley-Schultz over bluenose meetings while denying involvement in fly casting at Oregon or getting a PhD from Florida after being asked about them during an interview on TV; twelve lives are lost when cars pile up, and all 94 people aboard die as rockets crash.
2. The roadside jingle has an alert system with each word read before the next one is shown while Barlow escapes police who seem to have mind-reading machines & TV eyes; he draws carbonated orange drink from another glassy wrapper but spills all over himself, noticing people still wear clothes and smoke/eat in shops near movie theaters showing _Babies Are Terrible_, Don't Have Children_.
3. A man who has been living underground for a long time comes out finding everything changed - horse races faster than ever without sense or reason, he can't understand any horses in his second race which seems like pigs one day and mudders another; they don't wear clothes/smoke tobacco anymore but eat food from shops near movie theaters showing _Babies Are Terrible_, Don't Have Children_.
4. A psychiatrist, an African man named Ryan-Ngana discuss their plan with Barlow for world domination; he doesn't take due to prejudice but agrees on seeing what arrangement can be made between them despite that. They want Congress create emergency act allowing dictatorship where they would rule as the dictator, Poprob insists using him in their plan and Sam Immerman swindles Verna out of money so she isn't part; others visiting investments lead Barlow demanding letter from Polar President plus session empowering Congress giving temporary emergency powers making him dictator or no other offers will be considered.
5. Barlow wins unanimously as new Polar president dismisses old one completely taking control over world finances and starting program building his own palace with painters working on portraits/statues; TV shows ad for _Parfum Assault Criminale_ but Mrs. Garvy thinks it's impossible because she thought only rocket that went there crashed onto Moon, while Buzz Rentshaw makes fun of her hazy convictions about space travel on _Henry's Other Mistress_.
6. A woman named Mrs. Garvy goes to a freud with amnesia and can't remember things like everyone else did so he suggests she take vacation on Venus for its soap root; later family finds an article in his journal about going there after disciplining herself out of rocket-ship obsession, enjoying the trip despite not seeing anything due to meteorite season exploring tropical island with free soap root/delicious fruits from Earth transplants playing cards or craps during takeoff & landing. Nations envy Columbia's freedom so they discuss an ambitious plan called "The Venusian Way" - sending one city at time wrecking empty cities for steel, Black-Kupperman reads out loud but another responds feebly saying we have no time to waste because it must be American!
ASSISTANT: Short summary:
'''  # noqa


@pytest.fixture
def args():
    mm = MagicMock()
    mm.ctx = 4096
    yield mm


@pytest.mark.skip
def test_01_summary(args):
    before = len(TEXT1)
    after = do_summary(summary_query(TEXT1, end=ASSISTANT_PREFIX), args)
    after = do_summary(summary_query(after, end=ASSISTANT_PREFIX), args)
    sum1 = sha256(after.encode()).hexdigest()
    after = do_summary(summary_query(after, end=ASSISTANT_PREFIX), args)
    sum2 = sha256(after.encode()).hexdigest()
    after = do_summary(summary_query(after, end=ASSISTANT_PREFIX), args)
    sum3 = sha256(after.encode()).hexdigest()
    assert sum3 != sum2
    assert len(after) < args.ctx // 2


# @pytest.mark.skip
def test_02_query_api(args):
    before = len(TEXT1)
    preset = DIVINE.copy()
    after = query_api(summary_query(TEXT1, end=ASSISTANT_PREFIX), preset)
    after = query_api(summary_query(after, end=ASSISTANT_PREFIX), preset)
    sum1 = sha256(after.encode()).hexdigest()

    temp_delta = 0.1
    preset['temperature'] += temp_delta
    preset['repetition_penalty'] += 0.1
    preset['repetition_penalty_range'] += 256
    after = query_api(summary_query(after, end=ASSISTANT_PREFIX), preset)
    sum2 = sha256(after.encode()).hexdigest()
    print(f'{len(after)=} {sum2=}')

    preset['temperature'] += temp_delta
    preset['repetition_penalty'] += 0.1
    preset['repetition_penalty_range'] += 128
    after = query_api(summary_query(after, end=ASSISTANT_PREFIX), preset)
    sum3 = sha256(after.encode()).hexdigest()
    print(f'{len(after)=} {sum3=}')

    preset['temperature'] += temp_delta
    preset['repetition_penalty'] += 0.1
    preset['repetition_penalty_range'] += 128
    after = query_api(summary_query(after, end=ASSISTANT_PREFIX), preset)
    sum4 = sha256(after.encode()).hexdigest()
    print(f'{len(after)=} {sum4=}')

    preset['temperature'] += temp_delta
    preset['repetition_penalty'] += 0.1
    preset['repetition_penalty_range'] += 128
    after = query_api(summary_query(after, end=ASSISTANT_PREFIX), preset)
    sum5 = sha256(after.encode()).hexdigest()
    print(f'{len(after)=} {sum5=}')

    assert sum5 != sum4

    assert len(after) < args.ctx // 2
