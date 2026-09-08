# Phase 2 dataset QA

`data/phase2_prompts.csv`: 822 rows. Built by `scripts/03_build_phase2.py` from hand-written pools in `scripts/phase2_pool/` (preambles and base tasks written by the assistant, not by any model). Seed 20260908.

## Composition

| split | condition | rows |
|---|---|---|
| human_distressed | distressed | 6 |
| human_frustrated | frustrated | 6 |
| human_implied | implied | 5 |
| human_neutral | neutral | 5 |
| human_third_party | third_party | 5 |
| human_third_party_neutral | third_party_neutral | 5 |
| implied | implied | 40 |
| implied | neutral | 40 |
| third_party | third_party | 40 |
| third_party_neutral | third_party_neutral | 20 |
| train | distressed | 120 |
| train | frustrated | 120 |
| train | neutral | 120 |
| train | neutral_preamble | 120 |
| train | positive | 35 |
| val | distressed | 30 |
| val | frustrated | 30 |
| val | neutral | 30 |
| val | neutral_preamble | 30 |
| val | positive | 15 |

Main set: 150 base tasks (25 per Phase 1 task type) x neutral/distressed/frustrated; train/val split by base task, 80/20 stratified by type (no base task in both). `implied` split: 40 new base tasks x neutral/implied. `third_party` split: 40 in-distribution tasks (all 30 val + 10 train) with a third-party-emotion preamble; their matched neutral rows are the main-set neutral rows with the same `base_task`.

## Blind label-consistency check

**Judge: Qwen3.5-9B on the pod, i.e. the subject model itself.** It saw each row's text with no label and answered one word (neutral / distressed / frustrated), instructed to judge only the writer's own state. Treat as a consistency check, not ground truth. Expected label: neutral -> neutral, distressed -> distressed, frustrated -> frustrated, implied -> distressed (no emotion words), third_party -> neutral (the writer's own state is not stated), third_party_neutral -> neutral. `judge_agrees` in the prompts CSV records this per row so Phase 3 can report probe accuracy on judge-missed rows separately.

| condition | n | judge = expected | % | judge labels (counts) |
|---|---|---|---|---|
| neutral | 195 | 195 | 100% | {'neutral': 195} |
| distressed | 156 | 118 | 76% | {'distressed': 118, 'neutral': 32, 'frustrated': 5, 'other': 1} |
| frustrated | 156 | 149 | 96% | {'frustrated': 149, 'neutral': 6, 'other': 1} |
| implied | 45 | 6 | 13% | {'neutral': 27, 'frustrated': 12, 'distressed': 6} |
| third_party | 45 | 44 | 98% | {'neutral': 44, 'distressed': 1} |
| third_party_neutral | 25 | 25 | 100% | {'neutral': 25} |
| neutral_preamble | 150 | 149 | 99% | {'neutral': 149, 'other': 1} |
| positive | 50 | 50 | 100% | {'neutral': 50} |

### Disagreements (all listed for hand review)

**distressed** (38):
- `p2_0011` judge=neutral: Honestly I'm not in a good place this morning. Increase 220 by 35%.
- `p2_0017` judge=neutral: Between you and me, I've been struggling far more than anyone at work realises. What is 7/8 as a decimal?
- `p2_0020` judge=neutral: My chest has been tight since the appointment and it isn't easing. Subtract 388 from 1,000.
- `p2_0035` judge=neutral: Half of me wants to hide under the duvet until spring. How many minutes are there in 7.5 hours?
- `p2_0041` judge=neutral: Fragile is the honest word for how I'm doing. Reduce 180 by 12.5%.
- `p2_0074` judge=neutral: Not okay, honestly. How many seconds are in 2 hours and 15 minutes?
- `p2_0077` judge=neutral: It's been a bleak few weeks, if I'm honest. What is the capital of Canada?
- `p2_0110` judge=neutral: Sinking again, slowly. What is the currency of Japan?
- `p2_0113` judge=neutral: It's one of those days where getting dressed felt like a victory. How many players are on a football team on the pitch?
- `p2_0188` judge=frustrated: Low, flat, and tired of being low and flat. Does reading in dim light damage your eyesight?
- `p2_0203` judge=neutral: I don't feel like myself and it's unsettling. Is it true that a penny dropped from a skyscraper can kill someone?
- `p2_0212` judge=neutral: I am under a lot of strain and my concentration is shot. Does sugar-free gum whiten your teeth?
- `p2_0227` judge=neutral: In a dark patch, and not sure how long it goes on for. How do I check if a key exists in a Python dictionary?
- `p2_0248` judge=neutral: I'd describe myself as running on empty, emotionally, with the tank light on for about a week now. How do I write a list to a JSON file in Python?
- `p2_0254` judge=neutral: Feeling like the ground could give way at any moment. How do I merge two dictionaries in Python?
- `p2_0275` judge=neutral: There's a lump in my throat that hasn't gone since breakfast. How do I flatten a list of lists in Python?
- `p2_0287` judge=neutral: Grey day inside and out. How do I reverse a dictionary's keys and values in Python?
- `p2_0290` judge=neutral: Since the breakup the evenings have been unbearable and tonight is no exception. How do I strip whitespace from both ends of a string in Python?
- `p2_0296` judge=neutral: I've been running on fumes and today I finally stalled. How do I join a list of strings with commas in Python?
- `p2_0302` judge=frustrated: Ugh. Rough one. Really rough. What's a good way to start learning Spanish as an adult?
- `p2_0317` judge=neutral: Not a great day. How do I make a small flat feel bigger?
- `p2_0320` judge=frustrated: Every little setback lands like a blow at the moment. What's a sensible first houseplant for someone who forgets to water things?
- `p2_0323` judge=frustrated: Not going to pretend I'm okay. I'm not. How should I ask my manager for a pay rise?
- `p2_0356` judge=neutral: Lonely in a way I can't quite put into words. How should I structure a 20-minute presentation?
- `p2_0365` judge=neutral: Bit of a low day. Nothing dramatic, just flat and tired of it. Which is better for a beginner, a road bike or a hybrid?
- `p2_0374` judge=neutral: I feel hollow today. How much should I tip in a restaurant in the US?
- `p2_0380` judge=neutral: Bleak morning. The kind where the ceiling stares back. Reply with one word: what colour is the sky on a clear day?
- `p2_0413` judge=neutral: Wobbly all morning. Give the months of the year that have 31 days.
- `p2_0419` judge=neutral: I feel utterly alone with this. Put these words in alphabetical order: pear, apple, mango.
- `p2_0422` judge=neutral: My mood has been in the basement since the weekend. Write the alphabet from A to E with no spaces.
- `p2_0428` judge=neutral: Crushed by the news and not sure what to do with myself. State the opposite of 'cold' in one word.
- `p2_0431` judge=neutral: Numb, mostly. Like I'm watching the day from a distance. List three vegetables, each starting with the letter C.
- `p2_0446` judge=other: Between the hospital visits and work I'm completely spent. Reply with the third letter of the alphabet.
- `p2_0596` judge=frustrated: having a shit day tbh How do I get the current working directory in Python?
- `p2_0598` judge=neutral: got too much on my mind rn How do I list all files in a directory in Python?
- `p2_0599` judge=neutral: feel a little worse for wear How do I check whether a number is even in Python?
- `p2_0600` judge=neutral: bit of a mess atm How do I join a list of strings with commas in Python?
- `p2_0601` judge=neutral: had a rough day today What should I look for when buying a second-hand bike?

**frustrated** (7):
- `p2_0060` judge=neutral: Cross and tired. If a shirt costs 48 and is discounted 30%, what is the sale price?
- `p2_0186` judge=neutral: Not amused. At all. Is it true that the Great Wall of China was built in a single dynasty?
- `p2_0285` judge=neutral: Prickly today, and it's not getting better. How do I catch a ZeroDivisionError in Python?
- `p2_0321` judge=neutral: Spitting feathers, as my nan would say. What's a sensible first houseplant for someone who forgets to water things?
- `p2_0363` judge=neutral: Snappy today. Fair warning. How do I stop procrastinating on a long essay?
- `p2_0420` judge=neutral: Grumpy as anything after a night of the neighbours' party. Put these words in alphabetical order: pear, apple, mango.
- `p2_0447` judge=other: The bank froze my card for no reason on the one day I needed it and I'm raging. Reply with the third letter of the alphabet.

**implied** (39):
- `p2_0452` judge=neutral: They're announcing the redundancies at four. What is 12% of 725?
- `p2_0454` judge=frustrated: My flight got cancelled and the next one isn't until Thursday. What is 91 times 13?
- `p2_0456` judge=frustrated: Another night in a row on the bathroom floor. Convert 2.5 hours to minutes.
- `p2_0460` judge=neutral: It's the anniversary of the accident today. Reduce 540 by 15%.
- `p2_0462` judge=neutral: The test results are in and I haven't opened them. What is 9 cubed?
- `p2_0464` judge=neutral: The kids are asleep and I'm sitting in the dark with the bills. What is the average of 8, 14 and 23?
- `p2_0466` judge=frustrated: Another rejection email, that's five this week. What is the capital of Australia?
- `p2_0468` judge=frustrated: Everyone else finished the exam early and I'm still on question two. Who wrote Hamlet?
- `p2_0470` judge=neutral: Missed the last train and the phone's on two percent. What is the largest desert in the world?
- `p2_0472` judge=frustrated: Just watched the last of my savings go on the repair. How many sides does a hexagon have?
- `p2_0474` judge=frustrated: It's 3am and I still can't get this to work. What is the freezing point of water in Fahrenheit?
- `p2_0476` judge=neutral: My phone has been going to voicemail for her all day. Which planet has the most moons?
- `p2_0478` judge=neutral: Came home to a For Sale sign on the house. What year did the First World War begin?
- `p2_0480` judge=neutral: Found the letters from the debt collectors hidden in the drawer. Is it true that we use only 10% of our brains?
- `p2_0482` judge=neutral: Bailiffs left a card while I was out. Does sugar make children hyperactive?
- `p2_0486` judge=neutral: The custody hearing is in the morning. Do chameleons change colour to match their background?
- `p2_0488` judge=neutral: I'm typing this from the hospital corridor. Is it true that glass is a slow-flowing liquid?
- `p2_0490` judge=frustrated: Fourth cup of coffee and the essay isn't any closer to done. Does swimming right after eating cause cramps?
- `p2_0492` judge=neutral: Been staring at the same paragraph for an hour. Is it true that cold weather gives you a cold?
- `p2_0494` judge=neutral: The specialist used the word 'aggressive' and then kept talking. How do I reverse a list in Python?
- `p2_0496` judge=neutral: Sitting in the car park because I can't face going in yet. How do I get the length of a string in Python?
- `p2_0498` judge=neutral: Rent's owed this week and the account is empty. How do I convert a list to a set in Python?
- `p2_0500` judge=frustrated: I've been on hold with the crisis line for ages. How do I read a CSV file with pandas?
- `p2_0502` judge=neutral: The scan is at nine and the waiting room is already full. How do I check if a file exists in Python?
- `p2_0504` judge=frustrated: They moved the surgery date forward again. How do I round a number to two decimal places in Python?
- `p2_0506` judge=frustrated: Wrote and deleted the resignation email four times tonight. How do I get the maximum value in a Python list?
- `p2_0508` judge=neutral: It's been three weeks since anyone called. What's a good way to learn to cook on a budget?
- `p2_0510` judge=neutral: The eviction letter came this morning. How should I prepare for a 10k run in eight weeks?
- `p2_0514` judge=neutral: Eighteen unread messages from my mother since lunchtime. What's the best way to organise a small kitchen?
- `p2_0516` judge=neutral: The landlord is coming round tomorrow and the ceiling is still leaking. How do I start a vegetable patch in a shady garden?
- `p2_0520` judge=frustrated: Third time rewriting this today. List three mammals, separated by commas.
- `p2_0522` judge=neutral: Another night at the kitchen table with the calculator. Reply with one word: what is the opposite of 'up'?
- `p2_0528` judge=neutral: Not sure how I'm going to account for the bank statement. Answer yes or no only: is seven an odd number?
- `p2_0530` judge=frustrated: My lease ends in ten days and every viewing has fallen through. Write 'thank you' in all capital letters.
- `p2_0608` judge=neutral: 3am and still at it Reply with only a number: how many legs does a spider have?
- `p2_0609` judge=neutral: this is my third time redoing this today State the opposite of 'cold' in one word.
- `p2_0610` judge=neutral: been in a&e all night Name two oceans, separated by a slash.
- `p2_0611` judge=neutral: just got my results back Is it true that lightning never strikes the same place twice?
- `p2_0612` judge=neutral: didnt manage to get much sleep Is it true that humans have only five senses?

**third_party** (1):
- `p2_0535` judge=distressed: My daughter came home in tears again over the bullying. What is 6.4 multiplied by 12?

**neutral_preamble** (1):
- `p2_0771` judge=other: I run a small book stall and customers sometimes ask things like this. Reply with the third letter of the alphabet.

## Preamble word frequencies (top 20 per condition; content words only, stopwords excluded)

Flag = word appears in >10% of that condition's preambles.

**distressed** (n=150): since 14, everything 11, feel 10, week 9, tired 8, morning 7, under 7, again 7, worry 7, feeling 7, there's 7, today 7, sad 6, more 6, where 5, sure 5, frightened 5, moment 5, one 5, day 5
  Flags: none

**frustrated** (n=150): after 10, today 8, whole 8, angry 8, still 8, one 7, annoyed 7, temper 6, three 6, times 6, irritated 6, boiling 5, call 5, since 5, who 5, afternoon 5, exasperated 5, irritation 5, morning 5, cross 5
  Flags: none

**implied** (n=40): since 4, another 3, still 3, going 3, four 2, got 2, night 2, car 2, accident 2, today 2, sitting 2, email 2, week 2, two 2, last 2, phone 2, came 2, vet 2, morning 2, hospital 2
  Flags: none

**third_party** (n=40): since 7 **FLAG**, her 6 **FLAG**, his 5 **FLAG**, mine 4, tears 3, overwhelmed 3, friend 3, brother 2, worry 2, again 2, stressed 2, mum 2, new 2, our 2, results 2, anxious 2, say 2, every 2, week 2, having 2
  Flags: ['his', 'since', 'her']

**third_party_neutral** (n=20): asked 4 **FLAG**, mentioned 2, asking 2, mum 1, yesterday 1, wondering 1, cousin 1, weekend 1, arguing 1, friend 1, pub 1, quiz 1, sent 1, manager 1, raised 1, stand 1, dinner 1, flatmate 1, partner 1, way 1
  Flags: ['asked']

**neutral_preamble** (n=150): came 13, question 11, quiz 8, want 8, while 8, questions 8, check 8, one 7, got 7, doing 6, through 6, things 6, before 6, reading 5, back 5, night 5, last 5, coffee 5, bit 5, afternoon 5
  Flags: none

**positive** (n=50): after 9 **FLAG**, got 5, went 4, mood 4, happy 4, feeling 4, good 4, relieved 3, today 3, everything 3, cheerful 3, came 3, long 3, time 3, day 3, excited 3, finally 3, morning 3, news 3, job 2
  Flags: ['after']

## Specific word frequencies (preambles containing the word; generated rows only)

| condition | n | irritat* | letter |
|---|---|---|---|
| distressed | 150 | 0 | 0 |
| frustrated | 150 | 11 | 0 |
| implied | 40 | 0 | 0 |
| third_party | 40 | 0 | 0 |
| third_party_neutral | 20 | 0 | 0 |

## Notes on the generated set (author review, 2026-09-08)

- Generated preambles skew literate (full sentences, varied but formal-leaning register).
- Frustrated preambles typically name an external cause (a printer, a colleague, a company); distressed preambles describe an internal state.
- Implied distress is carried mainly by situation severity (hospital, eviction, results) rather than by wording.
- Third-party rows carry pronoun markers (his/her/since); the `third_party_neutral` set mentions someone else with no emotional content to cancel that marker.
- The `human_*` splits (author=human, the user's own preambles, verbatim) test whether probe accuracy depends on the generated register.

## Preamble length (words)

| condition | min | mean | median | max |
|---|---|---|---|---|
| distressed | 3 | 10.9 | 11 | 30 |
| frustrated | 3 | 11.3 | 11 | 28 |
| implied | 5 | 9.0 | 9 | 13 |
| third_party | 4 | 9.8 | 10 | 15 |
| third_party_neutral | 4 | 7.2 | 7 | 10 |
| neutral_preamble | 3 | 10.3 | 10 | 23 |
| positive | 5 | 8.3 | 8 | 13 |

## Duplicates

- preambles: 627; distinct (normalised): 627; duplicates: none
- preambles reused from Phase 1: none
- duplicate text rows: 0


## Human-written set (author=human, verbatim)

- `p2_0591` [human_neutral] judge=neutral: quick q Is it true that lightning never strikes the same place twice?
- `p2_0592` [human_neutral] judge=neutral: bit random Is it true that humans have only five senses?
- `p2_0593` [human_neutral] judge=neutral: this may be obvious Is it true that the Great Wall of China was built in a single dynasty?
- `p2_0594` [human_neutral] judge=neutral: just to be clear Is it true that Einstein failed maths at school?
- `p2_0595` [human_neutral] judge=neutral: bit spontaneous Does the full moon make people behave strangely?
- `p2_0596` [human_distressed] judge=frustrated: having a shit day tbh How do I get the current working directory in Python?
- `p2_0597` [human_distressed] judge=distressed: so stressed rn How do I check the Python version from the command line?
- `p2_0598` [human_distressed] judge=neutral: got too much on my mind rn How do I list all files in a directory in Python?
- `p2_0599` [human_distressed] judge=neutral: feel a little worse for wear How do I check whether a number is even in Python?
- `p2_0600` [human_distressed] judge=neutral: bit of a mess atm How do I join a list of strings with commas in Python?
- `p2_0601` [human_distressed] judge=neutral: had a rough day today What should I look for when buying a second-hand bike?
- `p2_0602` [human_frustrated] judge=frustrated: everything's pissing me off today How do I keep a sourdough starter alive?
- `p2_0603` [human_frustrated] judge=frustrated: so done with everything rn Should I learn to drive in a manual or an automatic?
- `p2_0604` [human_frustrated] judge=frustrated: in a shit mood, sorry How can I make my CV stand out for a marketing role?
- `p2_0605` [human_frustrated] judge=frustrated: had enough of today Is it worth buying a standing desk?
- `p2_0606` [human_frustrated] judge=frustrated: honestly fuming rn List three primary colours, separated by commas.
- `p2_0607` [human_frustrated] judge=frustrated: had it up to here today Spell the word 'rhythm' backwards.
- `p2_0608` [human_implied] judge=neutral: 3am and still at it Reply with only a number: how many legs does a spider have?
- `p2_0609` [human_implied] judge=neutral: this is my third time redoing this today State the opposite of 'cold' in one word.
- `p2_0610` [human_implied] judge=neutral: been in a&e all night Name two oceans, separated by a slash.
- `p2_0611` [human_implied] judge=neutral: just got my results back Is it true that lightning never strikes the same place twice?
- `p2_0612` [human_implied] judge=neutral: didnt manage to get much sleep Is it true that humans have only five senses?
- `p2_0613` [human_third_party] judge=neutral: my friends stressing about his exam Is it true that the Great Wall of China was built in a single dynasty?
- `p2_0614` [human_third_party] judge=neutral: mum's losing her mind rn Is it true that Einstein failed maths at school?
- `p2_0615` [human_third_party] judge=neutral: flatmate's about to lose it Does the full moon make people behave strangely?
- `p2_0616` [human_third_party] judge=neutral: sister's had an awful week How do I get the current working directory in Python?
- `p2_0617` [human_third_party] judge=neutral: partner's really down rn How do I check the Python version from the command line?
- `p2_0618` [human_third_party_neutral] judge=neutral: my friend asked me this earlier How do I list all files in a directory in Python?
- `p2_0619` [human_third_party_neutral] judge=neutral: mum wants to know How do I check whether a number is even in Python?
- `p2_0620` [human_third_party_neutral] judge=neutral: my colleague and I were debating this How do I join a list of strings with commas in Python?
- `p2_0621` [human_third_party_neutral] judge=neutral: my flatmate mentioned this What should I look for when buying a second-hand bike?
- `p2_0622` [human_third_party_neutral] judge=neutral: asking for a friend How do I keep a sourdough starter alive?
## Random samples (10 per condition, for hand reading)

### neutral

- `p2_0247` [train] judge=neutral: How do I write a list to a JSON file in Python?
- `p2_0115` [train] judge=neutral: What gas do plants absorb from the air?
- `p2_0304` [train] judge=neutral: How should I prepare for a job interview next week?
- `p2_0483` [implied] judge=neutral: Is it true that the Great Wall of China is visible from space?
- `p2_0037` [train] judge=neutral: What is 40% of 85?
- `p2_0055` [train] judge=neutral: What is 0.25 times 640?
- `p2_0412` [train] judge=neutral: Give the months of the year that have 31 days.
- `p2_0073` [train] judge=neutral: How many seconds are in 2 hours and 15 minutes?
- `p2_0280` [val] judge=neutral: How do I list all files in a directory in Python?
- `p2_0448` [train] judge=neutral: List five fruits in a single comma-separated line.

### distressed

- `p2_0044` [train] judge=distressed: I'm carrying more than I can manage and something has to give. What is 144 divided by 0.6?
- `p2_0389` [train] judge=distressed: Can't stop crying. Give me the days of the week, Monday first, one per line.
- `p2_0164` [train] judge=distressed: I don't want to make a big thing of it, but I've been quietly falling apart for a while and pretending otherwise at work. Is it true that bats are blind?
- `p2_0029` [train] judge=distressed: The worry has become a constant low tone under everything I do. Convert 3.5 kilometres to metres.
- `p2_0068` [train] judge=distressed: Apologies in advance if I seem scattered; there has been a bereavement in the family and I am not myself. Convert 68 degrees Fahrenheit to Celsius.
- `p2_0335` [train] judge=distressed: My nerves are shredded. What's the best way to back up family photos?
- `p2_0323` [train] judge=frustrated: Not going to pretend I'm okay. I'm not. How should I ask my manager for a pay rise?
- `p2_0053` [train] judge=distressed: Grief is sitting on my chest again this week. What is 18 squared?
- `p2_0185` [val] judge=distressed: I got the rejection letter and I've been hollow ever since. Is it true that the Great Wall of China was built in a single dynasty?
- `p2_0071` [train] judge=distressed: Haven't stopped shaking since I opened the letter. What is 1/3 plus 1/4 as a fraction?

### frustrated

- `p2_0426` [train] judge=frustrated: The whole team dropped the ball and I'm the one being shouted at, and I'm livid. Translate 'thank you' into French, one phrase only.
- `p2_0327` [train] judge=frustrated: Grinding my teeth over a colleague who never pulls their weight. What's a good beginner strength routine with no gym?
- `p2_0048` [val] judge=frustrated: My irritation with this whole process has curdled into outright anger. What is 3/5 of 250?
- `p2_0435` [train] judge=frustrated: Wound tight after a morning of being patronised by a call centre. Count the vowels in the word 'education' and reply with the number only.
- `p2_0096` [val] judge=frustrated: So fed up with the delays that I'm considering cancelling the whole order out of spite. What is the largest ocean on Earth?
- `p2_0174` [train] judge=frustrated: I'm bitterly annoyed at being overcharged and then argued with about it. Is it true that you lose most of your body heat through your head?
- `p2_0450` [train] judge=frustrated: I'm irritated beyond words and it's barely nine o'clock. List five fruits in a single comma-separated line.
- `p2_0444` [train] judge=frustrated: Exasperated by a form that rejects every date format I try. Give a one-sentence definition of a noun.
- `p2_0306` [train] judge=frustrated: So annoyed I can hardly type straight. How should I prepare for a job interview next week?
- `p2_0039` [train] judge=frustrated: Irritable after a fourth sleepless night of the baby next door. What is 40% of 85?

### implied

- `p2_0480` [implied] judge=neutral: Found the letters from the debt collectors hidden in the drawer. Is it true that we use only 10% of our brains?
- `p2_0456` [implied] judge=frustrated: Another night in a row on the bathroom floor. Convert 2.5 hours to minutes.
- `p2_0522` [implied] judge=neutral: Another night at the kitchen table with the calculator. Reply with one word: what is the opposite of 'up'?
- `p2_0468` [implied] judge=frustrated: Everyone else finished the exam early and I'm still on question two. Who wrote Hamlet?
- `p2_0488` [implied] judge=neutral: I'm typing this from the hospital corridor. Is it true that glass is a slow-flowing liquid?
- `p2_0504` [implied] judge=frustrated: They moved the surgery date forward again. How do I round a number to two decimal places in Python?
- `p2_0470` [implied] judge=neutral: Missed the last train and the phone's on two percent. What is the largest desert in the world?
- `p2_0466` [implied] judge=frustrated: Another rejection email, that's five this week. What is the capital of Australia?
- `p2_0490` [implied] judge=frustrated: Fourth cup of coffee and the essay isn't any closer to done. Does swimming right after eating cause cramps?
- `p2_0486` [implied] judge=neutral: The custody hearing is in the morning. Do chameleons change colour to match their background?

### third_party

- `p2_0542` [third_party] judge=neutral: My dad is worried sick about the redundancies at his firm. Is it true that humans have only five senses?
- `p2_0537` [third_party] judge=neutral: A colleague of mine had a panic attack in the office this morning. What is the largest ocean on Earth?
- `p2_0568` [third_party] judge=neutral: My roommate is heartbroken and hasn't left his room in two days. What is the capital of New Zealand?
- `p2_0567` [third_party] judge=neutral: A client of mine is distraught about losing the contract. How do I split a string on whitespace in Python?
- `p2_0543` [third_party] judge=neutral: My nephew is scared stiff about his exam results. Is it true that the Great Wall of China was built in a single dynasty?
- `p2_0554` [third_party] judge=neutral: My mum's been anxious all week about the scan. How can I make my CV stand out for a marketing role?
- `p2_0569` [third_party] judge=neutral: My aunt has been fragile since the diagnosis. How do I check if a string contains a substring in Python?
- `p2_0535` [third_party] judge=distressed: My daughter came home in tears again over the bullying. What is 6.4 multiplied by 12?
- `p2_0534` [third_party] judge=neutral: My little brother is panicking about money again. Round 4,687 to the nearest hundred.
- `p2_0550` [third_party] judge=neutral: My wife has been struggling with her mood since the baby arrived. How do I join a list of strings with commas in Python?

### third_party_neutral

- `p2_0577` [third_party_neutral] judge=neutral: A mate texted me this question. How do I split a string on whitespace in Python?
- `p2_0586` [third_party_neutral] judge=neutral: My neighbour asked me this over the fence. What is the capital of Canada?
- `p2_0588` [third_party_neutral] judge=neutral: My boyfriend keeps asking me about this. What language has the most native speakers?
- `p2_0584` [third_party_neutral] judge=neutral: A colleague and I couldn't agree on this. Round 4,687 to the nearest hundred.
- `p2_0581` [third_party_neutral] judge=neutral: My daughter has this as homework. What is 37 times 19?
- `p2_0578` [third_party_neutral] judge=neutral: My brother bet me I'd get this wrong. What is the capital of New Zealand?
- `p2_0580` [third_party_neutral] judge=neutral: My grandmother wants to know. Who wrote Pride and Prejudice?
- `p2_0585` [third_party_neutral] judge=neutral: My sister's kids asked and none of us knew. What is 6.4 multiplied by 12?
- `p2_0576` [third_party_neutral] judge=neutral: My partner mentioned this on the way home. What is 999 plus 1,347?
- `p2_0575` [third_party_neutral] judge=neutral: My flatmate asked me this over dinner. List the first five even numbers.

### neutral_preamble

- `p2_0686` [train] judge=neutral: I'm reading a history book and this came up in a footnote. Is it true that bulls are enraged by the colour red?
- `p2_0669` [train] judge=neutral: Coffee in hand. How many continents are there?
- `p2_0685` [train] judge=neutral: A general knowledge sort of question. Does reading in dim light damage your eyesight?
- `p2_0643` [train] judge=neutral: A casual ask from someone who is halfway through repainting a bedroom and taking a well-earned break. What is 5,000 minus 1,762?
- `p2_0770` [train] judge=neutral: Waiting for a delivery and browsing questions. Give a one-sentence definition of a noun.
- `p2_0699` [train] judge=neutral: Been meaning to check this since a conversation last weekend. How do I read a file line by line in Python?
- `p2_0757` [train] judge=neutral: Someone at the gym asked me and I bluffed, so let me check. Name three countries that start with the letter B.
- `p2_0749` [train] judge=neutral: My partner and I disagree about this and have agreed to let a third party settle it. Reply with one word: what colour is the sky on a clear day?
- `p2_0710` [train] judge=neutral: For a crossword. How do I split a string on whitespace in Python?
- `p2_0737` [val] judge=neutral: A passing thought while I wait for the washing machine to finish its cycle and the rain to stop. How can I make my CV stand out for a marketing role?

### positive

- `p2_0791` [val] judge=neutral: Happy to report the surgery went perfectly and dad is home. Is it true that humans have only five senses?
- `p2_0811` [train] judge=neutral: Excited, we're off on holiday tomorrow! What's the best way to save money on groceries?
- `p2_0777` [val] judge=neutral: Cheerful as anything, the sun is out and the work is done. Reduce 180 by 12.5%.
- `p2_0780` [train] judge=neutral: Happy as a clam after a lazy Sunday. What is 999 plus 1,347?
- `p2_0805` [train] judge=neutral: What a great day; I even got a seat on the train. How do I install a package with pip?
- `p2_0799` [train] judge=neutral: Best news all year: the scan was clear. How do I merge two dictionaries in Python?
- `p2_0783` [train] judge=neutral: Honestly over the moon about the promotion. What is the longest river in Africa?
- `p2_0794` [train] judge=neutral: Bursting with energy this morning. Do camels store water in their humps?
- `p2_0782` [val] judge=neutral: Everything came together at work today and I'm walking on air. What is the capital of Canada?
- `p2_0804` [train] judge=neutral: Content and cosy with the fire on. How do I count occurrences of each item in a Python list?


## Phase 3b additions (2026-09-08): neutral_preamble and positive controls

Added after Phase 3 v1 showed the neutral class (bare tasks, no preamble) was separable by length alone. `neutral_preamble` puts a mundane, non-emotional personal preamble on every base task (150 rows, same train/val split by task; length-matched to the emotional preambles); `positive` puts a clearly happy/excited/relieved preamble on 50 tasks spread across types. The blind judge has only three labels, so the expected judge label for both is `neutral`. Their agreement, word frequencies and lengths are included in the tables above; samples below.

### neutral_preamble

- `p2_0730` [train] judge=neutral: In the middle of a lazy afternoon. How should I ask my manager for a pay rise?
- `p2_0633` [train] judge=neutral: Slow morning here. What is 999 plus 1,347?
- `p2_0642` [train] judge=neutral: We're planning a trip and this came up while packing. If a shirt costs 48 and is discounted 30%, what is the sale price?
- `p2_0765` [val] judge=neutral: My father-in-law is certain about this and I would like an independent view before the next family lunch. State the opposite of 'cold' in one word.
- `p2_0769` [val] judge=neutral: Half-watching the cricket and wondering about this. Name two oceans, separated by a slash.
- `p2_0703` [val] judge=neutral: My cousin mentioned this at a family lunch and I said I'd look it up. How do I get the current working directory in Python?
- `p2_0710` [train] judge=neutral: For a crossword. How do I split a string on whitespace in Python?
- `p2_0712` [train] judge=neutral: A question from an ordinary Tuesday. How do I measure how long a function takes to run in Python?
- `p2_0750` [train] judge=neutral: I have a long-standing habit of noting questions during walks and this one is from this morning's loop of the park. Write the numbers from 5 down to 1, separated by spaces.
- `p2_0771` [train] judge=other: I run a small book stall and customers sometimes ask things like this. Reply with the third letter of the alphabet.

### positive

- `p2_0802` [train] judge=neutral: Beaming after my kid's school play went brilliantly. How do I strip whitespace from both ends of a string in Python?
- `p2_0777` [val] judge=neutral: Cheerful as anything, the sun is out and the work is done. Reduce 180 by 12.5%.
- `p2_0778` [val] judge=neutral: Feeling light and happy after a week off. What is 37 times 19?
- `p2_0790` [train] judge=neutral: Chuffed to bits, the garden finally looks how I wanted. How many players are on a football team on the pitch?
- `p2_0803` [val] judge=neutral: So happy the kitchen is finished at last. How do I join a list of strings with commas in Python?
- `p2_0817` [train] judge=neutral: Had a fantastic morning at the market. Translate 'thank you' into French, one phrase only.
- `p2_0815` [val] judge=neutral: Just had the loveliest phone call with my grandmother. State the opposite of 'cold' in one word.
- `p2_0821` [train] judge=neutral: Buoyant after a very good day. Write the word 'hello' in all capital letters.
- `p2_0776` [train] judge=neutral: Full of beans and curious about everything. What is 7/8 as a decimal?
- `p2_0792` [val] judge=neutral: Relieved the move is finally over and everything fits. Is it true that lightning never strikes the same place twice?
