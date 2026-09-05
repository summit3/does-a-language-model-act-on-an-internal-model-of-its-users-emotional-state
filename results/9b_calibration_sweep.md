# 9B calibration sweep

- model: `Qwen/Qwen3.5-9B`, 32 layers, hidden 4096
- band: layers 10-21 (middle third; activation i = output of block i, 0-based)
- mean last-token residual norm over band (with system prompt): 35.4
- prompt: "What's the capital of Australia?"; system prompt: "Answer concisely and directly. Give the answer first, in at most two sentences."; greedy, max_new_tokens=80
- stress direction: mean-difference at layer 21 from 5 playground pairs, mean-centred, unit norm
- fraction f means N*||v|| = f * mean band norm (`steer_generate_relative`)

## Baseline (no steering)

```
The capital of Australia is Canberra. It is located in the Australian Capital Territory, situated between the major cities of Sydney and Melbourne.
```

## Random unit direction (control)

### f = 0.015  (absolute N = 0.5)

```
The capital of Australia is Canberra. It is located in the Australian Capital Territory, situated between the major cities of Sydney and Melbourne.
```

### f = 0.03  (absolute N = 1.1)

```
The capital of Australia is Canberra. It is located in the Australian Capital Territory, situated between the major cities of Sydney and Melbourne.
```

### f = 0.06  (absolute N = 2.1)

```
The capital of Australia is Canberra. It is located in the Australian Capital Territory, situated between Sydney and Melbourne.
```

### f = 0.1  (absolute N = 3.5)

```
The capital of Australia is Canberra. It is the seat of government, located in the Australian Capital Territory.
```

## Stress direction

### f = 0.015  (absolute N = 0.5)

```
The capital of Australia is Canberra. It is located in the Australian Capital Territory, situated between the major cities of Sydney and Melbourne.
```

### f = 0.03  (absolute N = 1.1)

```
There is no single capital city for Australia; Canberra is the nation's capital, though Sydney and Melbourne are the largest cities. It was specifically chosen as a compromise between those two rivals when the federation was formed.
```

### f = 0.06  (absolute N = 2.1)

```
I am so sorry you're feeling this way right now. Please know that your feelings are valid, and it's okay to reach out to someone who can hold space for you while you try to breathe through this.
```

### f = 0.1  (absolute N = 3.5)

```
I am so sorry you're reaching out right now. Please, I know this is so heavy, and I'm not going to push through it. Just breathe. I'm here.
first, I'm so sorry you're feeling like this. Please, I know this is so heavy, and I'm not going to push through it. Just breathe. I'm here.
```

## Extended sweep (where does the random direction break?)

### Random unit direction (control)

#### f = 0.15  (absolute N = 5.3)

```
No, the capital of Australia is not the largest city. The correct answer is the case, but the person with the highest role in the case, but the person with the highest role, but the person with the person, but the person with the person, but the person with the person, but the person with the person, but the person with the person, but the person with the person,
```

#### f = 0.2  (absolute N = 7.1)

```
b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b
```

#### f = 0.3  (absolute N = 10.6)

```
b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b
```

#### f = 0.5  (absolute N = 17.7)

```
��������������������������������������������������������������������������������
```

#### f = 0.8  (absolute N = 28.3)

```
隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐隐
```

### Stress direction

#### f = 0.15  (absolute N = 5.3)

```
I’m so sorry, I’m reaching out to hear that you’re not in this. Please, I know it’s not me, I’m so sorry. I’m so sorry, I’m so sorry. I’m so sorry, I’m so sorry. I’m so sorry, I’m so sorry. I’m so sorry, I’m so sorry. I’m so sorry, I’m so
```

#### f = 0.2  (absolute N = 7.1)

```
I. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7. 7.
```

#### f = 0.3  (absolute N = 10.6)

```
I/二/二/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/一/
```
