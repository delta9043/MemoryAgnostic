import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    "/data/delta9043/models/Qwen2.5-7B-Instruct",
    # "/data/delta9043/models/Qwen3-8B",
]

PROMPT = """Your task is to filter a single utterance by removing parts that carry no informative content.
 
[Input Utterance]
Speaker: Moon
Content: Fever dream high in the quiet of the night
You know that I caught it (oh yeah, you're right, I want it)
Bad, bad boy, shiny toy with a price
You know that I bought it (oh yeah, you're right, I want it)
Killing me slow, out the window
I'm always waiting for you to be waiting below
Devils roll the dice, angels roll their eyes
What doesn't kill me makes me want you more
And it's new, the shape of your body
It's blue, the feeling I've got
And it's ooh, whoa-oh
It's a cruel summer
"It's cool, " that's what I tell 'em
No rules in breakable heaven
But ooh, whoa-oh
It's a cruel summer with you (yeah, yeah)
Hang your head low in the glow of the vending machine
I'm not dying (oh yeah, you're right, I want it)
You say that we'll just screw it up in these trying times
We're not trying (oh yeah, you're right, I want it)
So cut the headlights, summer's a knife
I'm always waiting for you just to cut to the bone
Devils roll the dice, angels roll their eyes
And if I bleed, you'll be the last to know, oh
It's new, the shape of your body
It's blue, the feeling I've got
And it's ooh, whoa-oh
It's a cruel summer
"It's cool, " that's what I tell 'em
No rules in breakable heaven
But ooh, whoa-oh
It's a cruel summer with you
I'm drunk in the back of the car
And I cried like a baby coming home from the bar (oh)
Said, "I'm fine, " but it wasn't true
I don't wanna keep secrets just to keep you
And I snuck in through the garden gate
Every night that summer, just to seal my fate (oh)
And I screamed, "For whatever it's worth
I love you, ain't that the worst thing you ever heard?"
He looks up, grinnin' like a devil
It's new, the shape of your body
It's blue, the feeling I've got
And it's ooh, whoa-oh
It's a cruel summer
"It's cool, " that's what I tell 'em
No rules in breakable heaven
But ooh, whoa-oh
It's a cruel summer with you
I'm drunk in the back of the car
And I cried like a baby coming home from the bar (oh)
Said, "I'm fine, " but it wasn't true
I don't wanna keep secrets just to keep you
And I snuck in through the garden gate
Every night that summer, just to seal my fate (oh)
And I screamed, "For whatever it's worth
I love you, ain't that the worst thing you ever heard?"
(Yeah, yeah, yeah, yeah)
 
[Requirements]
1. Discard social filler: Remove acknowledgements and conversational routines that introduce no new factual or semantic information.
2. Discard redundant confirmations: Remove confirmations unless they modify or finalize a decision.
3. Keep informative content unchanged: Do NOT paraphrase, summarize, or add any new information. Only remove the non-informative parts.
4. If no informative content is present: Output an empty string.
 
[Output Format]
Return a JSON object:
 
```json
{{"filtered_content": "remaining informative content, or empty string if none"}}
```
 
[Example]
Input:
Speaker: Alice
Content: Oh yeah, sure! That sounds great. Let's meet at Central Park at 3pm on Saturday.
 
Output:
```json
{{"filtered_content": "Let's meet at Central Park at 3pm on Saturday."}}
```
 
[Example]
Input:
Speaker: Bob
Content: Haha yeah, totally! I know right?
 
Output:
```json
{{"filtered_content": ""}}
```
 
Now process the above utterance. Return ONLY the JSON object, no other explanations.
"""
MAX_NEW_TOKENS = 200
N_RUNS = 3

def run_benchmark(model_path):
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    )
    print(model.config._attn_implementation)
    model.eval()

    mem = torch.cuda.memory_allocated(0) / 1024**3
    print(f"GPU memory allocated: {mem:.2f} GB")

    messages = [{"role": "user", "content": PROMPT}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to("cuda:0")
    attention_mask = encoded["attention_mask"].to("cuda:0")

    input_len = input_ids.shape[-1]
    print(f"input_len: {input_len} tokens")
    results = []

    for i in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        new_tokens = output_ids.shape[-1] - input_len
        tps = new_tokens / elapsed
        results.append(tps)
        print(f"  [Run {i}] new={new_tokens} | {elapsed:.2f}s | {tps:.2f} tok/s")

    print(f"  → mean (전체): {sum(results)/len(results):.2f} tok/s")
    print(f"  → mean (run 1+): {sum(results[1:])/len(results[1:]):.2f} tok/s")

    del model
    del tokenizer
    torch.cuda.empty_cache()

for mp in MODELS:
    run_benchmark(mp)