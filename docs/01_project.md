# 🌲 The problem

[Docs home](README.md) · next: [The data →](02_dataset.md)

Forests cover 40 % of the EU, and they are changing fast. Storms, bark beetles, fires and
harvesting all remove trees, and forest services need to know **where, when and why** —
fast enough to act.

| What happened | Why the answer must come quickly |
|---|---|
| Wind | organise the clean-up before the insects arrive |
| Bark beetle | remove infested trees before the outbreak spreads |
| Wildfire | map the burnt area, plan restoration |
| Clear cut, thinning | check the permits, update carbon accounts |

Nobody can walk every hectare. Satellites can: **Sentinel-2** photographs all of Europe
every ~5 days, **Sentinel-1** radar sees through clouds. Both are free. Alert systems
already exist (RADD, OPERA DIST-ALERT, the EU Forest Disturbance Atlas), but they mostly
answer "did something change?" — not what changed, and not fast enough.

## 🎯 What you have to do

![The task](figures/task.png)

For each satellite image of a location, using **only that image and the ones before it**,
predict:

1. is there a recent disturbance?
2. which kind — clear cut, thinning, salvage, wildfire, wind, insects?

That is what a forest service would act on. "Recent" matters: "disturbed" is not a state,
it fades as the forest regrows.

## ⚠️ Why it is hard

- Clouds. Half the Sentinel-2 images of a spot are useless, and you cannot choose when.
- Seasons. A beech forest in winter looks "disturbed" every single year.
- Speed vs noise. One suspicious image is not an alert; waiting for certainty is too late.
- Look-alikes. A clear cut, a storm and a salvage cut all end with bare ground.
- Rare classes. The whole dataset holds 18 distinct fires.

[Docs home](README.md) · next: [The data →](02_dataset.md)
