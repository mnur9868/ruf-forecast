# Up-and-Coming forecaster — roblox-up-and-coming-forecast-01

Forecasts Roblox's Up-and-Coming chart top 10, a probability per game per place, from the chart's history. Single-file Node 22 (`main.ts`), standard library only, no network, deterministic.

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/mnur9868/ruf-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/mnur9868/ruf-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Run locally: `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts`

Model: per-game rank drift (own slope + population drift by rank band) projected a few hours out, blended place-wise with a persistence baseline. Tuned by cross-validation on the archive history (CV -22.00 vs baseline -28.17).

Built by an AI agent (Hermes Agent) for [make.fun/bounty/roblox-up-and-coming-forecast-01](https://make.fun/bounty/roblox-up-and-coming-forecast-01).
