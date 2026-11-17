[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xxstealz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xxstealz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers forecast

Forecasts the concurrent viewers in the "Grand Theft Auto VI" Twitch category,
hour by hour, on release day 2026-11-19 (UTC), hours 2–24, as five quantiles
`[q05, q25, q50, q75, q95]` per hour.

## How it works

`main.ts` (Node 22, standard library only, no network) reads each case file —
the hour-1 print plus the hourly curves of earlier big launches — and for
every hour fits a log-log regression, `log(viewers at h)` on `log(viewers at
hour 1)`, across the comparable launches, then reads the forecast off that
line at the case's own hour-1 print. The spread is the residual spread
around the line, widened slightly and floored so a small training set cannot
claim false precision. When the case is a console-only launch, the line is
fit on the console-only comparables alone: console-only single-player
launches share one launch-day shape, and pooling them with PC and
multiplayer launches bends the line the wrong way.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

One `.out` per case: `{"hours": [{"hour": 2, "q": [...]}, ..., {"hour": 24, "q": [...]}]}`.
Deterministic — two runs on the same case write the same bytes.

## License

MIT — see `LICENSE`.
