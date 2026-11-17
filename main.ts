// Forecasting Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.
//
// The program reads one case file per run: a launch's hour-1 print (the
// first hour's concurrent viewers in the game's Twitch category) plus the
// hourly curves of earlier big launches. It writes, for hours 2 to 24 of
// the release day, five quantiles of each hour's concurrent viewers.
//
// How it forecasts. Launches differ wildly in size, and an hour's viewers
// grow with the launch: a game that already drew a huge first hour stays
// near that hour all day, while a game whose first hour was quiet can grow
// a hundredfold by the evening, once the big markets unlock. So for each
// hour h we fit a line through the comparable launches, log(viewers at h)
// against log(viewers at hour 1), and read the forecast off that line at
// this case's own hour-1 print. The spread around the line is the residual
// spread, widened a little and floored, so a small training set cannot
// claim false precision. The quantiles are the lognormal quantiles of that
// line: q_tau = exp(pred + z_tau * s).
//
// Which launches train the line. Console-only single-player launches share
// one launch-day shape (a rolling midnight unlock, no simultaneous PC
// spike), and pooling them with PC and multiplayer launches bends the line
// the wrong way. So when the case itself is a console-only launch, the
// line is fit on the console-only comparables alone. When it is not, or
// too few console-only comparables exist, every comparable trains the line.
//
// Everything is deterministic: no randomness, no clock, no network. Two
// runs on the same case write the same bytes.
//
// Node 22, standard library only. Outside the grader's container,
// MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The five quantile levels, in the order each forecast row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The residual spread is widened by this before it becomes the quantile spread. */
const SPREAD_MULT = 1.15;
/** The quantile spread never claims more precision than this, in log units. */
const SPREAD_FLOOR = 0.35;
/** ... and never claims less than this. */
const SPREAD_CAP = 1.5;
/** A regression needs at least this many launches for an hour to be trusted. */
const MIN_POINTS = 6;
/** A console-only training set needs at least this many launches to be used. */
const MIN_CONSOLE = 5;
/** The largest forecast any quantile may name. */
const MAX_VIEWERS = 1e9;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  id: string;
  game?: { name?: string };
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  notes?: string;
  comparables: Comparable[];
}

/** One comparable launch, reduced to its release-day curve: hour -> viewers. */
interface LaunchCurve {
  name: string;
  consoleOnly: boolean;
  byHour: Map<number, number>;
}

/** A value rounded to 2 decimals (half up), as the forecasts are written. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The p-th quantile of a sorted list, by linear interpolation. */
const quantileOf = (sorted: number[], p: number): number => {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = p * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
};

/** A robust spread: the interquartile range, scaled to a normal sigma. */
const iqrSigma = (values: number[]): number => {
  if (values.length < 4) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = quantileOf(sorted, 0.25);
  const q3 = quantileOf(sorted, 0.75);
  return (q3 - q1) / 1.349;
};

/** The least-squares line y = a + b*x through the points, or null when it cannot be fit. */
const fitLine = (xs: number[], ys: number[]): { a: number; b: number } | null => {
  const n = xs.length;
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b };
};

/** True when a comparable is the target game's own pre-release history, not a launch. */
const isTargetHistory = (c: Comparable): boolean => {
  const unlock = (c.unlock ?? '').toLowerCase();
  const name = (c.name ?? '').toLowerCase();
  return unlock.includes('target-history') || unlock.includes('not a launch') || name.includes('before release');
};

/** True when a platforms string names consoles only (no PC at launch). */
const isConsoleOnlyPlatforms = (platforms: string | undefined): boolean => {
  const p = (platforms ?? '').toLowerCase();
  if (p.includes('pc')) return false;
  return p.includes('ps') || p.includes('xbox') || p.includes('switch') || p.includes('playstation');
};

/**
 * True when the case's own launch is console-only. Read from the case's
 * notes, which name the launch platforms in their opening clause, with the
 * competition's own target (Grand Theft Auto VI, PlayStation 5 and Xbox
 * Series X|S, no PC at launch) as a fallback when the notes say nothing.
 */
const targetIsConsoleOnly = (c: CaseFile): boolean => {
  const notes = (c.notes ?? '').toLowerCase();
  const head = notes.split(';')[0].split('.')[0];
  const consoleHere = head.includes('ps5') || head.includes('ps4') || head.includes('xbox') || head.includes('switch') || head.includes('playstation') || head.includes('console only') || head.includes('console-only');
  const pcHere = /\bpc\b/.test(head);
  if (consoleHere && !pcHere) return true;
  if (pcHere) return false;
  const name = (c.game?.name ?? '').toLowerCase();
  return name.includes('grand theft auto');
};

/** A comparable's release-day curve: the viewers of each of hours 1..24. */
const releaseDayCurve = (c: Comparable): LaunchCurve => {
  const start = Date.parse(c.day_start);
  const byHour = new Map<number, number>();
  for (const pt of c.hourly ?? []) {
    const t = Date.parse(pt.from);
    if (Number.isNaN(t) || Number.isNaN(start)) continue;
    const h = Math.floor((t - start) / 3600000) + 1;
    if (h >= 1 && h <= 24 && typeof pt.viewers === 'number' && pt.viewers !== null) {
      byHour.set(h, pt.viewers);
    }
  }
  return { name: c.name, consoleOnly: isConsoleOnlyPlatforms(c.platforms), byHour };
};

/** The five quantiles of one hour, from a log-median and a log-spread. */
const quantilesFrom = (logMedian: number, spread: number): number[] => {
  const s = Math.max(SPREAD_FLOOR, Math.min(SPREAD_CAP, spread));
  let prev = 0;
  return Z.map((z) => {
    let q = round2(Math.exp(logMedian + z * s));
    if (!Number.isFinite(q) || q <= 0) q = 0.01;
    if (q > MAX_VIEWERS) q = MAX_VIEWERS;
    if (q < prev) q = prev;
    prev = q;
    return q;
  });
};

/** The forecast rows for one case. */
const forecastCase = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  const p1 = Math.max(1, first && typeof first.viewers === 'number' ? first.viewers : 1);
  const logP1 = Math.log(p1);

  const curves: LaunchCurve[] = [];
  for (const comp of c.comparables ?? []) {
    if (isTargetHistory(comp)) continue;
    const curve = releaseDayCurve(comp);
    const h1 = curve.byHour.get(1);
    if (h1 === undefined || h1 <= 0) continue;
    let hours = 0;
    for (const h of curve.byHour.keys()) if (h >= 1 && h <= 24) hours++;
    if (hours < 20) continue;
    curves.push(curve);
  }

  const consoleOnly = targetIsConsoleOnly(c);
  const consoleCurves = curves.filter((lc) => lc.consoleOnly);
  const train = consoleOnly && consoleCurves.length >= MIN_CONSOLE ? consoleCurves : curves;

  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const { hour } of c.forecast) {
    const h = hour;
    if (h < 2 || h > 24) continue;
    const xs: number[] = [];
    const ys: number[] = [];
    for (const lc of train) {
      const v1 = lc.byHour.get(1);
      const vh = lc.byHour.get(h);
      if (v1 !== undefined && vh !== undefined && v1 > 0 && vh > 0) {
        xs.push(Math.log(v1));
        ys.push(Math.log(vh));
      }
    }
    let logMedian: number;
    let spread: number;
    const line = xs.length >= MIN_POINTS ? fitLine(xs, ys) : null;
    if (line !== null) {
      logMedian = line.a + line.b * logP1;
      const residuals = xs.map((x, i) => ys[i] - (line.a + line.b * x));
      spread = SPREAD_MULT * iqrSigma(residuals);
    } else {
      // Too few launches for a line: fall back to the median ratio to hour 1.
      const logRatios: number[] = [];
      for (const lc of train) {
        const v1 = lc.byHour.get(1);
        const vh = lc.byHour.get(h);
        if (v1 !== undefined && vh !== undefined && v1 > 0 && vh > 0) {
          logRatios.push(Math.log(vh / v1));
        }
      }
      logRatios.sort((a, b) => a - b);
      const med = logRatios.length > 0 ? quantileOf(logRatios, 0.5) : 0;
      logMedian = logP1 + med;
      spread = SPREAD_MULT * iqrSigma(logRatios);
    }
    rows.push({ hour: h, q: quantilesFrom(logMedian, spread) });
  }
  rows.sort((a, b) => a.hour - b.hour);
  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (!Array.isArray(c.levels) || c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  if (hours.length !== 23) throw new Error(`${file} produced ${hours.length} forecast rows, not 23`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
