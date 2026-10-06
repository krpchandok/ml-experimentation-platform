import html
import json
from dataclasses import dataclass
from typing import Optional

from dashboard import art, wording

VISUAL_CYCLE_MS = 2200
MIN_BAKE_MS = 650
DOOR_MS = 320
TRAVEL_MS = 1100
MAX_QUEUE = 3
MAX_STATIONS = 6
STATIC_HEIGHT = 300
LANE_HEIGHT = 250


@dataclass
class SceneNumbers:
    verdict: str
    workers: int
    worker_busy: Optional[float]
    compute_s: float
    prep_s: float
    oven_busy: Optional[float]
    steps_per_s: Optional[float]
    read_mib_s: Optional[float]
    slowest_label: str
    slowest_stage: Optional[str]
    estimate: bool = False


def timing(numbers):
    compute = max(numbers.compute_s, 1e-6)
    prep = max(numbers.prep_s, 0.0)
    longest = max(compute, prep)
    scale = VISUAL_CYCLE_MS / longest
    bake = max(MIN_BAKE_MS, compute * scale - 2 * DOOR_MS)
    arrival = max(prep * scale, MIN_BAKE_MS * 0.5)
    real_cycle_ms = longest * 1000
    return {
        "bake": round(bake),
        "door": DOOR_MS,
        "travel": TRAVEL_MS,
        "arrival": round(arrival),
        "maxQueue": MAX_QUEUE,
        "slowdown": max(1, round(VISUAL_CYCLE_MS / real_cycle_ms)) if real_cycle_ms > 0 else 1,
        "ovenBound": compute >= prep,
    }


def image(name, alt, css_class=""):
    uri = art.data_uri(name)
    if uri is None:
        return f'<div class="missing {css_class}" role="img" aria-label="{html.escape(alt)}">{html.escape(alt)}</div>'
    return f'<img class="{css_class}" src="{uri}" alt="{html.escape(alt)}">'


def oven_state(oven_busy):
    if oven_busy is None:
        return "oven_half_open", "oven, state unknown"
    if oven_busy >= 0.8:
        return "oven_closed", "closed oven, baking"
    if oven_busy <= 0.5:
        return "oven_open", "open oven, empty and waiting"
    return "oven_half_open", "half-open oven"


def station_cards(numbers):
    if numbers.workers == 0:
        return ('<div class="station solo">' + image("rolling_pin", "rolling pin", "station-art") +
                '<div><b>No bakers</b><br><span>The head baker makes the dough too</span></div></div>')
    cards = []
    for index in range(min(numbers.workers, MAX_STATIONS)):
        name = art.STATION_ART[index % len(art.STATION_ART)]
        busy = min(numbers.worker_busy if numbers.worker_busy is not None else 0, 1.0)
        cards.append(f'<div class="station">{image(name, name.replace("_", " "), "station-art")}'
                     f'<div class="station-text"><b>Baker {index + 1}</b>'
                     f'<div class="bar" role="img" aria-label="{busy * 100:.0f}% busy"><span style="width:{busy * 100:.0f}%"></span></div>'
                     f'<span>{busy * 100:.0f}% busy</span></div></div>')
    if numbers.workers > MAX_STATIONS:
        cards.append(f'<div class="station more">+{numbers.workers - MAX_STATIONS} more</div>')
    return "".join(cards)


def stage(key, body, value, numbers):
    slow = numbers.slowest_stage == key
    ribbon = (f'<div class="ribbon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
              f'<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17.5v.5"/></svg>'
              f'{html.escape(numbers.slowest_label)}</div>') if slow else ""
    return (f'<section class="stage{" slow" if slow else ""}" aria-label="{wording.STAGE_NAMES[key]}">{ribbon}'
            f'<h3>{wording.STAGE_NAMES[key]}</h3><p class="detail">{wording.STAGE_DETAILS[key]}</p>'
            f'<div class="body">{body}</div><p class="value">{html.escape(value)}</p></section>')


def arrow():
    return ('<div class="arrow" aria-hidden="true"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M4 12h15"/><path d="M13 6l6 6-6 6"/></svg></div>')


def static_scene(numbers):
    oven_name, oven_alt = oven_state(numbers.oven_busy)
    glow = " glow" if numbers.oven_busy is not None and numbers.oven_busy >= 0.8 else ""
    pantry = "".join(image(name, name.replace("_", " "), "pantry-art") for name in ("flour_sack", "egg_basket", "milk_bottle"))
    oven = f'<div class="oven-static{glow}">{image(oven_name, oven_alt, "oven-art")}</div>'
    goods = "".join(image(art.FOOD[index], art.FOOD[index][3:], "food-art") for index in (0, 7, 8))
    read = wording.NOT_ESTIMATED if numbers.read_mib_s is None else f"{numbers.read_mib_s:.1f} MB read per second"
    busy = "-" if numbers.oven_busy is None else f"Busy {min(numbers.oven_busy, 1.0) * 100:.0f}% of the time"
    rate = "-" if not numbers.steps_per_s else f"{numbers.steps_per_s:.1f} trays per second"
    bakers = wording.bakers(numbers.workers).capitalize()
    return ('<div class="pipeline">' +
            stage("pantry", f'<div class="pantry">{pantry}</div>', read, numbers) + arrow() +
            stage("bakers", f'<div class="stations">{station_cards(numbers)}</div>', bakers, numbers) + arrow() +
            stage("oven", oven, busy, numbers) + arrow() +
            stage("goods", f'<div class="goods">{goods}</div>', rate, numbers) + '</div>')


def lane(numbers):
    peel = image("baker_peel", "baker's peel", "peel")
    ovens = "".join(f'<div class="oven-layer {state}">{image(name, alt, "oven-art")}</div>'
                    for state, name, alt in (("open", "oven_open", "open oven"),
                                             ("half", "oven_half_open", "half-open oven"),
                                             ("closed", "oven_closed", "closed oven")))
    return f"""
<div class="lane" aria-label="Animation of trays moving through the bakery">
  <div class="lane-bakers">{image('rolling_pin', 'rolling pin', 'lane-station')}{image('flour_sack', 'flour sack', 'lane-station')}</div>
  <div class="counter-top" aria-hidden="true"></div>
  <div class="queue-zone" aria-hidden="true"></div>
  <div class="lane-oven"><div class="heat"></div>{ovens}<div class="oven-label" aria-live="off"></div></div>
  <div class="lane-out"><div class="out-stack"></div><div class="tally">Trays baked: <b class="count">0</b></div></div>
  <template id="peel-tpl">{peel}</template>
</div>"""


def scene_html(numbers, theme_tokens, show_animation=True):
    config = timing(numbers)
    food_uris = [art.data_uri(name) for name in art.FOOD]
    t = theme_tokens
    slowdown_note = wording.ANIMATION_NOTE.format(factor=config["slowdown"])
    label = (wording.ESTIMATE_LABEL + ": " if numbers.estimate else "")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600&family=Nunito:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root {{ --bg: transparent; --surface: {t['surface']}; --surface2: {t['surface2']}; --text: {t['text']}; --text2: {t['text2']};
  --muted: {t['muted']}; --border: {t['border']}; --gradient: {t['gradient']}; --glow: {t['glow']}; --track: {t['track']};
  --warn: {t['warn']}; --warn-bg: {t['warn_bg']}; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Nunito, system-ui, sans-serif; color: var(--text); background: transparent; }}
h3 {{ font-family: Fredoka, Nunito, sans-serif; font-weight: 600; font-size: 1.05rem; margin: 0; }}
.pipeline {{ display: grid; grid-template-columns: 1fr 28px 1.35fr 28px 1fr 28px 1fr; align-items: stretch; gap: 6px; }}
.stage {{ position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 14px 14px 10px;
  display: flex; flex-direction: column; min-height: 248px; }}
.stage.slow {{ border: 3px solid transparent; background: linear-gradient(var(--surface), var(--surface)) padding-box, var(--gradient) border-box; }}
.ribbon {{ margin: -4px -4px 10px; display: flex; gap: 6px; align-items: center; justify-content: center;
  background: var(--warn-bg); color: var(--text); border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px;
  font-size: .78rem; font-weight: 800; text-align: center; }}
.ribbon svg {{ color: var(--warn); flex: none; }}

.detail {{ margin: 2px 0 8px; color: var(--muted); font-size: .8rem; }}
.value {{ margin: 8px 0 0; font-weight: 800; font-size: .92rem; color: var(--text); }}
.body {{ flex: 1; display: flex; align-items: center; justify-content: center; }}
.arrow {{ display: flex; align-items: center; justify-content: center; color: var(--muted); }}
.pantry, .goods {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }}
.pantry-art, .food-art {{ width: 54px; height: 54px; }}
.stations {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; width: 100%; }}
.station {{ display: flex; gap: 6px; align-items: center; background: var(--surface2); border-radius: 12px; padding: 6px; font-size: .74rem; }}
.station.solo {{ grid-column: span 2; font-size: .85rem; }}
.station.more {{ justify-content: center; font-weight: 800; color: var(--text2); }}
.station-art {{ width: 34px; height: 34px; flex: none; }}
.station-text {{ flex: 1; min-width: 0; }}
.bar {{ height: 6px; border-radius: 9px; background: var(--track); overflow: hidden; margin: 3px 0; }}
.bar span {{ display: block; height: 100%; background: var(--gradient); border-radius: 9px; }}
.oven-static {{ position: relative; display: flex; justify-content: center; }}
.oven-static .oven-art {{ width: 150px; height: 150px; position: relative; z-index: 1; }}
.oven-static.glow::before, .lane-oven .heat {{ content: ""; position: absolute; inset: 18% 14% 6%; border-radius: 40%;
  background: radial-gradient(circle, var(--glow) 0%, transparent 70%); filter: blur(6px); }}
.missing {{ display: inline-flex; align-items: center; justify-content: center; width: 54px; height: 54px; border-radius: 14px;
  background: var(--surface2); border: 1px dashed var(--border); color: var(--muted); font-size: .6rem; font-weight: 800;
  text-transform: uppercase; text-align: center; padding: 2px; }}
.oven-static .missing, .lane-oven .missing {{ width: 130px; height: 130px; }}
.caption {{ margin: 10px 4px 0; font-size: .9rem; color: var(--text2); }}
.lane {{ position: relative; height: 210px; margin-top: 16px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; overflow: hidden; }}
.counter-top {{ position: absolute; left: 0; right: 0; bottom: 34px; height: 14px; background: var(--surface2);
  border-top: 2px solid var(--border); }}
.lane-bakers {{ position: absolute; left: 14px; bottom: 46px; display: flex; gap: 2px; }}
.lane-station {{ width: 54px; height: 54px; }}
.lane-oven {{ position: absolute; left: 58%; bottom: 30px; width: 150px; height: 150px; transform: translateX(-50%); }}
.lane-oven .heat {{ opacity: 0; transition: opacity .35s ease; }}
.lane-oven.baking .heat {{ opacity: 1; animation: flicker 1.1s ease-in-out infinite alternate; }}
.oven-layer {{ position: absolute; inset: 0; opacity: 0; transition: opacity .18s ease, filter .35s ease; }}
.oven-layer .oven-art {{ width: 150px; height: 150px; }}
.lane-oven.state-open .oven-layer.open, .lane-oven.state-half .oven-layer.half, .lane-oven.state-closed .oven-layer.closed {{ opacity: 1; }}
.lane-oven.state-open .oven-layer.open {{ filter: saturate(.55) brightness(.92); }}
.oven-label {{ position: absolute; left: 50%; top: -6px; transform: translateX(-50%); white-space: nowrap; font-size: .78rem;
  font-weight: 800; background: var(--surface2); border: 1px solid var(--border); border-radius: 999px; padding: 2px 10px; }}
.lane-out {{ position: absolute; right: 16px; bottom: 46px; width: 150px; text-align: center; }}
.out-stack {{ height: 70px; position: relative; }}
.tally {{ font-size: .85rem; color: var(--text2); }}
.traveller {{ position: absolute; left: 70px; bottom: 44px; width: 64px; height: 64px; transform: translateX(0);
  transition: transform var(--travel) cubic-bezier(.4,.05,.3,1), opacity .2s ease; z-index: 3; }}
.traveller .peel {{ position: absolute; inset: 0; width: 64px; height: 64px; }}
.traveller .dough {{ position: absolute; left: 4px; top: 22px; width: 32px; height: 32px; }}
.popped {{ position: absolute; left: 50%; bottom: 0; width: 46px; height: 46px; margin-left: -23px;
  animation: pop .7s cubic-bezier(.2,1.6,.4,1) both; }}
@keyframes pop {{ 0% {{ transform: translate(-120px, 10px) scale(.4); opacity: 0; }} 60% {{ opacity: 1; }}
  100% {{ transform: translate(var(--dx, 0px), var(--dy, 0px)) scale(1); opacity: 1; }} }}
@keyframes flicker {{ from {{ transform: scale(.97); filter: blur(6px) brightness(.95); }} to {{ transform: scale(1.04); filter: blur(7px) brightness(1.1); }} }}
.notes {{ display: flex; justify-content: space-between; gap: 12px; margin: 8px 4px 0; font-size: .8rem; color: var(--muted); }}
.reduced {{ display: none; }}
@media (prefers-reduced-motion: reduce) {{
  .lane, .notes.motion {{ display: none; }}
  .reduced {{ display: block; margin: 10px 4px 0; font-size: .85rem; color: var(--muted); }}
}}
@media (max-width: 720px) {{ .pipeline {{ grid-template-columns: 1fr; }} .arrow {{ transform: rotate(90deg); }}
  .stage {{ min-height: 0; }} }}
@media (max-width: 520px) {{
  .lane {{ height: 190px; }}
  .lane-bakers {{ left: 8px; }}
  .lane-station {{ width: 36px; height: 36px; }}
  .lane-oven {{ left: auto; right: 92px; transform: none; width: 104px; height: 104px; }}
  .oven-layer .oven-art {{ width: 104px; height: 104px; }}
  .lane-oven .missing {{ width: 96px; height: 96px; }}
  .lane-out {{ right: 4px; width: 84px; }}
  .out-stack {{ height: 54px; }}
  .popped {{ width: 32px; height: 32px; margin-left: -16px; }}
  .tally {{ font-size: .75rem; }}
  .traveller {{ left: 44px; width: 48px; height: 48px; }}
  .traveller .peel {{ width: 48px; height: 48px; }}
  .traveller .dough {{ left: 3px; top: 16px; width: 24px; height: 24px; }}
}}
</style></head><body>
{static_scene(numbers)}
<p class="caption">{html.escape(label + numbers.slowest_label)}.</p>
{lane(numbers) if show_animation else ""}
<div class="notes motion"><span>{html.escape(slowdown_note)}</span><span>{html.escape(label.strip(': ') or 'Measured')}</span></div>
<p class="reduced">{html.escape(wording.REDUCED_MOTION_NOTE)}</p>
<script>
(() => {{
  const cfg = {json.dumps(config)};
  const foods = {json.dumps([uri for uri in food_uris if uri])};
  const lane = document.querySelector('.lane');
  if (!lane || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const oven = lane.querySelector('.lane-oven');
  const label = lane.querySelector('.oven-label');
  const stack = lane.querySelector('.out-stack');
  const count = lane.querySelector('.count');
  const peelHtml = lane.querySelector('#peel-tpl').innerHTML;
  const traveller = lane.clientWidth < 520 ? 48 : 64;
  const startX = lane.clientWidth < 520 ? 44 : 70;
  const doorX = oven.offsetLeft - startX - traveller - 4;
  const queue = [];
  let baking = false, baked = 0, foodIndex = 0, blocked = false;
  lane.style.setProperty('--travel', cfg.travel + 'ms');
  const setOven = (state, text) => {{
    oven.classList.remove('state-open', 'state-half', 'state-closed');
    oven.classList.add('state-' + state);
    oven.classList.toggle('baking', state === 'closed');
    label.textContent = text;
  }};
  setOven('open', 'Waiting for dough');
  const slotX = (slot) => doorX - slot * (traveller - 10);
  const layoutQueue = () => queue.forEach((tray, slot) => {{ tray.style.transform = `translateX(${{slotX(slot)}}px)`; }});
  const spawn = () => {{
    if (queue.length >= cfg.maxQueue) {{ blocked = true; return; }}
    const tray = document.createElement('div');
    tray.className = 'traveller';
    const food = foods.length ? foods[foodIndex++ % foods.length] : '';
    tray.innerHTML = peelHtml + (food ? `<img class="dough" src="${{food}}" alt="">` : '');
    tray.dataset.food = food;
    lane.appendChild(tray);
    queue.push(tray);
    requestAnimationFrame(() => requestAnimationFrame(() => {{ layoutQueue(); }}));
    setTimeout(() => {{ tray.dataset.arrived = '1'; tryBake(); }}, cfg.travel);
  }};
  const tryBake = () => {{
    if (baking || !queue.length || !queue[0].dataset.arrived) return;
    baking = true;
    const tray = queue.shift();
    setOven('half', 'Tray going in');
    tray.style.opacity = '0';
    setTimeout(() => tray.remove(), 220);
    layoutQueue();
    if (blocked) {{ blocked = false; spawn(); }}
    setTimeout(() => {{
      setOven('closed', 'Baking');
      setTimeout(() => {{
        setOven('half', 'Tray coming out');
        const item = document.createElement('img');
        item.className = 'popped';
        item.src = tray.dataset.food || '';
        item.alt = '';
        item.style.setProperty('--dx', ((baked % 3) - 1) * 40 + 'px');
        item.style.setProperty('--dy', -Math.min(Math.floor(baked / 3), 2) * 18 + 'px');
        if (item.src) stack.appendChild(item);
        if (stack.children.length > 9) stack.removeChild(stack.firstChild);
        baked += 1;
        count.textContent = baked;
        setTimeout(() => {{
          baking = false;
          if (!queue.length || !queue[0].dataset.arrived) setOven('open', 'Empty, waiting for dough');
          tryBake();
        }}, cfg.door);
      }}, cfg.bake);
    }}, cfg.door);
  }};
  const prefill = cfg.ovenBound ? cfg.maxQueue : 1;
  for (let i = 0; i < prefill; i++) setTimeout(spawn, i * 260);
  setInterval(spawn, cfg.arrival);
}})();
</script>
</body></html>"""


def height(show_animation=True):
    return STATIC_HEIGHT + (LANE_HEIGHT if show_animation else 40) + 40
