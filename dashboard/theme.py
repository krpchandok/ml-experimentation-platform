import html
from urllib.parse import quote

import streamlit as st

from dashboard import art, wording

TOKENS = {
    "light": {
        "bg": "#FBF5EC", "surface": "#FFFCF7", "surface2": "#F4E9DA", "text": "#2E2219", "text2": "#5C4A3D",
        "muted": "#6F5F53", "border": "#E6D5BF", "accent": "#B93C0C", "on_accent": "#FFFFFF",
        "gradient": "linear-gradient(120deg, #F59E0B 0%, #E8590C 55%, #D9466F 100%)",
        "shadow": "0 1px 2px rgba(46,34,25,.06), 0 8px 24px rgba(46,34,25,.07)",
        "glow": "rgba(245,158,11,.55)", "good": "#0B7A0B", "warn": "#9A6200", "bad": "#B42525", "info": "#2A63B8",
        "good_bg": "#E7F4E4", "warn_bg": "#FCF0D4", "bad_bg": "#FBE3E0", "info_bg": "#E3EDFA", "track": "#EADCC8",
    },
    "dark": {
        "bg": "#17120E", "surface": "#211913", "surface2": "#2B211A", "text": "#F6EDE1", "text2": "#DCCBB8",
        "muted": "#B3A194", "border": "#3A2D24", "accent": "#F4A261", "on_accent": "#1B130D",
        "gradient": "linear-gradient(120deg, #F59E0B 0%, #E8590C 55%, #D9466F 100%)",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35)",
        "glow": "rgba(255,170,60,.65)", "good": "#5FD35F", "warn": "#F5C04A", "bad": "#FF7B73", "info": "#7FB0F5",
        "good_bg": "#1C2E1A", "warn_bg": "#33280F", "bad_bg": "#3A1C1A", "info_bg": "#18263A", "track": "#3A2D24",
    },
}

ICONS = {
    "check": '<path d="M4 12.5l5 5L20 7" />',
    "alert": '<path d="M12 4l9 16H3z" /><path d="M12 10v4" /><path d="M12 17.5v.5" />',
    "stop": '<circle cx="12" cy="12" r="9" /><path d="M8 8l8 8M16 8l-8 8" />',
    "clock": '<circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />',
    "info": '<circle cx="12" cy="12" r="9" /><path d="M12 11v6" /><path d="M12 7.5v.5" />',
    "pause": '<circle cx="12" cy="12" r="9" /><path d="M10 9v6M14 9v6" />',
    "spark": '<path d="M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2z" />',
    "arrow": '<path d="M4 12h15" /><path d="M13 6l6 6-6 6" />',
    "up": '<path d="M12 19V5" /><path d="M6 11l6-6 6 6" />',
    "down": '<path d="M12 5v14" /><path d="M6 13l6 6 6-6" />',
}

TONE_ICON = {"good": "check", "warn": "alert", "bad": "stop", "info": "info", "busy": "clock", "pause": "pause"}
SEVERITY_TONE = {"critical": "bad", "bottleneck": "warn", "warning": "warn", "ok": "good", "info": "info",
                 "unknown": "info"}
STATUS_TONE = {"completed": "good", "profiled": "info", "running": "busy", "failed": "bad", "interrupted": "pause"}


def mode():
    kind = getattr(st.context.theme, "type", None)
    return "dark" if kind == "dark" else "light"


def tokens():
    return TOKENS[mode()]


def icon_mask(name):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2.4" '
           f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    return "url(\"data:image/svg+xml," + quote(svg) + "\")"


def icon_css():
    return "\n".join(f".bh-ic-{name} {{ -webkit-mask: {icon_mask(name)} center / contain no-repeat; "
                     f"mask: {icon_mask(name)} center / contain no-repeat; }}" for name in ICONS)


def icon(name, size=16):
    return (f'<span class="bh-icon bh-ic-{name}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>')


def escape(text):
    return html.escape(str(text), quote=True)


def pill(text, tone="info", icon_name=None):
    glyph = icon(icon_name or TONE_ICON.get(tone, "info"), 14)
    return f'<span class="bh-pill bh-{tone}">{glyph}<span>{escape(text)}</span></span>'


def status_pill(status):
    label, _ = wording.STATUS.get(status, (status or "Unknown", None))
    return pill(label, STATUS_TONE.get(status, "info"))


def severity_pill(severity, text=None):
    label = text or wording.SEVERITY.get(severity, wording.SEVERITY["unknown"])[0]
    return pill(label, SEVERITY_TONE.get(severity, "info"))


def tip(term):
    definition = wording.GLOSSARY.get(term)
    if not definition:
        return ""
    text = escape(f"{term.capitalize()}: {definition}")
    return (f'<span class="bh-tip" tabindex="0" role="button" aria-label="{text}" data-tip="{text}">?</span>')


def css():
    t = tokens()
    return f"""
<style>
:root {{
  --bh-bg: {t['bg']}; --bh-surface: {t['surface']}; --bh-surface2: {t['surface2']}; --bh-text: {t['text']};
  --bh-text2: {t['text2']}; --bh-muted: {t['muted']}; --bh-border: {t['border']}; --bh-accent: {t['accent']};
  --bh-on-accent: {t['on_accent']}; --bh-gradient: {t['gradient']}; --bh-shadow: {t['shadow']};
  --bh-good: {t['good']}; --bh-warn: {t['warn']}; --bh-bad: {t['bad']}; --bh-info: {t['info']};
  --bh-good-bg: {t['good_bg']}; --bh-warn-bg: {t['warn_bg']}; --bh-bad-bg: {t['bad_bg']}; --bh-info-bg: {t['info_bg']};
  --bh-track: {t['track']};
}}
.block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
[data-testid="stHeader"] {{ background: transparent; }}
.bh-header {{ display: flex; align-items: center; gap: 18px; padding: 18px 22px; margin-bottom: 8px;
  background: var(--bh-surface); border: 1px solid var(--bh-border); border-radius: 24px; box-shadow: var(--bh-shadow);
  position: relative; overflow: hidden; }}
.bh-header::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 4px; background: var(--bh-gradient); }}
.bh-header .art {{ width: 64px; height: 64px; }}
.bh-header .bh-brand {{ font-family: Fredoka, Nunito, sans-serif; font-size: 1.7rem; font-weight: 600; color: var(--bh-text); line-height: 1.1; }}
.bh-header .bh-tagline {{ color: var(--bh-text2); font-size: 1rem; margin-top: 2px; }}
.bh-header .bh-mitt {{ margin-left: auto; width: 52px; height: 52px; opacity: .9; }}
.bh-page-title {{ font-family: Fredoka, Nunito, sans-serif; font-weight: 600; font-size: 2rem; color: var(--bh-text); margin: 18px 0 4px; }}
.bh-lead {{ font-size: 1.12rem; color: var(--bh-text2); max-width: 760px; line-height: 1.55; margin-bottom: 6px; }}
.bh-card {{ background: var(--bh-surface); border: 1px solid var(--bh-border); border-radius: 22px; padding: 20px 22px;
  box-shadow: var(--bh-shadow); color: var(--bh-text); height: 100%; box-sizing: border-box; }}
.bh-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 16px; margin: 10px 0 6px; }}
.bh-grid-4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 10px 0 6px; }}
.bh-kicker {{ font-size: .78rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--bh-muted); }}
.bh-title {{ font-family: Fredoka, Nunito, sans-serif; font-weight: 600; font-size: 1.25rem; margin: 2px 0 6px; color: var(--bh-text); }}
.bh-big {{ font-family: Fredoka, Nunito, sans-serif; font-weight: 600; font-size: 2.3rem; line-height: 1.1; color: var(--bh-text); }}
.bh-big-sm {{ font-family: Fredoka, Nunito, sans-serif; font-weight: 600; font-size: 1.6rem; line-height: 1.15; color: var(--bh-text); }}
.bh-sub {{ color: var(--bh-text2); font-size: .95rem; line-height: 1.45; }}
.bh-muted {{ color: var(--bh-muted); font-size: .88rem; line-height: 1.4; }}
.bh-row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.bh-pill {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 3px 11px 3px 9px;
  font-size: .85rem; font-weight: 700; color: var(--bh-text); border: 1px solid var(--bh-border); white-space: nowrap; }}
.bh-icon {{ display: inline-block; flex: none; background-color: currentColor; vertical-align: middle; }}
{icon_css()}
.bh-good {{ background: var(--bh-good-bg); }} .bh-good .bh-icon {{ color: var(--bh-good); }}
.bh-warn {{ background: var(--bh-warn-bg); }} .bh-warn .bh-icon {{ color: var(--bh-warn); }}
.bh-bad {{ background: var(--bh-bad-bg); }} .bh-bad .bh-icon {{ color: var(--bh-bad); }}
.bh-info {{ background: var(--bh-info-bg); }} .bh-info .bh-icon {{ color: var(--bh-info); }}
.bh-busy {{ background: var(--bh-warn-bg); }} .bh-busy .bh-icon {{ color: var(--bh-warn); }}
.bh-pause {{ background: var(--bh-surface2); }} .bh-pause .bh-icon {{ color: var(--bh-muted); }}
.bh-best {{ display: inline-flex; align-items: center; gap: 6px; background: var(--bh-gradient); color: #fff;
  border-radius: 999px; padding: 4px 12px; font-weight: 800; font-size: .85rem; text-shadow: 0 1px 1px rgba(0,0,0,.25); }}
.bh-card.bh-winner {{ border: 2px solid transparent; background: linear-gradient(var(--bh-surface), var(--bh-surface)) padding-box,
  var(--bh-gradient) border-box; }}
.bh-target-art {{ display: flex; align-items: center; justify-content: space-between; }}
.bh-target-art .art {{ width: 72px; height: 72px; }}
.bh-coin {{ width: 22px; height: 22px; vertical-align: -5px; margin-right: 4px; }}
.bh-fix {{ border: 2px solid transparent; background: linear-gradient(var(--bh-surface), var(--bh-surface)) padding-box,
  var(--bh-gradient) border-box; border-radius: 24px; padding: 22px 24px; box-shadow: var(--bh-shadow); margin: 14px 0 8px; }}
.bh-fix-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }}
.bh-fix-col {{ background: var(--bh-surface2); border-radius: 18px; padding: 16px 18px; }}
.bh-arrow {{ display: inline-flex; vertical-align: middle; color: var(--bh-muted); margin: 0 6px; }}
.bh-tip {{ position: relative; display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px;
  margin-left: 6px; border-radius: 50%; font-size: .72rem; font-weight: 800; cursor: help; vertical-align: 2px;
  color: var(--bh-text2); background: var(--bh-surface2); border: 1px solid var(--bh-border); }}
.bh-tip:hover::after, .bh-tip:focus::after {{ content: attr(data-tip); position: absolute; z-index: 30; left: 50%; bottom: 135%;
  transform: translateX(-50%); width: 240px; padding: 10px 12px; border-radius: 12px; background: var(--bh-text);
  color: var(--bh-bg); font-size: .82rem; font-weight: 600; line-height: 1.35; text-transform: none; letter-spacing: 0;
  box-shadow: var(--bh-shadow); white-space: normal; }}
.bh-tip:focus {{ outline: 2px solid var(--bh-accent); outline-offset: 2px; }}
.bh-summary {{ font-family: Fredoka, Nunito, sans-serif; font-weight: 500; font-size: 1.55rem; line-height: 1.35;
  color: var(--bh-text); margin: 6px 0 4px; max-width: 900px; }}
.bh-tech {{ border-left: 3px solid var(--bh-border); padding: 4px 0 4px 12px; color: var(--bh-text2); font-size: .9rem; margin-top: 8px; }}
.bh-bar {{ height: 8px; border-radius: 99px; background: var(--bh-track); overflow: hidden; margin-top: 6px; }}
.bh-bar > span {{ display: block; height: 100%; border-radius: 99px; background: var(--bh-gradient); }}
.bh-ba {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 10px 0; }}
.bh-ba .bh-big {{ font-size: 2rem; }}
.bh-footer {{ margin-top: 42px; padding: 16px 4px; border-top: 1px solid var(--bh-border); color: var(--bh-muted); font-size: .85rem; }}
.bh-footer a {{ color: var(--bh-accent); }}
.art-missing {{ display: inline-flex; align-items: center; justify-content: center; border-radius: 18px;
  background: var(--bh-surface2); border: 1px dashed var(--bh-border); color: var(--bh-muted); font-size: .7rem;
  font-weight: 700; text-transform: uppercase; width: 64px; height: 64px; }}
.bh-empty {{ text-align: center; padding: 32px 20px; }}
.bh-empty .art {{ width: 120px; height: 120px; }}
.bh-runs .bh-card {{ display: flex; flex-direction: column; gap: 8px; }}
.bh-food {{ width: 36px; height: 36px; }}
@media (max-width: 700px) {{ .bh-fix-cols {{ grid-template-columns: 1fr; }} .bh-header .bh-mitt {{ display: none; }} }}
</style>
"""


def inject():
    st.html(css())


def header():
    st.html(f"""
<div class="bh-header" role="banner">
  {art.img('01_croissant', 'croissant')}
  <div><div class="bh-brand">{escape(wording.BRAND)}</div><div class="bh-tagline">{escape(wording.TAGLINE)}</div></div>
  {art.img('oven_mitt', 'oven mitt', 'bh-mitt')}
</div>""")
    if not art.available():
        st.info(art.missing_message(), icon=":material/image:")


def page_title(text, lead=None):
    st.html(f'<div class="bh-page-title">{escape(text)}</div>' + (f'<div class="bh-lead">{escape(lead)}</div>' if lead else ""))


def footer():
    st.html(f'<div class="bh-footer">{art.attribution_html()} Charts and numbers come from your own runs.</div>')


def technical():
    return bool(st.session_state.get("tech", False))
