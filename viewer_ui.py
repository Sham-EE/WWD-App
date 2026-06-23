"""Shared, compact viewer controls (nav + bulk toggles) for the 3D viewers.

Keeps the Background-Filtering / Detection / Visualizer viewers consistent and
small so the point cloud itself gets the space — a single tight row for frame
navigation, and helpers for "select all / clear all" over a group of toggles.
"""
import streamlit as st


def nav_row(state_key, n, key_prefix):
    """One compact row: ⏮ ◀ ▶ ⏭ · Play · delay · frame slider (collapsed labels).

    `state_key` is a plain session var holding the frame index (NOT a widget key),
    so external play-loops can freely set it then rerun. The frame slider is
    keyless and driven by `value=`, with a prefix-unique (hidden) label so two
    viewers on one page don't share a keyless widget id. Returns
    ``(i, playing, delay)``."""
    st.session_state.setdefault(state_key, 0)
    st.session_state[state_key] = max(0, min(st.session_state[state_key], max(n - 1, 0)))
    c = st.columns([0.6, 0.6, 0.6, 0.6, 0.9, 1.3, 6])
    if c[0].button("⏮", key=f"{key_prefix}_first", help="First frame", use_container_width=True):
        st.session_state[state_key] = 0
    if c[1].button("◀", key=f"{key_prefix}_prev", help="Previous", use_container_width=True):
        st.session_state[state_key] = max(0, st.session_state[state_key] - 1)
    if c[2].button("▶", key=f"{key_prefix}_next", help="Next", use_container_width=True):
        st.session_state[state_key] = min(n - 1, st.session_state[state_key] + 1)
    if c[3].button("⏭", key=f"{key_prefix}_last", help="Last frame", use_container_width=True):
        st.session_state[state_key] = n - 1
    playing = c[4].toggle("▶", key=f"{key_prefix}_play", help="Auto-play")
    delay = c[5].slider("delay", 0.0, 1.0, 0.15, 0.05, key=f"{key_prefix}_delay",
                        label_visibility="collapsed", help="Play delay (s)")
    i = c[6].slider(f"frame_{key_prefix}", min_value=0, max_value=max(n - 1, 1),
                    value=st.session_state[state_key], label_visibility="collapsed")
    st.session_state[state_key] = i
    return i, playing, delay


def bulk_toggle_buttons(keys, key_prefix, rerun_scope="fragment"):
    """Render '✅ All' / '⬜ None' buttons that set every key in `keys` to
    True / False. Pair with toggles that read their value from session_state
    (use `ensure_toggle_defaults` so they don't fight a `value=` default)."""
    a, b = st.columns(2)
    if a.button("✅ All", key=f"{key_prefix}_all", use_container_width=True,
                help="Turn on every overlay"):
        for k in keys:
            st.session_state[k] = True
        st.rerun(scope=rerun_scope)
    if b.button("⬜ None", key=f"{key_prefix}_none", use_container_width=True,
                help="Turn off every overlay"):
        for k in keys:
            st.session_state[k] = False
        st.rerun(scope=rerun_scope)


def ensure_toggle_defaults(defaults):
    """Seed toggle keys in session_state so the widgets can be created with a
    `key` and NO `value=` — that way bulk_toggle_buttons can flip them without
    Streamlit's 'set via Session State and default' warning. `defaults` is
    {key: bool}."""
    for k, v in defaults.items():
        st.session_state.setdefault(k, bool(v))
