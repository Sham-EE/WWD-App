"""Standalone V2X wrong-way-driver dashboard — generated entirely in-app.

Replaces the external single-file React asset (assets/wwd_v2x_dashboard.html): we
own the rendering here, so the map is drawn from OUR exact georeferenced geometry
(the wrong-way lane + the driver at its true lat/lon, no offset) and every panel is
built from the detection event the WWD Simulator broadcasts.

`build_dashboard_html(event)` returns a self-contained HTML document (Leaflet +
Esri World Imagery satellite, no API key) embedded by pages/8_V2X_Dashboard.py.
"""
import hashlib
import html as _html
import json
import time
import uuid


# ITIS (J2540) phrase code for "wrong way driver".
ITIS_WRONG_WAY = 8708


def build_tim(event):
    """A J2735-style TIM (Traveler Information Message) payload for the alert — the
    JSON a roadside unit would UPER-encode and broadcast over C-V2X."""
    ev_uuid = event.get("event_uuid") or str(uuid.uuid4())
    inter = event.get("intersection") or {}
    region = inter.get("applicableRegion") or {}
    lat = float(event.get("lat", 0.0))
    lon = float(event.get("lon", 0.0))
    msg = {
        "messageType": "TravelerInformation",
        "msgCnt": 1,
        "timeStamp": int(time.time()),
        "packetID": hashlib.sha1(ev_uuid.encode()).hexdigest()[:18].upper(),
        "eventUUID": ev_uuid,
        "dataFrames": [{
            "frameType": "advisory",
            "itisCodes": [ITIS_WRONG_WAY],          # 8708 = wrong-way driver
            "phrase": "Wrong-way driver ahead",
            "priority": 7,
            "anchor": {"lat": round(lat, 7), "lon": round(lon, 7),
                       "elevation_m": inter.get("elevation", 0)},
            "heading_deg_true": event.get("heading"),
            "speed_mps": event.get("speed"),
            "road": inter.get("roadName") or event.get("site"),
            "applicableRegion": region,
        }],
    }
    return msg, ev_uuid


def _hex_blob(msg):
    """A deterministic pseudo-UPER hex rendering of the message (for the 'Copy Hex'
    feel — not a real ASN.1 encode, but stable per event)."""
    raw = hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest().upper()
    blob = (raw * 4)[:256]
    return " ".join(blob[i:i + 2] for i in range(0, len(blob), 2))


# Pipeline stages mirrored from the real detector path (LiDAR → WWD → broadcast).
_PIPELINE = [
    ("Capturing point-cloud frames", "Ouster OS LiDAR · south + north"),
    ("Merging north + south LiDAR", "registered to the s110 base frame"),
    ("Background filtering", "removing static scene (DBSCAN density model)"),
    ("Object detection", "DBSCAN clustering → oriented boxes"),
    ("Trajectory tracking", "Kalman filter, per-object IDs"),
    ("Wrong-way confirmation", "heading vs. lane flow over N frames"),
    ("V2X broadcast", "J2735 TIM 8708 via C-V2X RSU"),
]

_RECEIVERS = [
    ("🚗", "C-V2X vehicles in range", "In-vehicle alert: WRONG WAY AHEAD"),
    ("🚓", "Law enforcement", "Geofenced heads-up + CAD dispatch (GPS + timestamp)"),
    ("🛰️", "RSU / TMC uplink", "Event logged, ITIS 8708 relayed upstream"),
]


def _esc(v):
    return _html.escape(str(v))


# Self-playing Leaflet animation: the driver moves along `path` in real time and each
# synthetic C-V2X vehicle lights up (turns red + WRONG-WAY popup) the moment the driver
# enters its range — so the tab plays live on its own, no Streamlit reruns. Single
# braces here (injected into the f-string as an opaque value, not f-string-parsed).
_ANIM_JS = r"""
var D = __GEO__;
var map = L.map('map',{zoomControl:true}).setView(D.center, 18);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:21,maxNativeZoom:19,attribution:'Imagery © Esri'}).addTo(map);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  {maxZoom:21,maxNativeZoom:19,opacity:.9}).addTo(map);
if (D.region && D.region.nwLat){
  L.rectangle([[D.region.nwLat,D.region.nwLon],[D.region.seLat,D.region.seLon]],
    {color:'#3b82f6',weight:1,dashArray:'5,5',fill:false}).addTo(map).bindTooltip('Geofence');
}
if (D.laneNodes && D.laneNodes.length){
  L.polyline(D.laneNodes, {color:'#f59e0b',weight:5,opacity:.9}).addTo(map);
}
(D.sensors||[]).forEach(function(s){
  L.circleMarker([s[0],s[1]],{radius:5,color:'#22d3ee',weight:2,fillOpacity:.6}).addTo(map).bindTooltip('LiDAR');
});
// synthetic C-V2X receivers
var vehMarks = (D.vehicles||[]).map(function(v){
  var col = v.kind==='police' ? '#a78bfa' : '#3b82f6';
  var m = L.circleMarker([v.lat,v.lon],{radius:6,color:col,weight:2,fillColor:col,fillOpacity:.85})
            .addTo(map).bindTooltip(v.label);
  return {v:v, m:m, col:col, alerted:false};
});
var counter = document.getElementById('alertCount'), cnt = 0;
var R = D.alertRadius||90, fps = D.fps||10, SUBS = 4;
var path = (D.path && D.path.length>1) ? D.path : [D.driver, D.driver];
var trail = L.polyline([], {color:'#ff3b3b',weight:2,opacity:.55,dashArray:'4,5'}).addTo(map);
var arrow = L.polyline([path[0],path[0]], {color:'#ff3b3b',weight:4}).addTo(map);
var driver = L.circleMarker(path[0], {radius:8,color:'#fff',weight:2,fillColor:'#ff3b3b',fillOpacity:.95})
               .addTo(map).bindTooltip('Wrong-way driver');
map.fitBounds(L.polyline(path).getBounds().pad(0.35));
function hav(a,b){ var Re=6371000, r=Math.PI/180;
  var dLat=(b[0]-a[0])*r, dLon=(b[1]-a[1])*r;
  var s=Math.sin(dLat/2)*Math.sin(dLat/2)+Math.cos(a[0]*r)*Math.cos(b[0]*r)*Math.sin(dLon/2)*Math.sin(dLon/2);
  return 2*Re*Math.asin(Math.min(1,Math.sqrt(s))); }
var seg=0, sub=0;
function reset(){ seg=0; sub=0; cnt=0; trail.setLatLngs([]); if(counter) counter.textContent=0;
  vehMarks.forEach(function(vl){ vl.alerted=false; vl.m.setStyle({color:vl.col,fillColor:vl.col}); vl.m.setRadius(6); vl.m.closePopup(); }); }
function tick(){
  var a=path[seg], b=path[Math.min(seg+1,path.length-1)], t=sub/SUBS;
  var pos=[a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t];
  driver.setLatLng(pos); trail.addLatLng(pos);
  var dir=[b[0]-a[0], b[1]-a[1]], nrm=Math.hypot(dir[0],dir[1])||1, L0=0.00035;
  arrow.setLatLngs([pos, [pos[0]+dir[0]/nrm*L0, pos[1]+dir[1]/nrm*L0]]);
  vehMarks.forEach(function(vl){
    if(!vl.alerted && hav(pos,[vl.v.lat,vl.v.lon]) < R){
      vl.alerted=true; cnt++;
      vl.m.setStyle({color:'#ff3b3b',fillColor:'#ff3b3b'}); vl.m.setRadius(8);
      vl.m.bindPopup('⚠️ C-V2X: WRONG WAY AHEAD').openPopup();
      if(counter) counter.textContent=cnt;
    }
  });
  sub++;
  if(sub>=SUBS){ sub=0; seg++;
    if(seg>=path.length-1){ setTimeout(function(){ reset(); tick(); }, 900); return; } }
  setTimeout(tick, 1000/(fps*SUBS));
}
tick();
function cp(id){ var t=document.getElementById(id).innerText; navigator.clipboard&&navigator.clipboard.writeText(t); }
"""


def build_dashboard_html(event, *, height=980):
    """Full standalone dashboard HTML for `event` (the WWD broadcast payload)."""
    tim, ev_uuid = build_tim(event)
    tim_json = json.dumps(tim, indent=2)
    hexblob = _hex_blob(tim)

    inter = event.get("intersection") or {}
    site = event.get("site", "Intersection")
    lane = event.get("lane", "—")
    direction = event.get("direction", "—")
    legal = event.get("legal_name", "—")
    speed = event.get("speed", 0.0)
    heading = event.get("heading", 0)
    lat = float(event.get("lat", inter.get("center", [0, 0])[0]))
    lon = float(event.get("lon", inter.get("center", [0, 0])[1]))
    exact = bool(event.get("lat_exact"))
    conf = event.get("confirm") or {}

    vehicles = event.get("vehicles", [])
    n_veh = len(vehicles)
    # Geometry injected into the Leaflet script (all true lat/lon). The dashboard
    # ANIMATES the driver along `path` in real time and lights up `vehicles` as the
    # wrong-way driver enters their C-V2X range — so the tab plays on its own.
    geo_payload = json.dumps({
        "center": inter.get("center", [lat, lon]),
        "laneNodes": inter.get("laneNodes", []),
        "driver": [lat, lon],
        "path": event.get("path") or [[lat, lon]],
        "fps": float(event.get("fps", 10)),
        "alertRadius": float(event.get("alert_radius_m", 90)),
        "vehicles": vehicles,
        "region": inter.get("applicableRegion") or {},
        "sensors": event.get("sensors", []),
    })
    anim = _ANIM_JS.replace("__GEO__", geo_payload)

    pipeline_rows = "\n".join(
        f'<div class="step"><span class="dot"></span><div><b>{_esc(t)}</b>'
        f'<small>{_esc(s)}</small></div><span class="ok">✓</span></div>'
        for t, s in _PIPELINE)
    receiver_rows = "\n".join(
        f'<div class="rx"><span class="rxic">{ic}</span><div><b>{_esc(name)}</b>'
        f'<small>{_esc(act)}</small></div><span class="ok">● received</span></div>'
        for ic, name, act in _RECEIVERS)

    loc_txt = f"{lat:.6f}, {lon:.6f}" + ("" if exact else " (approx)")
    conf_txt = ""
    if conf:
        conf_txt = (f"<div class='kv'><span>Confirmed after</span><b>{_esc(conf.get('frames','—'))} frames "
                    f"(~{_esc(conf.get('seconds','—'))}s)</b></div>"
                    f"<div class='kv'><span>Max angle vs flow</span><b>{_esc(conf.get('max_angle','—'))}°</b></div>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
:root{{--bg:#0b0f17;--card:#121823;--line:#1f2a3a;--ink:#dbe3f0;--mut:#8a97ab;--red:#ff3b3b;--amb:#f59e0b;--grn:#34d399;--blue:#3b82f6}}
*{{box-sizing:border-box}} html,body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,Segoe UI,sans-serif}}
.hdr{{display:flex;align-items:center;gap:12px;padding:12px 16px;background:linear-gradient(90deg,#1a1020,#10131c);border-bottom:1px solid var(--line)}}
.hdr h1{{font-size:17px;margin:0}} .hdr .sub{{color:var(--mut);font-size:12px}}
.badge{{margin-left:auto;background:var(--red);color:#fff;font-weight:700;padding:6px 12px;border-radius:8px;font-size:13px;animation:pulse 1.2s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.55}}}}
#map{{height:46vh;min-height:320px;width:100%;border-bottom:1px solid var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px;padding:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}
.card h2{{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}}
.kv{{display:flex;justify-content:space-between;gap:10px;padding:5px 0;border-bottom:1px dashed var(--line);font-size:14px}}
.kv:last-child{{border-bottom:0}} .kv span{{color:var(--mut)}} .kv b{{text-align:right}}
.big{{font-size:22px;font-weight:800;color:var(--red)}}
.step,.rx{{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--line)}}
.step:last-child,.rx:last-child{{border-bottom:0}}
.step b,.rx b{{display:block;font-size:13px}} .step small,.rx small{{color:var(--mut);font-size:11px}}
.step .dot{{width:9px;height:9px;border-radius:50%;background:var(--grn);box-shadow:0 0 8px var(--grn)}}
.step .ok,.rx .ok{{margin-left:auto;color:var(--grn);font-size:12px;font-weight:700}}
.rxic{{font-size:18px}}
pre{{background:#0a0e16;border:1px solid var(--line);border-radius:8px;padding:10px;max-height:230px;overflow:auto;font-size:11.5px;color:#bfe3c8;margin:0}}
.row{{display:flex;gap:8px;margin-top:8px}}
button{{background:#1f2a3a;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:6px 12px;cursor:pointer;font-size:12px}}
button:hover{{background:#27344a}}
.tag{{display:inline-block;background:#231a10;color:var(--amb);border:1px solid #4a3410;border-radius:6px;padding:2px 8px;font-size:11px;margin-bottom:8px}}
</style></head><body>
<div class="hdr"><div>🚨</div><div><h1>V2X Broadcast — Wrong-Way Driver</h1>
<div class="sub">{_esc(site)}</div></div><div class="badge">● ALERT ACTIVE</div></div>
<div id="map"></div>
<div class="grid">
  <div class="card"><h2>Detection</h2>
    <div class="big">{_esc(direction)}-bound · wrong way</div>
    <div class="kv"><span>Lane</span><b>{_esc(lane)}</b></div>
    <div class="kv"><span>Legal direction</span><b>{_esc(legal)}</b></div>
    <div class="kv"><span>Speed</span><b>{_esc(speed)} m/s</b></div>
    <div class="kv"><span>Heading (true)</span><b>{_esc(heading)}°</b></div>
    <div class="kv"><span>Position</span><b>{_esc(loc_txt)}</b></div>
    {conf_txt}
  </div>
  <div class="card"><h2>J2735 TIM message</h2>
    <span class="tag">ITIS {ITIS_WRONG_WAY} · wrong-way driver</span>
    <div class="kv"><span>Event UUID</span><b>{_esc(ev_uuid)}</b></div>
    <div class="kv"><span>Packet ID</span><b>{_esc(tim['packetID'])}</b></div>
    <pre id="tim">{_esc(tim_json)}</pre>
    <div class="row"><button onclick="cp('tim')">Copy JSON</button>
      <button onclick="cp('hex')">Copy Hex</button></div>
    <pre id="hex" style="display:none">{_esc(hexblob)}</pre>
  </div>
  <div class="card"><h2>Detection → broadcast pipeline</h2>{pipeline_rows}</div>
  <div class="card"><h2>Receivers notified</h2>
    <div class="kv"><span>C-V2X vehicles alerted (live)</span><b><span id="alertCount">0</span> / {n_veh}</b></div>
    {receiver_rows}</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>{anim}</script></body></html>"""
