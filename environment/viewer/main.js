// SkyBlood viewer: replays a dispatcher episode on an OpenStreetMap basemap.
// World units are km: x = east, z = -north, y = altitude (exaggerated).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/CSS2DRenderer.js';

const params = new URLSearchParams(location.search);
const ALT = 3.0;           // cruise altitude in the viewer (km, exaggerated)
const PAD_H = 0.25;
const $ = (id) => document.getElementById(id);
const W = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

const S = {
  meta: null, steps: [], books: [], events: [],
  simT: 0, maxT: 0, playing: true, ended: false,
  speed: Number(params.get('speed')) || 4, camMode: 'auto', lastCur: -1,
};

// ---------- scene ----------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xbfd4e6);
scene.fog = new THREE.Fog(0xbfd4e6, 150, 400);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(-5, 95, 70);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
$('scene').appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: 'fixed', top: '0', pointerEvents: 'none' });
$('scene').appendChild(labels.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-5, 0, 5);
controls.maxPolarAngle = Math.PI * 0.47;
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-40, 80, 30);
scene.add(sun);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  labels.setSize(innerWidth, innerHeight);
});

// CSS2DRenderer owns the outer element's transform, so styling and offsets go on an inner element.
function label(html, cls, pos, parent = scene) {
  const outer = document.createElement('div');
  const inner = document.createElement('div');
  inner.className = cls;
  inner.innerHTML = html;
  outer.appendChild(inner);
  const obj = new CSS2DObject(outer);
  obj.position.copy(pos);
  obj.inner = inner;
  parent.add(obj);
  return obj;
}

function mesh(geom, color, opts = {}) {
  return new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color, flatShading: true, ...opts }));
}

async function addBasemap() {
  const b = await (await fetch('basemap.json')).json();
  const tex = await new THREE.TextureLoader().loadAsync('basemap.jpg');
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const w = b.x_max_km - b.x_min_km, h = b.y_max_km - b.y_min_km;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex }));
  plane.rotation.x = -Math.PI / 2;
  plane.position.copy(W((b.x_min_km + b.x_max_km) / 2, (b.y_min_km + b.y_max_km) / 2, 0));
  scene.add(plane);
}

// hub: pad, catapult, recovery wire
const wireMat = new THREE.LineBasicMaterial({ color: 0x222222 });
function addHub() {
  const pad = mesh(new THREE.BoxGeometry(2.4, PAD_H, 2.4), 0x30363d);
  pad.position.y = PAD_H / 2;
  scene.add(pad);
  const rail = mesh(new THREE.BoxGeometry(0.25, 0.12, 1.8), 0x8b949e);
  rail.position.set(-0.6, PAD_H + 0.35, 0);
  rail.rotation.x = 0.35;
  scene.add(rail);
  for (const x of [0.4, 1.1]) {
    const pole = mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.6), 0xd0d7de);
    pole.position.set(x, PAD_H + 0.8, 0.9);
    scene.add(pole);
  }
  const wire = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0.4, PAD_H + 1.55, 0.9), new THREE.Vector3(1.1, PAD_H + 1.55, 0.9)]), wireMat);
  scene.add(wire);
  label('Zipline Muhanga', 'hub-label', W(0, 0, 1.5));
}

// hospitals with labels and order badges
const badges = [];
function addFacilities() {
  for (const f of S.meta.facilities) {
    const g = new THREE.Group();
    g.position.copy(W(f.x, f.y));
    const base = mesh(new THREE.CylinderGeometry(0.7, 0.7, 0.3, 16), 0xffffff);
    base.position.y = 0.15;
    const crossA = mesh(new THREE.BoxGeometry(0.9, 0.12, 0.28), 0xe5383b);
    const crossB = mesh(new THREE.BoxGeometry(0.28, 0.12, 0.9), 0xe5383b);
    crossA.position.y = crossB.position.y = 0.36;
    g.add(base, crossA, crossB);
    scene.add(g);
    label(`${f.short}<br><small>${f.dist.toFixed(1)} km</small>`, 'fac-label', W(f.x, f.y, 0.4));
    badges.push(label('', 'badge hidden', W(f.x, f.y, 0.4)).inner);
  }
}

// low-poly fixed-wing drone
const drone = new THREE.Group();
{
  const body = mesh(new THREE.BoxGeometry(0.32, 0.26, 1.5), 0xf2f2f2);
  const nose = mesh(new THREE.ConeGeometry(0.17, 0.45, 6), 0xff7a1a);
  nose.rotation.x = Math.PI / 2;
  nose.position.z = 0.97;
  const wing = mesh(new THREE.BoxGeometry(2.6, 0.05, 0.42), 0xff7a1a);
  wing.position.z = 0.15;
  const tail = mesh(new THREE.BoxGeometry(0.9, 0.04, 0.25), 0xf2f2f2);
  tail.position.z = -0.65;
  const fin = mesh(new THREE.BoxGeometry(0.04, 0.4, 0.3), 0xff7a1a);
  fin.position.set(0, 0.2, -0.65);
  drone.add(body, nose, wing, tail, fin);
  drone.scale.setScalar(2.2);
  scene.add(drone);
}

// storm cell
const storm = new THREE.Group();
{
  const col = mesh(new THREE.CylinderGeometry(1, 1, 7, 48, 1, true), 0x37474f,
    { transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false });
  col.position.y = 3.5;
  const disk = mesh(new THREE.CircleGeometry(1, 48), 0x263238, { transparent: true, opacity: 0.25, depthWrite: false });
  disk.rotation.x = -Math.PI / 2;
  disk.position.y = 0.05;
  storm.add(col, disk);
  scene.add(storm);
}

// planned route
const routeMat = new THREE.LineDashedMaterial({ color: 0x1f6feb, dashSize: 1.2, gapSize: 0.6 });
const route = new THREE.Line(new THREE.BufferGeometry(), routeMat);
scene.add(route);

// parachute box (one, reused) and floating texts
const chute = new THREE.Group();
{
  const box = mesh(new THREE.BoxGeometry(0.5, 0.4, 0.5), 0xc9302c);
  const canopy = mesh(new THREE.SphereGeometry(0.9, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2), 0xffffff,
    { side: THREE.DoubleSide });
  canopy.position.y = 1.3;
  chute.add(box, canopy);
  chute.visible = false;
  scene.add(chute);
}

// ---------- data ----------

function addStep(step, ticks) {
  const prev = S.steps[S.steps.length - 1];
  step.nDeliv = (prev?.nDeliv || 0) + (step.delivery ? 1 : 0);
  step.nEmerg = (prev?.nEmerg || 0) + (step.delivery?.emergency ? 1 : 0);
  step.nExpired = (prev?.nExpired || 0) + step.n_expired;
  step.flightTicks = step.forced ? (step.action >= 3 ? step.ticks : 2) : Math.max(step.t1 - step.t0, 1e-6);
  step.ff = step.forced && step.energy > 0 ? Math.min(Math.max(step.b0 / step.energy, 0.1), 0.95) : 1;
  S.steps.push(step);
  S.books.push(...ticks);
  const fac = S.meta.facilities;
  if (step.delivery) {
    const d = step.delivery;
    S.events.push({ t: step.t1, pos: W(fac[d.k].x, fac[d.k].y), text: `+${d.gain.toFixed(1)}${d.emergency ? ' EMERG' : ''}`, good: true });
  } else if (step.breakdown.wasted < 0) {
    S.events.push({ t: step.t1, pos: W(step.target[0], step.target[1]), text: 'wasted −2', good: false });
  }
  if (step.forced) {
    const p = forcedPoint(step);
    S.events.push({ t: step.t0 + step.ff * step.flightTicks, pos: W(p.x, p.y), text: 'FORCED LANDING', good: false });
  }
  S.maxT = step.t1;
  $('scrub').max = Math.max(S.maxT, 1);
}

function forcedPoint(s) {
  const f = s.action >= 3 ? s.ff : 0;
  return { x: s.pos0[0] + (s.target[0] - s.pos0[0]) * f, y: s.pos0[1] + (s.target[1] - s.pos0[1]) * f };
}

function stepIndex(t) {
  let lo = 0, hi = S.steps.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (S.steps[mid].t0 <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

function bookIndex(t) {
  let lo = 0, hi = S.books.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (S.books[mid].t <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

// ---------- per-frame state ----------

const atHub = (p) => p[0] === 0 && p[1] === 0;
const lerp = (a, b, f) => a + (b - a) * f;
const name = (k) => S.meta.facilities[k].short;

function droneAt(s, t) {
  const f = Math.min(Math.max((t - s.t0) / s.flightTicks, 0), 1);
  const h0 = atHub(s.pos0) ? PAD_H + 0.3 : ALT;
  const flying = (s.action >= 3 && s.dist > 0) || (s.action === 9 && !atHub(s.pos0) && !s.breakdown.invalid);
  if (s.forced) {
    const p = forcedPoint(s);
    const end = s.t0 + s.ff * s.flightTicks;
    if (t < end) {
      const g = s.action >= 3 ? (t - s.t0) / (end - s.t0) : 0;
      const x = lerp(s.pos0[0], p.x, g), y = lerp(s.pos0[1], p.y, g);
      return { x, y, h: lerp(h0, ALT, Math.min(g * 3, 1)), state: 'LOW BATTERY', yaw: yawOf(s), f };
    }
    const fall = Math.min((t - end) / 1.5, 1);
    if (fall < 1) return { x: p.x, y: p.y, h: ALT * (1 - fall), state: 'FORCED LANDING', yaw: yawOf(s), f };
    return { x: 0, y: 0, h: PAD_H + 0.3, state: 'RECOVERY (forced landing)', yaw: 0, f, hidden: t < s.t1 - 0.5 };
  }
  if (flying) {
    const h1 = s.action === 9 ? PAD_H + 0.3 : ALT;
    const e = f * f * (3 - 2 * f);
    const x = lerp(s.pos0[0], s.target[0], e), y = lerp(s.pos0[1], s.target[1], e);
    const state = s.action === 9 ? 'RETURNING TO HUB' : `EN ROUTE → ${name(s.action - 3)}`;
    return { x, y, h: lerp(h0, h1, e) + 1.2 * Math.sin(Math.PI * e), state, yaw: yawOf(s), f };
  }
  const [x, y] = s.pos0;
  if (!atHub(s.pos0)) {
    const a = (t - s.t0) * 1.4;
    const state = s.breakdown.invalid ? 'INVALID ACTION' : s.action === 0 ? 'HOLDING PATTERN' : `OVER ${nearest(s.pos0)}`;
    return { x: x + 1.2 * Math.cos(a), y: y + 1.2 * Math.sin(a), h: ALT, state, yaw: a + Math.PI, f };
  }
  const state = s.breakdown.invalid ? 'INVALID ACTION'
    : s.action === 1 ? 'ON PAD: LOADING UNIT' : s.action === 2 ? 'ON PAD: SWAPPING BATTERY' : 'ON PAD: HOLDING';
  return { x, y, h: PAD_H + 0.3, state, yaw: 0, f };
}

function yawOf(s) {
  const dx = s.target[0] - s.pos0[0], dy = s.target[1] - s.pos0[1];
  return Math.atan2(dx, -dy);
}

function nearest(p) {
  const f = S.meta.facilities.find((f) => Math.abs(f.x - p[0]) < 1e-6 && Math.abs(f.y - p[1]) < 1e-6);
  return f ? f.short : 'route';
}

function batteryAt(s, f, t) {
  if (s.forced) return t < s.t0 + s.ff * s.flightTicks ? lerp(s.b0, 0, Math.min(f / s.ff, 1)) : t >= s.t1 - 0.5 ? s.b1 : 0;
  if (s.action === 2) return f > 0.5 ? s.b1 : s.b0;
  return lerp(s.b0, s.b1, f);
}

const clockText = (t) => {
  const m = 360 + Math.round(t * 3);
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
};

// ---------- update ----------

function update() {
  if (!S.steps.length) return;
  const t = S.simT, cur = stepIndex(t), s = S.steps[cur];
  const d = droneAt(s, t);

  drone.visible = !d.hidden;
  drone.position.copy(W(d.x, d.y, d.h));
  drone.rotation.set(0, d.yaw, 0);

  // storm, interpolated between tick snapshots
  const bi = bookIndex(t), b0 = S.books[bi], b1 = S.books[Math.min(bi + 1, S.books.length - 1)];
  const g = b1.t > b0.t ? Math.min(Math.max((t - b0.t) / (b1.t - b0.t), 0), 1) : 0;
  storm.position.copy(W(lerp(b0.storm[0], b1.storm[0], g), lerp(b0.storm[1], b1.storm[1], g)));
  storm.scale.set(S.meta.storm_r, 1, S.meta.storm_r);

  // order badges from the order book at this time
  b0.urg.forEach((u, k) => {
    const el = badges[k];
    el.className = u === 0 ? 'badge hidden' : u === 2 ? 'badge emerg' : 'badge routine';
    el.textContent = u === 2 ? `EMERG ${b0.left[k]}t` : `routine ${b0.left[k]}t`;
  });

  // planned route
  const flying = d.state.startsWith('EN ROUTE') || d.state.startsWith('RETURNING') || d.state === 'LOW BATTERY';
  route.visible = flying && t < s.t0 + s.flightTicks;
  if (route.visible) {
    const pts = [];
    for (let i = 0; i <= 24; i++) {
      const e = i / 24;
      pts.push(W(lerp(s.pos0[0], s.target[0], e), lerp(s.pos0[1], s.target[1], e), 0.3));
    }
    route.geometry.setFromPoints(pts);
    route.computeLineDistances();
    routeMat.color.set(s.storm ? 0xe5383b : 0x1f6feb);
  }

  // wire catch at the end of a return
  wireMat.color.set(s.action === 9 && !s.forced && d.f > 0.85 ? 0xff7a1a : 0x222222);

  // parachute drops and floating texts
  chute.visible = false;
  for (const ev of S.events) {
    const age = t - ev.t;
    if (!ev.obj) ev.obj = label(ev.text, `float ${ev.good ? 'good' : 'bad'}`, ev.pos);
    ev.obj.visible = age >= 0 && age < 2.5;
    if (ev.obj.visible) ev.obj.position.copy(ev.pos).setY(ALT + 1 + age * 1.2);
    if (ev.good && age >= 0 && age < 1.5) {
      chute.visible = true;
      chute.position.copy(ev.pos).setY(ALT * (1 - age / 1.5) + 0.2);
    }
  }

  // HUD
  const bat = batteryAt(s, d.f, t);
  $('clock').textContent = clockText(t);
  $('tick').textContent = `t = ${t.toFixed(1)} / ${S.meta.T}`;
  $('bat-text').textContent = `${bat.toFixed(0)} / ${S.meta.max_range} km`;
  $('bat-bar').style.width = `${Math.max(0, 100 * bat / S.meta.max_range)}%`;
  $('bat-bar').style.background = bat < 40 ? 'var(--emerg)' : bat < 80 ? 'var(--accent)' : 'var(--ok)';
  const packs = s.action === 1 ? (d.f > 0.5 ? s.packs1 : s.packs0) : t >= s.t1 - 1e-6 || s.forced ? s.packs1 : s.packs0;
  document.querySelectorAll('.slot').forEach((el, i) => el.classList.toggle('full', i < packs));
  $('drone-state').textContent = d.state;

  // stats: completed steps plus the current one once it has finished
  const done = t >= s.t1 - 1e-6 ? s : S.steps[cur - 1];
  $('s-return').textContent = (done?.cum ?? 0).toFixed(1);
  $('s-deliv').textContent = done?.nDeliv ?? 0;
  $('s-emerg').textContent = done?.nEmerg ?? 0;
  $('s-expired').textContent = done?.nExpired ?? 0;
  $('s-forced').textContent = done?.forced_total ?? 0;

  if (cur !== S.lastCur) {
    renderDecision(s);
    renderLog(cur);
    S.lastCur = cur;
  }
  $('scrub').value = t;
  $('scrub-label').textContent = clockText(t);
  updateCamera(d);
}

function renderDecision(s) {
  const kind = S.meta.kind;
  let html = `<div class="note">${clockText(s.t0)} · decision ${s.i + 1}: <b>${s.name}</b></div>`;
  if (!kind) {
    html += '<div class="note">Rule-based heuristic (no network scores).</div>';
  } else {
    const vals = s.top.map((r) => r[1]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    for (const [n, v, valid] of s.top) {
      const w = kind === 'prob' ? v : hi > lo ? (v - lo) / (hi - lo) : 1;
      const cls = `prob${n === s.name ? ' chosen' : ''}${valid ? '' : ' invalid'}`;
      html += `<div class="${cls}"><span class="name">${n}</span><div class="pbar"><div style="width:${(100 * w).toFixed(0)}%">` +
        `</div></div><span class="val">${kind === 'prob' ? (100 * v).toFixed(1) + '%' : v.toFixed(2)}</span></div>`;
    }
    html += `<div class="note">${kind === 'prob' ? 'Policy probabilities' : 'Q-values'} (top 4); greedy with invalid-action masking.` +
      (s.mask_changed ? ' <b>Masking changed this decision.</b>' : '') + '</div>';
  }
  $('decision-body').innerHTML = html;
}

function renderLog(cur) {
  const rows = [];
  for (let i = Math.max(0, cur - 15); i <= cur; i++) {
    const s = S.steps[i];
    const cls = s.reward >= 0 ? 'pos' : 'neg';
    rows.push(`<div class="${i === cur ? 'current' : ''}"><span>${clockText(s.t0)} ${s.name}</span>` +
      `<span class="${cls}">${s.reward >= 0 ? '+' : ''}${s.reward.toFixed(2)}</span></div>`);
  }
  $('log-body').innerHTML = rows.reverse().join('');
}

// ---------- camera ----------

const tmp = new THREE.Vector3();
function updateCamera(d) {
  if (S.camMode === 'overview') return;
  const goal = W(d.x, d.y, 0);
  const k = S.camMode === 'follow' ? 0.25 : 0.04;
  tmp.copy(goal).sub(controls.target).multiplyScalar(k);
  controls.target.add(tmp);
  camera.position.add(tmp);
}

function setCamera(mode) {
  S.camMode = mode;
  if (mode === 'overview') {
    controls.target.set(-5, 0, 5);
    camera.position.set(-5, 95, 70);
  } else if (mode === 'follow') {
    camera.position.copy(drone.position).add(new THREE.Vector3(6, 9, 12));
    controls.target.copy(drone.position).setY(0);
  }
}

// ---------- controls ----------

$('play').onclick = () => {
  if (S.ended && S.simT >= S.maxT) S.simT = 0;
  S.playing = !S.playing;
  $('play').textContent = S.playing ? 'Pause' : 'Play';
};
$('speed').value = String(S.speed);
if ($('speed').value !== String(S.speed)) {
  $('speed').add(new Option(String(S.speed), String(S.speed), true, true));
}
$('speed').onchange = (e) => { S.speed = Number(e.target.value); };
$('camera').onchange = (e) => setCamera(e.target.value);
$('camera').value = params.get('cam') || 'auto';
$('scrub').oninput = (e) => { S.simT = Math.min(Number(e.target.value), S.maxT); };

function banner(text) {
  $('banner').textContent = text;
  $('banner').style.display = text ? 'block' : 'none';
}

// ---------- data sources ----------

function start(meta) {
  S.meta = meta;
  S.books = [meta.start];
  const demand = meta.p_order && meta.p_order !== 0.01 ? ` · demand ×${(meta.p_order / 0.01).toFixed(1)}` : '';
  $('policy').textContent = `· ${meta.policy} · seed ${meta.seed}${demand}`;
  addFacilities();
}

async function loadReplay(url) {
  const trace = await (await fetch(url)).json();
  start(trace.meta);
  for (const { step, ticks } of trace.steps) addStep(step, ticks);
  S.ended = true;
  const t = params.get('t');
  if (t !== null) {
    S.simT = Math.min(Number(t), S.maxT);
    S.playing = false;
    $('play').textContent = 'Play';
  }
}

function connectLive() {
  const q = new URLSearchParams({
    algo: params.get('algo') || 'heuristic', config: params.get('config') || 'best',
    seed: params.get('seed') || '10003', speed: String(S.speed),
  });
  if (params.get('p_order')) q.set('p_order', params.get('p_order'));
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws?${q}`);
  banner('connecting…');
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'init') { start(msg.meta); banner(''); }
    else if (msg.type === 'step') addStep(msg.step, msg.ticks);
    else if (msg.type === 'end') { S.ended = true; banner('shift complete'); }
  };
  ws.onclose = () => { if (!S.ended) banner('connection closed'); };
}

// ---------- loop ----------

const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (S.playing && S.steps.length) S.simT = Math.min(S.simT + dt * S.speed, S.maxT);
  update();
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  requestAnimationFrame(frame);
}

addHub();
await addBasemap();
if (params.get('ep')) await loadReplay(params.get('ep'));
else connectLive();
update();
setCamera($('camera').value);
frame();
