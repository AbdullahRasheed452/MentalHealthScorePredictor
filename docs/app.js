// Computes the wellbeing score in the browser from docs/model.json
// (a logistic regression exported by src/train.py).

const form = document.getElementById("form");
const $ = (id) => document.getElementById(id);
let model = null;
// The score stays hidden until the user changes an answer. Most students in the
// training data reported depression, so the untouched defaults score low, and
// showing that before anyone has answered would be misleading.
let touched = false;

// Words shown next to the 1–5 sliders.
const LEVELS = {
  pressure: ["Very low", "Low", "Moderate", "High", "Very high"],
  satisfaction: ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"],
  finance: ["None", "Mild", "Moderate", "High", "Severe"],
};

// The form asks for CGPA out of 4.0; the model was trained on a 10-point scale.
const CGPA_SCALE = 10 / 4;

// Factors shown to the user. Each maps to one or more model features.
const GROUPS = [
  { label: "Academic pressure", features: ["Academic Pressure"],
    tip: "Break big deadlines into smaller steps, and ask tutors for help early when a module feels overwhelming." },
  { label: "Study satisfaction", features: ["Study Satisfaction"],
    tip: "If your course isn't working for you, an academic advisor can help you look at options." },
  { label: "Study hours", features: ["Work/Study Hours"],
    tip: "Plan real breaks into long days. Rest is part of studying well." },
  { label: "Financial stress", features: ["Financial Stress"],
    tip: "Most universities offer hardship funds and free money-advice services. They're worth asking about." },
  { label: "Sleep", features: ["Sleep: Less than 5 hours", "Sleep: 5-6 hours", "Sleep: More than 8 hours"],
    tip: "Aim for 7–8 hours with a regular bedtime, and avoid screens just before sleep." },
  { label: "Diet", features: ["Diet: Moderate", "Diet: Unhealthy"],
    tip: "Regular meals with fruit and vegetables help with energy and mood." },
  { label: "CGPA", features: ["CGPA"], tip: "" },
  { label: "Family history", features: ["Family history"],
    tip: "You can't change this, but it's useful to mention if you ever speak to a doctor." },
];

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const toScore = (risk) => Math.min(99, Math.max(1, Math.round(100 * (1 - risk))));

function readCgpa() {
  const raw = form.elements.cgpa.value.trim();
  const v = Number(raw);
  const valid = raw !== "" && Number.isFinite(v) && v >= 0 && v <= 4;
  return { valid, value: valid ? v * CGPA_SCALE : null };
}

function readInputs() {
  const f = new FormData(form);
  const num = (name) => Number(f.get(name));
  const sleep = f.get("sleep");
  const diet = f.get("diet");
  return {
    "Academic Pressure": num("pressure"),
    "CGPA": readCgpa().value,
    "Study Satisfaction": num("satisfaction"),
    "Work/Study Hours": num("hours"),
    "Financial Stress": num("finance"),
    "Sleep: Less than 5 hours": sleep === "Less than 5 hours" ? 1 : 0,
    "Sleep: 5-6 hours": sleep === "5-6 hours" ? 1 : 0,
    "Sleep: More than 8 hours": sleep === "More than 8 hours" ? 1 : 0,
    "Diet: Moderate": diet === "Moderate" ? 1 : 0,
    "Diet: Unhealthy": diet === "Unhealthy" ? 1 : 0,
    "Family history": f.get("family") === "Yes" ? 1 : 0,
  };
}

// Returns the risk and each feature's contribution to it (log-odds), measured
// relative to an average student in the training data. A missing value (an
// invalid CGPA) is treated as average, so it contributes nothing.
function predict(x) {
  let logit = model.intercept;
  const contrib = {};
  for (const feat of model.features) {
    const value = x[feat.name] ?? feat.mean;
    const c = feat.coef * ((value - feat.mean) / feat.std);
    contrib[feat.name] = c;
    logit += c;
  }
  return { logit, risk: sigmoid(logit), contrib };
}

// How many score points each answer adds or removes on its own: the change it
// would make to an average student's score. Measured from the average rather
// than from the user's own score, because near 1 or 99 the score is saturated
// and every single factor would look tiny.
function factorPoints({ contrib }) {
  const average = sigmoid(model.intercept);
  return GROUPS.map((g) => {
    const own = g.features.reduce((sum, name) => sum + (contrib[name] ?? 0), 0);
    return { ...g, points: 100 * (average - sigmoid(model.intercept + own)) };
  });
}

function band(score) {
  if (score >= 70) return { name: "Doing well", color: "var(--good)",
    text: "Your answers look similar to students who reported good mental wellbeing. Keep up the habits that are working." };
  if (score >= 40) return { name: "Some strain", color: "var(--mid)",
    text: "Some of your answers are linked to higher stress and low mood. The factors below show where small changes might help most." };
  return { name: "Under pressure", color: "var(--low)",
    text: "Your answers look similar to students who reported depression symptoms. Please consider talking to someone you trust." };
}

const fmt = (p) => (p > 0 ? "+" : "−") + Math.abs(Math.round(p));

function renderFactors(groups) {
  const shown = groups.filter((g) => Math.abs(g.points) >= 1)
    .sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
  const list = $("factors");
  list.replaceChildren();

  if (!shown.length) {
    const li = document.createElement("li");
    li.innerHTML = '<span class="empty">Your answers are close to the survey average on every factor.</span>';
    list.append(li);
  }
  // A fixed minimum scale keeps small effects looking small.
  const scale = Math.max(15, ...shown.map((g) => Math.abs(g.points)));
  for (const g of shown) {
    const helping = g.points > 0;
    const li = document.createElement("li");
    li.setAttribute("aria-label",
      `${g.label}: ${helping ? "helping" : "pulling down"} your score by about ${Math.abs(Math.round(g.points))} points`);
    const name = document.createElement("span");
    name.className = "f-name";
    name.textContent = g.label;
    const val = document.createElement("span");
    val.className = "f-val";
    val.textContent = fmt(g.points);
    const track = document.createElement("span");
    track.className = "f-track";
    track.setAttribute("aria-hidden", "true");
    const bar = document.createElement("span");
    bar.className = `f-bar ${helping ? "pos" : "neg"}`;
    bar.style.width = `${(Math.abs(g.points) / scale) * 50}%`;
    track.append(bar);
    li.append(name, track, val);
    list.append(li);
  }

  // Tips for the two factors pulling the score down the most.
  const tips = shown.filter((g) => g.points <= -1 && g.tip).slice(0, 2);
  $("tips-box").hidden = !tips.length;
  $("tips").replaceChildren(...tips.map((g) => {
    const li = document.createElement("li");
    const b = document.createElement("strong");
    b.textContent = `${g.label}: `;
    li.append(b, g.tip);
    return li;
  }));
}

function updateOutputs() {
  for (const out of document.querySelectorAll("output[data-for]")) {
    const name = out.dataset.for;
    const v = Number(form.elements[name].value);
    out.textContent = LEVELS[name] ? `${LEVELS[name][v - 1]} (${v}/5)` : `${v} h`;
  }
  const { valid } = readCgpa();
  $("cgpa-error").hidden = valid;
  form.elements.cgpa.setAttribute("aria-invalid", String(!valid));
}

function update() {
  updateOutputs();
  if (!model) return;

  document.body.classList.toggle("has-score", touched);
  $("bands").hidden = !touched;
  $("details").hidden = !touched;
  if (!touched) {
    $("score").textContent = "–";
    $("band").textContent = "Your score will appear here";
    $("band").style.color = "";
    $("band-text").textContent = "Adjust the answers to match your own situation.";
    $("ring").style.strokeDashoffset = "";
    syncMobileBar();
    return;
  }

  const prediction = predict(readInputs());
  const score = toScore(prediction.risk);
  const b = band(score);
  $("score").textContent = score;
  $("band").textContent = b.name;
  $("band").style.color = b.color;
  $("band-text").textContent = b.text;
  $("ring").style.strokeDashoffset = 2 * Math.PI * 52 * (1 - score / 100);
  $("ring").style.stroke = b.color;
  $("marker").style.left = `${score}%`;
  $("support").hidden = score >= 40;
  $("mb-score").textContent = score;
  $("mb-band").textContent = b.name;
  $("mb-band").style.color = b.color;

  renderFactors(factorPoints(prediction));
  syncMobileBar();
}

// On narrow screens the result is below the form, so a small bar at the bottom
// shows the live score while the result card itself is out of view.
let resultVisible = false;
function syncMobileBar() {
  $("mobile-bar").hidden = !touched || resultVisible;
}
new IntersectionObserver(([entry]) => {
  resultVisible = entry.isIntersecting;
  syncMobileBar();
}, { threshold: 0.15 }).observe($("result"));
$("mb-jump").addEventListener("click", () => $("result").scrollIntoView({ behavior: "smooth", block: "start" }));

form.addEventListener("input", () => {
  touched = true;
  update();
});

$("reset-btn").addEventListener("click", () => {
  form.reset();
  touched = false;
  update();
});

fetch("model.json")
  .then((r) => {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  })
  .then((m) => {
    model = m;
    $("model-info").textContent =
      `Model: logistic regression trained on ${m.trained_on_rows.toLocaleString()} student survey responses (test AUC ${m.test_auc}).`;
    update();
  })
  .catch(() => {
    $("band").textContent = "Couldn't load the model.";
    $("band-text").textContent = "Run python src/train.py first, then serve the docs folder (see README).";
  });

update();
