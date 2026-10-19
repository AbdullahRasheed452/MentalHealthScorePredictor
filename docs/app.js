// Computes the wellbeing score in the browser from docs/model.json
// (a logistic regression exported by src/train.py).

const form = document.getElementById("form");
let model = null;
// The score stays hidden until the user changes an answer. Most students in the
// training data reported depression, so the untouched defaults score low, and
// showing that before anyone has answered would be misleading.
let touched = false;

// Factors shown to the user. Each maps to one or more model features.
const GROUPS = [
  { label: "Academic pressure", features: ["Academic Pressure"],
    tip: "Break big deadlines into smaller steps, and ask tutors for help early when a module feels overwhelming." },
  { label: "Satisfaction with studies", features: ["Study Satisfaction"],
    tip: "If your course isn't working for you, an academic advisor can help you look at options." },
  { label: "Daily study / work hours", features: ["Work/Study Hours"],
    tip: "Plan real breaks into long days. Rest is part of studying well." },
  { label: "Financial stress", features: ["Financial Stress"],
    tip: "Most universities offer hardship funds and free money-advice services. They're worth asking about." },
  { label: "Sleep", features: ["Sleep: Less than 5 hours", "Sleep: 5-6 hours", "Sleep: More than 8 hours"],
    tip: "Aim for 7–8 hours with a regular bedtime, and avoid screens just before sleep." },
  { label: "Diet", features: ["Diet: Moderate", "Diet: Unhealthy"],
    tip: "Regular meals with fruit and vegetables help with energy and mood." },
  { label: "CGPA", features: ["CGPA"], tip: "" },
  { label: "Age", features: ["Age"], tip: "" },
  { label: "Family history of mental illness", features: ["Family history"],
    tip: "You can't change this, but it's useful to mention if you ever speak to a doctor." },
];

function readInputs() {
  const f = new FormData(form);
  const num = (name) => Number(f.get(name));
  const sleep = f.get("sleep");
  const diet = f.get("diet");
  return {
    "Age": num("age"),
    "Academic Pressure": num("pressure"),
    "CGPA": num("cgpa"),
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

// Returns the score and each feature's contribution to the risk (log-odds),
// measured relative to an average student in the training data.
function predict(x) {
  let logit = model.intercept;
  const contrib = {};
  for (const feat of model.features) {
    const c = feat.coef * ((x[feat.name] - feat.mean) / feat.std);
    contrib[feat.name] = c;
    logit += c;
  }
  const risk = 1 / (1 + Math.exp(-logit));
  return { score: Math.round(100 * (1 - risk)), contrib };
}

function band(score) {
  if (score >= 70) return { name: "Doing well", color: "var(--good)",
    text: "Your answers look similar to students who reported good mental wellbeing. Keep up the habits that are working." };
  if (score >= 40) return { name: "Some strain", color: "var(--mid)",
    text: "Some of your answers are linked to higher stress and low mood. The factors below show where small changes might help most." };
  return { name: "Under pressure", color: "var(--low)",
    text: "Your answers look similar to students who reported depression symptoms. Please consider talking to someone: a friend, your university counselling service, or a doctor." };
}

function renderFactors(contrib) {
  const items = GROUPS.map((g) => ({
    ...g,
    value: g.features.reduce((sum, name) => sum + (contrib[name] ?? 0), 0),
  })).filter((g) => Math.abs(g.value) >= 0.1);

  const down = items.filter((g) => g.value > 0).sort((a, b) => b.value - a.value);
  const up = items.filter((g) => g.value < 0).sort((a, b) => a.value - b.value);

  const strength = (v) => (Math.abs(v) >= 0.6 ? "strong" : Math.abs(v) >= 0.3 ? "moderate" : "small");

  const fill = (el, list, withTip) => {
    el.replaceChildren();
    if (!list.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "Nothing notable.";
      el.append(li);
      return;
    }
    for (const g of list.slice(0, 4)) {
      const li = document.createElement("li");
      li.textContent = `${g.label} (${strength(g.value)} effect)`;
      if (withTip && g.tip) {
        const tip = document.createElement("small");
        tip.textContent = g.tip;
        li.append(tip);
      }
      el.append(li);
    }
  };

  fill(document.getElementById("down"), down, true);
  fill(document.getElementById("up"), up, false);
}

function update() {
  for (const out of document.querySelectorAll("output[data-for]")) {
    out.textContent = form.elements[out.dataset.for].value;
  }
  if (!model) return;
  document.querySelector(".factors").hidden = !touched;
  if (!touched) {
    document.getElementById("score").textContent = "–";
    document.getElementById("band").textContent = "Your score will appear here";
    document.getElementById("band").style.color = "";
    document.getElementById("band-text").textContent =
      "Adjust the answers above to match your own situation.";
    return;
  }

  const { score, contrib } = predict(readInputs());
  const b = band(score);
  document.getElementById("score").textContent = score;
  document.getElementById("band").textContent = b.name;
  document.getElementById("band").style.color = b.color;
  document.getElementById("band-text").textContent = b.text;

  const bar = document.getElementById("bar");
  const circumference = 2 * Math.PI * 52;
  bar.style.strokeDashoffset = circumference * (1 - score / 100);
  bar.style.stroke = b.color;

  renderFactors(contrib);
}

form.addEventListener("input", () => {
  touched = true;
  update();
});

fetch("model.json")
  .then((r) => {
    if (!r.ok) throw new Error(r.status);
    return r.json();
  })
  .then((m) => {
    model = m;
    document.getElementById("model-info").textContent =
      `Model: logistic regression trained on ${m.trained_on_rows.toLocaleString()} student survey responses (test AUC ${m.test_auc}).`;
    update();
  })
  .catch(() => {
    document.getElementById("band").textContent = "Couldn't load the model.";
    document.getElementById("band-text").textContent =
      "Run python src/train.py first, then serve the docs folder (see README).";
  });

update();
