/**
 * ChurnDashboard.jsx — Phase 3: fully wired to live FastAPI backend
 *
 * All data comes from real API calls via hooks.js / api.js.
 * Zero mock data remains. Loading skeletons + error states on every tab.
 */

import { useState, useRef } from "react";
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";
import { useModelPerformance, useHealth, usePredict, useExplain, useBatch } from "./hooks";

// ── Design tokens ─────────────────────────────────────────────────
const C = {
  bg:        "#0B0E18",
  surface:   "#13172A",
  surface2:  "#1C2140",
  border:    "#252D4A",
  accent:    "#6C63FF",
  accentSoft:"rgba(108,99,255,0.13)",
  red:       "#FF5C72",
  redSoft:   "rgba(255,92,114,0.12)",
  green:     "#3DDFA8",
  greenSoft: "rgba(61,223,168,0.12)",
  amber:     "#FBBF24",
  amberSoft: "rgba(251,191,36,0.12)",
  text:      "#EEF0FA",
  muted:     "#7A80A0",
  white:     "#FFFFFF",
};

// ── Primitive components ──────────────────────────────────────────

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 14, padding: "20px 24px", ...style,
    }}>
      {children}
    </div>
  );
}

function SectionTitle({ children, hint }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 18 }}>
      <div style={{ width: 3, height: 16, background: C.accent, borderRadius: 2, flexShrink: 0, alignSelf: "center" }} />
      <h2 style={{ fontSize: 13, fontWeight: 600, color: C.text, textTransform: "uppercase", letterSpacing: "0.08em", margin: 0 }}>
        {children}
      </h2>
      {hint && <span style={{ fontSize: 11, color: C.muted, fontWeight: 400 }}>{hint}</span>}
    </div>
  );
}

function Skeleton({ h = 18, w = "100%", r = 6, style = {} }) {
  return (
    <div style={{
      height: h, width: w, borderRadius: r,
      background: `linear-gradient(90deg, ${C.surface2} 25%, ${C.border} 50%, ${C.surface2} 75%)`,
      backgroundSize: "200% 100%",
      animation: "shimmer 1.4s infinite",
      ...style,
    }} />
  );
}

function ErrorBanner({ message }) {
  return (
    <div style={{
      padding: "12px 16px", background: C.redSoft, border: `1px solid ${C.red}33`,
      borderRadius: 8, fontSize: 13, color: C.red, display: "flex", gap: 8, alignItems: "center",
    }}>
      <span>⚠</span> {message}
    </div>
  );
}

function Kpi({ label, value, sub, color = C.accent, icon, loading }) {
  return (
    <Card>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.07em" }}>{label}</span>
        <span style={{ fontSize: 18, opacity: 0.8 }}>{icon}</span>
      </div>
      {loading
        ? <Skeleton h={32} w="60%" style={{ marginBottom: 6 }} />
        : <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1, marginBottom: 4 }}>{value}</div>}
      {loading
        ? <Skeleton h={12} w="80%" />
        : <div style={{ fontSize: 11, color: C.muted }}>{sub}</div>}
    </Card>
  );
}

function TabBar({ tabs, active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 2, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: 4, marginBottom: 28 }}>
      {tabs.map(t => (
        <button key={t} onClick={() => onChange(t)} style={{
          flex: 1, padding: "9px 12px", border: "none", borderRadius: 8, cursor: "pointer",
          fontSize: 12, fontWeight: 600, letterSpacing: "0.03em", transition: "all 0.15s",
          background: active === t ? C.accent : "transparent",
          color: active === t ? C.white : C.muted,
        }}>
          {t}
        </button>
      ))}
    </div>
  );
}

function StatusPill({ status }) {
  const map = {
    "Low":    [C.green, C.greenSoft],
    "Medium": [C.amber, C.amberSoft],
    "High":   [C.red,   C.redSoft],
  };
  const [fg, bg] = map[status] ?? [C.muted, C.surface2];
  return (
    <span style={{ padding: "3px 12px", borderRadius: 20, background: bg, color: fg, fontSize: 12, fontWeight: 700 }}>
      {status} Risk
    </span>
  );
}

// ── Business Impact Tab ───────────────────────────────────────────

function BusinessTab({ data, loading, error }) {
  if (error) return <ErrorBanner message={`Failed to load metrics: ${error}`} />;

  const best = data?.best_model;
  const m    = data?.models?.[best];

  const fmt$  = (n) => `$${(n / 1000).toFixed(0)}K`;
  const fmtPct = (n) => `${(n * 100).toFixed(1)}%`;

  const barData = data
    ? Object.entries(data.models).map(([name, d]) => ({
        name: name === "LogisticRegression" ? "Log. Reg." : name,
        "Revenue Saved":   Math.round(d.revenue_saved_usd / 1000),
        "Retention Cost":  Math.round(d.retention_cost_usd / 1000),
        "Net Value":       Math.round(d.net_value_usd / 1000),
      }))
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        <Kpi loading={loading} label="Revenue Saved"   icon="💰" color={C.green}
             value={m ? fmt$(m.revenue_saved_usd) : "-"}
             sub={m ? `${m.caught_churners} churners caught` : ""} />
        <Kpi loading={loading} label="Retention Cost"  icon="📬" color={C.amber}
             value={m ? fmt$(m.retention_cost_usd) : "-"}
             sub="@ $20 per outreach" />
        <Kpi loading={loading} label="Net Value"       icon="📈" color={C.accent}
             value={m ? fmt$(m.net_value_usd) : "-"}
             sub="Savings minus cost" />
        <Kpi loading={loading} label="Recall"          icon="🎯" color={C.green}
             value={m ? fmtPct(m.recall) : "-"}
             sub={m ? `${m.caught_churners}/${m.actual_churners} churners` : ""} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: 18 }}>
        <Card>
          <SectionTitle>Business Value by Model</SectionTitle>
          {loading ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[1,2,3].map(i => <Skeleton key={i} h={40} />)}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={barData} barCategoryGap="28%">
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}K`} />
                <Tooltip contentStyle={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }}
                         labelStyle={{ color: C.text }} formatter={(v, n) => [`$${v}K`, n]} />
                <Legend wrapperStyle={{ fontSize: 11, color: C.muted }} />
                <Bar dataKey="Revenue Saved"  fill={C.green}  radius={[4,4,0,0]} />
                <Bar dataKey="Retention Cost" fill={C.amber}  radius={[4,4,0,0]} />
                <Bar dataKey="Net Value"      fill={C.accent} radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card>
          <SectionTitle>Customer Funnel</SectionTitle>
          {loading ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[1,2,3,4].map(i => <Skeleton key={i} h={28} />)}
            </div>
          ) : m ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {[
                { label: "Total Customers",   val: m.tp + m.fp + m.fn + m.tn, color: C.accent },
                { label: "Actual Churners",   val: m.actual_churners,          color: C.amber  },
                { label: "Caught by Model",   val: m.caught_churners,          color: C.green  },
              ].map(({ label, val, color }) => {
                const total = m.tp + m.fp + m.fn + m.tn;
                return (
                  <div key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                      <span style={{ fontSize: 12, color: C.muted }}>{label}</span>
                      <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>{val.toLocaleString()}</span>
                    </div>
                    <div style={{ height: 7, background: C.border, borderRadius: 4, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${(val / total) * 100}%`, background: color, borderRadius: 4, transition: "width 0.8s ease" }} />
                    </div>
                  </div>
                );
              })}
              <div style={{ padding: "12px 14px", background: C.redSoft, borderRadius: 8, marginTop: 4 }}>
                <div style={{ fontSize: 11, color: C.muted, marginBottom: 2 }}>Missed churners (FN)</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: C.red }}>{m.missed_churners}</div>
                <div style={{ fontSize: 11, color: C.muted }}>
                  = ${(m.missed_churners * 65 * 24).toLocaleString()} avoidable lost CLV
                </div>
              </div>
            </div>
          ) : null}
        </Card>
      </div>

      {/* Confusion matrix */}
      {m && (
        <Card>
          <SectionTitle hint="Threshold optimized for $500 FN / $20 FP cost matrix">
            Confusion Matrix — {best}
          </SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, maxWidth: 480 }}>
            {[
              { label: "True Negative",  val: m.tn, cost: null,            color: C.green,  tag: "TN — Correctly stayed" },
              { label: "False Positive", val: m.fp, cost: m.fp * 20,       color: C.amber,  tag: "FP — False alarm ($20 each)" },
              { label: "False Negative", val: m.fn, cost: m.fn * 500,      color: C.red,    tag: "FN — Missed churner ($500 each)" },
              { label: "True Positive",  val: m.tp, cost: null,            color: C.accent, tag: "TP — Correctly caught" },
            ].map(({ label, val, cost, color, tag }) => (
              <div key={label} style={{ padding: "14px 16px", background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 10 }}>
                <div style={{ fontSize: 28, fontWeight: 800, color }}>{val.toLocaleString()}</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>{tag}</div>
                {cost != null && <div style={{ fontSize: 12, fontWeight: 600, color, marginTop: 4 }}>${cost.toLocaleString()} total</div>}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Model Performance Tab ─────────────────────────────────────────

function PerformanceTab({ data, loading, error }) {
  if (error) return <ErrorBanner message={`Failed to load metrics: ${error}`} />;

  const models = data
    ? Object.entries(data.models).map(([name, d]) => ({
        name: name === "LogisticRegression" ? "Log. Reg." : name,
        fullName: name,
        auc:  +(d.roc_auc * 100).toFixed(2),
        cv:   +(d.cv_auc_mean * 100).toFixed(2),
        rec:  +(d.recall * 100).toFixed(1),
        prec: +(d.precision * 100).toFixed(1),
        f1:   +(d.f1 * 100).toFixed(1),
        thr:  d.threshold,
        isBest: name === data.best_model,
      }))
    : [];

  // Approximate ROC from AUC values for chart
  const rocData = Array.from({ length: 9 }, (_, i) => {
    const fpr = (i + 1) / 10;
    const row = { fpr: +fpr.toFixed(1) };
    models.forEach(m => {
      const auc = m.auc / 100;
      row[m.name] = +Math.min(1, Math.pow(fpr, 1 / (auc / (1 - auc) * 2))).toFixed(3);
    });
    return row;
  });
  rocData.unshift({ fpr: 0, ...Object.fromEntries(models.map(m => [m.name, 0])) });
  rocData.push({ fpr: 1, ...Object.fromEntries(models.map(m => [m.name, 1])) });

  const modelColors = [C.accent, C.green, C.amber];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <Card>
        <SectionTitle hint="★ = selected for deployment">Model Comparison</SectionTitle>
        {loading ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {[1,2,3].map(i => <Skeleton key={i} h={44} />)}
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr>
                  {["Model", "CV AUC", "Test AUC", "Recall", "Precision", "F1", "Threshold"].map(h => (
                    <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: C.muted, fontWeight: 500, borderBottom: `1px solid ${C.border}`, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={m.name} style={{ background: m.isBest ? C.accentSoft : "transparent" }}>
                    <td style={{ padding: "12px 14px", color: C.text, fontWeight: m.isBest ? 700 : 400 }}>
                      <span style={{ color: modelColors[i], marginRight: 6, fontSize: 10 }}>●</span>
                      {m.isBest && "★ "}{m.name}
                    </td>
                    <td style={{ padding: "12px 14px", color: C.muted }}>{m.cv}%</td>
                    <td style={{ padding: "12px 14px", color: C.text, fontWeight: 600 }}>{m.auc}%</td>
                    <td style={{ padding: "12px 14px", color: C.green, fontWeight: 700 }}>{m.rec}%</td>
                    <td style={{ padding: "12px 14px", color: C.text }}>{m.prec}%</td>
                    <td style={{ padding: "12px 14px", color: C.text }}>{m.f1}%</td>
                    <td style={{ padding: "12px 14px", color: C.amber }}>{m.thr}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div style={{ marginTop: 12, padding: "10px 14px", background: C.accentSoft, borderRadius: 8, fontSize: 12, color: C.muted }}>
          Threshold is optimized for cost matrix (FN=$500 / FP=$20), not the default 0.5.
          This shifts recall from ~80% → ~99% at the cost of precision.
        </div>
      </Card>

      <Card>
        <SectionTitle hint="approximate from AUC values">ROC Curves</SectionTitle>
        {loading ? <Skeleton h={220} /> : (
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="fpr" tick={{ fill: C.muted, fontSize: 10 }} tickFormatter={v => v.toFixed(1)}
                     label={{ value: "False Positive Rate", position: "insideBottom", offset: -4, fill: C.muted, fontSize: 11 }} />
              <YAxis tick={{ fill: C.muted, fontSize: 10 }}
                     label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: C.muted, fontSize: 11 }} />
              <Tooltip contentStyle={{ background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 11 }}
                       labelFormatter={v => `FPR: ${v}`} />
              <Legend wrapperStyle={{ color: C.muted, fontSize: 11 }} />
              {models.map((m, i) => (
                <Line key={m.name} type="monotone" dataKey={m.name} stroke={modelColors[i]}
                      strokeWidth={m.isBest ? 3 : 1.8} dot={false} name={`${m.name} (AUC=${m.auc}%)`} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>
    </div>
  );
}

// ── Live Predict Tab ──────────────────────────────────────────────

const DEFAULT_CUSTOMER = {
  gender: "Male", SeniorCitizen: 0, Partner: "No", Dependents: "No",
  tenure: 6, PhoneService: "Yes", MultipleLines: "No",
  InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No",
  DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes",
  StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check", MonthlyCharges: 85.5, TotalCharges: 513.0,
};

function FieldRow({ label, children }) {
  return (
    <div>
      <label style={{ fontSize: 11, color: C.muted, display: "block", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

const inputStyle = {
  width: "100%", padding: "9px 12px", background: C.surface2,
  border: `1px solid ${C.border}`, borderRadius: 7, color: C.text,
  fontSize: 13, boxSizing: "border-box", outline: "none",
};

function PredictTab() {
  const [form, setForm] = useState(DEFAULT_CUSTOMER);
  const { result, loading, error, predict }   = usePredict();
  const { result: explanation, loading: explLoading, explain } = useExplain();

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = () => {
    predict(form);
    explain(form);
  };

  const riskColor = result
    ? ({ Low: C.green, Medium: C.amber, High: C.red }[result.risk_tier] ?? C.muted)
    : C.muted;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
      {/* Left: form */}
      <Card style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        <SectionTitle>Customer Profile</SectionTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <FieldRow label="Tenure (months)">
              <input type="number" style={inputStyle} value={form.tenure}
                     onChange={e => set("tenure", +e.target.value)} />
            </FieldRow>
            <FieldRow label="Monthly Charges ($)">
              <input type="number" style={inputStyle} value={form.MonthlyCharges}
                     onChange={e => set("MonthlyCharges", +e.target.value)} />
            </FieldRow>
          </div>
          <FieldRow label="Total Charges ($)">
            <input type="number" style={inputStyle} value={form.TotalCharges}
                   onChange={e => set("TotalCharges", +e.target.value)} />
          </FieldRow>
          {[
            ["Contract",        "Contract",        ["Month-to-month","One year","Two year"]],
            ["Internet Service","InternetService",  ["Fiber optic","DSL","No"]],
            ["Payment Method",  "PaymentMethod",    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]],
            ["Multiple Lines",  "MultipleLines",    ["Yes","No","No phone service"]],
            ["Online Security", "OnlineSecurity",   ["Yes","No","No internet service"]],
            ["Online Backup",   "OnlineBackup",     ["Yes","No","No internet service"]],
            ["Tech Support",    "TechSupport",      ["Yes","No","No internet service"]],
            ["Streaming TV",    "StreamingTV",      ["Yes","No","No internet service"]],
          ].map(([label, key, opts]) => (
            <FieldRow key={key} label={label}>
              <select style={inputStyle} value={form[key]} onChange={e => set(key, e.target.value)}>
                {opts.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            </FieldRow>
          ))}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[["Gender","gender",["Male","Female"]],["Partner","Partner",["Yes","No"]],
              ["Dependents","Dependents",["Yes","No"]],["Paperless Billing","PaperlessBilling",["Yes","No"]]
             ].map(([label, key, opts]) => (
              <FieldRow key={key} label={label}>
                <select style={inputStyle} value={form[key]} onChange={e => set(key, e.target.value)}>
                  {opts.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </FieldRow>
            ))}
          </div>
          <button onClick={handleSubmit} disabled={loading}
            style={{
              padding: "12px", background: loading ? C.border : C.accent, color: C.white,
              border: "none", borderRadius: 8, fontSize: 13, fontWeight: 700,
              cursor: loading ? "not-allowed" : "pointer", marginTop: 4,
              transition: "background 0.15s",
            }}>
            {loading ? "Predicting…" : "Predict Churn Risk →"}
          </button>
          {error && <ErrorBanner message={error} />}
        </div>
      </Card>

      {/* Right: result */}
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <Card>
          <SectionTitle>Prediction</SectionTitle>
          {!result && !loading && (
            <div style={{ color: C.muted, fontSize: 13, padding: "40px 0", textAlign: "center" }}>
              Fill in the form and click Predict
            </div>
          )}
          {loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14, alignItems: "center", padding: "20px 0" }}>
              <Skeleton h={56} w="140px" r={10} />
              <Skeleton h={20} w="180px" />
              <Skeleton h={14} w="120px" />
            </div>
          )}
          {result && !loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div style={{ textAlign: "center", padding: "10px 0" }}>
                <div style={{ fontSize: 58, fontWeight: 900, color: riskColor, lineHeight: 1 }}>
                  {(result.churn_probability * 100).toFixed(1)}%
                </div>
                <div style={{ marginTop: 8 }}>
                  <StatusPill status={result.risk_tier} />
                </div>
              </div>
              <div style={{ height: 1, background: C.border }} />
              {[
                ["Prediction",     result.churn_prediction ? "⚠ Will Churn" : "✓ Will Stay", result.churn_prediction ? C.red : C.green],
                ["CLV at Risk",    `$${result.estimated_clv_at_risk.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, C.amber],
                ["Threshold Used", result.threshold_used, C.muted],
              ].map(([k, v, c]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 12, color: C.muted }}>{k}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: c }}>{v}</span>
                </div>
              ))}
              <div style={{ padding: "12px 14px", background: C.accentSoft, borderRadius: 8, fontSize: 12, color: C.muted, lineHeight: 1.6 }}>
                <strong style={{ color: C.text }}>Recommended action: </strong>
                {result.risk_tier === "High"   ? "Immediate proactive outreach — offer a contract upgrade or loyalty discount before next billing cycle." :
                 result.risk_tier === "Medium" ? "Flag for next monthly retention campaign. Monitor payment pattern." :
                 "No intervention needed. Standard engagement is sufficient."}
              </div>
            </div>
          )}
        </Card>

        {/* SHAP explanation */}
        <Card>
          <SectionTitle hint="top 8 factors">Why This Prediction?</SectionTitle>
          {!explanation && !explLoading && (
            <div style={{ color: C.muted, fontSize: 12, textAlign: "center", padding: "24px 0" }}>
              Run a prediction to see the driver breakdown
            </div>
          )}
          {explLoading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[1,2,3,4,5,6].map(i => <Skeleton key={i} h={22} />)}
            </div>
          )}
          {explanation && !explLoading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {explanation.top_drivers.slice(0, 8).map((d, i) => {
                const maxShap = Math.max(...explanation.top_drivers.map(x => x.abs_shap));
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 150, fontSize: 10, color: C.muted, textAlign: "right", flexShrink: 0, lineHeight: 1.3 }}>
                      {d.feature.replace(/_/g, " ").slice(0, 24)}
                    </div>
                    <div style={{ flex: 1, height: 18, background: C.border, borderRadius: 4, overflow: "hidden" }}>
                      <div style={{
                        height: "100%", borderRadius: 4,
                        width: `${(d.abs_shap / maxShap) * 100}%`,
                        background: d.direction === "increases_churn" ? C.red : C.accent,
                        opacity: 0.85,
                      }} />
                    </div>
                    <span style={{ fontSize: 10, fontWeight: 600, color: d.direction === "increases_churn" ? C.red : C.accent, width: 44, textAlign: "right" }}>
                      {d.raw_shap > 0 ? "+" : ""}{d.raw_shap.toFixed(3)}
                    </span>
                  </div>
                );
              })}
              <div style={{ marginTop: 6, fontSize: 11, color: C.muted }}>
                Red = increases churn risk · Blue = decreases churn risk
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

// ── Batch Upload Tab ──────────────────────────────────────────────

function BatchTab() {
  const { result, loading, error, submit } = useBatch();
  const fileRef = useRef();
  const [fileName, setFileName] = useState(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFileName(f.name);
    submit(f);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <Card>
        <SectionTitle hint="CSV with same columns as training data">Batch Prediction Upload</SectionTitle>
        <div style={{
          border: `2px dashed ${C.border}`, borderRadius: 10, padding: "36px 24px",
          textAlign: "center", cursor: "pointer", transition: "border-color 0.15s",
        }}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) { setFileName(f.name); submit(f); } }}
        >
          <input ref={fileRef} type="file" accept=".csv" style={{ display: "none" }} onChange={handleFile} />
          <div style={{ fontSize: 32, marginBottom: 10 }}>📂</div>
          <div style={{ fontSize: 14, color: C.text, fontWeight: 600 }}>
            {fileName ?? "Drop a CSV or click to browse"}
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>
            Expects Telco schema (20 feature columns). CustomerID and Churn columns are dropped if present.
          </div>
        </div>
        {loading && (
          <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 10 }}>
            <Skeleton h={24} />
            <Skeleton h={24} w="70%" />
            <Skeleton h={24} w="50%" />
          </div>
        )}
        {error && <div style={{ marginTop: 12 }}><ErrorBanner message={error} /></div>}
      </Card>

      {result && !loading && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
            <Kpi label="Total Customers"   value={result.total_customers.toLocaleString()}  icon="👥" color={C.text} />
            <Kpi label="Predicted Churners" value={result.predicted_churners.toLocaleString()} icon="⚠️" color={C.red} />
            <Kpi label="Churn Rate"        value={`${(result.churn_rate * 100).toFixed(1)}%`} icon="📉" color={C.amber} />
            <Kpi label="Revenue at Risk"   value={`$${(result.total_revenue_at_risk / 1000).toFixed(0)}K`} icon="💸" color={C.red} />
          </div>

          <Card>
            <SectionTitle>Risk Distribution (first 200 rows)</SectionTitle>
            <div style={{ overflowX: "auto", maxHeight: 320, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead style={{ position: "sticky", top: 0, background: C.surface }}>
                  <tr>
                    {["#","Churn Prob.","Prediction","Risk Tier","CLV at Risk"].map(h => (
                      <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: C.muted, fontWeight: 500, borderBottom: `1px solid ${C.border}` }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.slice(0, 200).map((p, i) => {
                    const color = { Low: C.green, Medium: C.amber, High: C.red }[p.risk_tier] ?? C.muted;
                    return (
                      <tr key={i} style={{ borderBottom: `1px solid ${C.border}22` }}>
                        <td style={{ padding: "7px 12px", color: C.muted }}>{i + 1}</td>
                        <td style={{ padding: "7px 12px", color: C.text, fontWeight: 600 }}>
                          {(p.churn_probability * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: "7px 12px" }}>
                          <span style={{ color: p.churn_prediction ? C.red : C.green, fontWeight: 600 }}>
                            {p.churn_prediction ? "Churn" : "Stay"}
                          </span>
                        </td>
                        <td style={{ padding: "7px 12px" }}>
                          <span style={{ color, fontWeight: 600 }}>{p.risk_tier}</span>
                        </td>
                        <td style={{ padding: "7px 12px", color: C.amber }}>${p.clv_at_risk.toLocaleString()}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

// ── Header status bar ─────────────────────────────────────────────

function ApiStatus() {
  const { data, loading, error } = useHealth();
  if (loading) return <span style={{ fontSize: 11, color: C.muted }}>Connecting…</span>;
  if (error)   return <span style={{ fontSize: 11, color: C.red }}>● API offline</span>;
  return (
    <span style={{ fontSize: 11, color: C.green }}>
      ● API live · {data.feature_count} features · threshold {data.threshold}
    </span>
  );
}

// ── Root App ──────────────────────────────────────────────────────

export default function ChurnDashboard() {
  const [tab, setTab] = useState("Business Impact");
  const { data, loading, error } = useModelPerformance();
  const best = data?.best_model;
  const m    = data?.models?.[best];

  return (
    <>
      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; background: ${C.bg}; }
        @keyframes shimmer {
          0%   { background-position: -200% 0; }
          100% { background-position:  200% 0; }
        }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
      `}</style>

      <div style={{ background: C.bg, minHeight: "100vh", color: C.text,
                    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
                    padding: "24px 32px", maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 28 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
              <div style={{ width: 36, height: 36, background: C.accent, borderRadius: 10,
                            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>
                📉
              </div>
              <h1 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0, letterSpacing: "-0.02em" }}>
                Churn Prediction
              </h1>
              {m && !loading && (
                <div style={{ padding: "3px 10px", background: C.greenSoft, color: C.green, borderRadius: 20, fontSize: 10, fontWeight: 700, letterSpacing: "0.05em" }}>
                  LIVE
                </div>
              )}
            </div>
            <div style={{ fontSize: 12, color: C.muted, marginLeft: 48 }}>
              {loading ? "Loading model…" : m
                ? `${best} · AUC ${(m.roc_auc * 100).toFixed(1)}% · Recall ${(m.recall * 100).toFixed(1)}% · Net value $${(m.net_value_usd / 1000).toFixed(0)}K`
                : "Telco Customer Churn"}
            </div>
          </div>
          <ApiStatus />
        </div>

        <TabBar
          tabs={["Business Impact", "Model Performance", "Live Predict", "Batch Upload"]}
          active={tab}
          onChange={setTab}
        />

        {tab === "Business Impact"    && <BusinessTab  data={data} loading={loading} error={error} />}
        {tab === "Model Performance"  && <PerformanceTab data={data} loading={loading} error={error} />}
        {tab === "Live Predict"       && <PredictTab />}
        {tab === "Batch Upload"       && <BatchTab />}

        <div style={{ marginTop: 36, paddingTop: 16, borderTop: `1px solid ${C.border}`,
                      fontSize: 11, color: C.muted, display: "flex", justifyContent: "space-between" }}>
          <span>Padmapriya · NYU Center for Data Science · FastAPI + React + SHAP</span>
          <span>IBM Telco Churn Dataset</span>
        </div>
      </div>
    </>
  );
}
