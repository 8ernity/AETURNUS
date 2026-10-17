# AETURNUS Design System: Glassmorphic Aviation Cockpit (HUD PdM)

## 1. Design Concept & Aerospace Vibe
AETURNUS is engineered as an **industrial-grade aviation cockpit and telemetry station** — combining the aesthetics of modern aerospace flight displays (SpaceX Crew Dragon, F-35 panoramic cockpit displays) with the precision of professional acoustic/telemetry hardware analyzers.

- **Theme**: Dark mode by default, ultra-high-contrast industrial flight telemetry.
- **Surface Depth**: Obsidian void backgrounds layered with multi-tier frosted glass panels (`backdrop-blur-xl`), razor-sharp micro-borders, and subtle cockpit mesh gradients.
- **Data Density**: High-density aviation telemetry with monospace numeric readouts (`tabular-nums`), HUD reticles, and live visual telemetry sparklines.
- **Status Signals**: Vivid neon photon glows for immediate situational awareness under split-second operational decisions.

---

## 2. Color Palette & Design Tokens

### Backgrounds & Surfaces
| Token Name | Hex / CSS Value | Description / Usage |
| :--- | :--- | :--- |
| `--bg-void` | `#060B13` | Deepest obsidian black-navy base |
| `--bg-cockpit` | `#0B1120` | Cockpit primary background |
| `--surface-glass` | `rgba(11, 17, 32, 0.72)` | Primary frosted card layer |
| `--surface-glass-elevated` | `rgba(15, 23, 42, 0.85)` | Elevated telemetry and popover card |
| `--border-hud` | `rgba(56, 189, 248, 0.16)` | Subtle cyan-tinted avionics micro-border |
| `--border-hud-highlight` | `rgba(0, 240, 255, 0.45)` | Active / focused card outline |

### Primary HUD & Telemetry Accents
| Accent | Hex | Usage |
| :--- | :--- | :--- |
| **HUD Cyan** | `#00F0FF` (`#38BDF8`) | Active telemetry channels, cycles, sensor readouts |
| **Avionics Blue** | `#3B82F6` | Secondary telemetry lines, safe operational baselines |
| **Deep Electric Indigo** | `#6366F1` | Primary CTA, brand identity, navigation active state |
| **Ion Purple** | `#A855F7` | Attribution layers, secondary diagnostic signals |

### Industrial Severity & Status Matrix
| Status | Hex | Badge Style | Operational Definition |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | `#FF1E44` | Neon Crimson (`rgba(255,30,68,0.18)`, border `#FF1E44`) | RUL ≤ 15 cyc — Immediate AOG / borescope dispatch |
| **HIGH** | `#FF7A00` | Vivid Amber (`rgba(255,122,0,0.18)`, border `#FF7A00`) | RUL 16-30 cyc — Schedule depot overhaul in 3 days |
| **MEDIUM** | `#FFB800` | Warning Gold (`rgba(255,184,0,0.18)`, border `#FFB800`) | RUL 31-60 cyc — Elevated sensor sampling frequency |
| **LOW** | `#00D2FF` | Aero Cyan (`rgba(0,210,255,0.15)`, border `#00D2FF`) | RUL 61-100 cyc — Routine flight envelope monitoring |
| **HEALTHY** | `#00FF87` | Photon Emerald (`rgba(0,255,135,0.18)`, border `#00FF87`) | RUL > 100 cyc — Nominal flight readiness |

---

## 3. Typography & Numerical Readouts
- **Primary Interface**: Inter / system UI font with crisp subpixel anti-aliasing.
- **Avionics Telemetry & Figures**: Monospace (`font-mono`, `JetBrains Mono` / `ui-monospace`), enforcing `tabular-nums` so cycles, RUL figures, and percentages align without visual jitter.
- **Labels & Micro-headers**: Uppercase, tracking-widest (`text-[10px] uppercase tracking-widest font-semibold text-slate-400`).

---

## 4. Component Layout Specifications

### A. Fleet Overview Dashboard
1. **Cockpit KPI HUD Ribbon**:
   - 5 aerospace telemetry tiles: Total Active Turbofans, Grounded/Critical Units, Fleet Mean Conservative RUL ($p_{10}$), Fleet Overall Health Index, and Pending Triage Work Orders.
   - Glassmorphic backplate with laser cyan micro-edges and live pulsing status dots.
2. **Fleet Engine Matrix**:
   - High-density card grid with cockpit HUD frame (`border-hud` with corner accent styling).
   - Real-time engine health meter, conservative RUL ($p_{10}$) readout, sparkline degradation trend, and primary subsystem attribution tag.
   - Interactive severity filter selector (All, Critical, High, Medium, Healthy) with instant count badges.
3. **Avionics Survival Curve & Risk Distribution**:
   - Kaplan-Meier / Weibull fleet survival projection chart with dual lines ($p_{50}$ median vs $p_{10}$ conservative limit) over dark cockpit grid.
   - Donut risk apportionment breakdown with action protocols.

### B. Engine Detail Telemetry Panel
1. **Avionics Flight Header**:
   - Digital engine identifier, regime tag, operating cycles counter.
   - Large digital RUL indicator with uncertainty interval $[p_{10} - p_{90}]$ box.
   - Instant PDF download action button with spinner feedback.
2. **Multi-Horizon Hazard Probability Radar (H=10, 30, 60)**:
   - High-contrast visual gauges tracking probabilities of failure across 10, 30, and 60 operating cycles.
   - Clear classification threshold indication for Rule M3 trigger.
3. **Telemetry Trajectory & Uncertainty Envelope**:
   - Recharts area curve showing historical degradation trajectory, conservative $p_{10}$ bound, and nominal limits.
4. **Subsystem Degradation & Local SHAP Attribution**:
   - 5-station turbine radar diagram (Fan/LPC, HPC, Combustor, HPT, LPT/Exhaust).
   - High-contrast horizontal attribution bars with positive/negative degradation contributions.

---

## 5. CSS Utility Classes (Injected into `web/src/index.css`)

```css
/* Cockpit Void Background */
.cockpit-void {
  background: radial-gradient(ellipse at 50% 0%, rgba(14, 27, 49, 0.6) 0%, rgba(6, 11, 19, 1) 75%),
              #060b13;
}

/* Glassmorphic Cockpit Panel */
.cockpit-glass {
  background: rgba(11, 17, 32, 0.72);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(56, 189, 248, 0.16);
  box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
}

.cockpit-glass-elevated {
  background: rgba(15, 23, 42, 0.85);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(0, 240, 255, 0.3);
  box-shadow: 0 12px 36px -4px rgba(0, 0, 0, 0.7), 0 0 16px rgba(0, 240, 255, 0.1);
}

/* HUD Reticle Corners */
.hud-corner {
  position: relative;
}
.hud-corner::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 8px;
  height: 8px;
  border-top: 2px solid #00f0ff;
  border-left: 2px solid #00f0ff;
  border-top-left-radius: 4px;
}
```
