"""
Maintenance Recommendation Report Generator (PDF)
Produces evidence-based, decision-support documentation from model outputs:
- RUL quantiles (p10, p50, p90)
- Top telemetry sensors
- Rule decision trace
- Model version
Includes mandatory "FOR DECISION SUPPORT ONLY" disclaimer.
"""

import io
import time
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


def generate_maintenance_report_pdf(engine_id: str, data: Dict[str, Any]) -> bytes:
    """
    Generates a clean PDF Maintenance Recommendation Report using ReportLab.
    Accepts engine/order data containing model outputs and rule traces.
    Returns bytes of the generated PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569")
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#991b1b")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    code_style = ParagraphStyle(
        "CodeTrace",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    elements = []

    # 1. Header
    elements.append(Paragraph("AETURNUS INDUSTRIAL INTELLIGENCE", subtitle_style))
    elements.append(Paragraph("Maintenance Recommendation Report", title_style))
    elements.append(Spacer(1, 6))

    # 2. Disclaimer Callout Box
    disclaimer_html = (
        "<b>DECISION SUPPORT ONLY:</b> This report is generated automatically from statistical "
        "machine learning models and predictive rules. It is provided strictly for maintenance planning "
        "and engineering decision support. It does not constitute a certified maintenance release, "
        "airworthiness directive, or regulatory signoff."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_html, disclaimer_style)]],
        colWidths=[540]
    )
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(disclaimer_table)
    elements.append(Spacer(1, 10))

    # Metadata extraction
    work_order_id = data.get("work_order_id", f"WO-{engine_id}-AUTO")
    cycle = data.get("current_cycle", data.get("cycle", data.get("created_at_cycle", 1)))
    severity = str(data.get("severity", "MEDIUM")).upper()
    model_version = str(data.get("model_version", "GBDT-v2.1"))
    rules_version = str(data.get("rules_version", "2026-10-07.1"))
    reported_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    pred_rul = data.get("predicted_rul") or {}
    p10_value = pred_rul.get("p10", data.get("rul_p10"))
    p50_value = pred_rul.get("p50", data.get("rul_p50"))
    p90_value = pred_rul.get("p90", data.get("rul_p90"))
    conservative_value = data.get("conservative_rul")
    censored_at_cap = bool(
        data.get("censored_at_cap", pred_rul.get("censored_at_cap", p90_value is None))
    )
    if p10_value is None or p50_value is None or conservative_value is None:
        raise ValueError(
            "Maintenance report requires RUL p10/p50 and conservative_rul; "
            "numeric report defaults are not allowed."
        )
    if not censored_at_cap and p90_value is None:
        raise ValueError(
            "Maintenance report requires a finite p90 for closed intervals or an "
            "explicit censored_at_cap flag for open intervals."
        )
    p10 = float(p10_value)
    p50 = float(p50_value)
    p90 = None if censored_at_cap else float(p90_value)
    conservative_rul = float(conservative_value)

    # 3. Target Asset & Report Summary Table
    meta_data = [
        [
            Paragraph("<b>Target Asset:</b>", body_style),
            Paragraph(str(engine_id), body_style),
            Paragraph("<b>Work Order:</b>", body_style),
            Paragraph(str(work_order_id), body_style)
        ],
        [
            Paragraph("<b>Current Cycle:</b>", body_style),
            Paragraph(str(cycle), body_style),
            Paragraph("<b>Severity Status:</b>", body_style),
            Paragraph(f"<font color='{_severity_color(severity)}'><b>{severity}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Model Version:</b>", body_style),
            Paragraph(model_version, body_style),
            Paragraph("<b>Rules Version:</b>", body_style),
            Paragraph(rules_version, body_style)
        ],
        [
            Paragraph("<b>Generated:</b>", body_style),
            Paragraph(reported_time, body_style),
            Paragraph("<b>Order Status:</b>", body_style),
            Paragraph(str(data.get("status", "ACTIVE")), body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[100, 170, 100, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 12))

    # 4. Model Predictions & Remaining Useful Life (Quantiles)
    elements.append(Paragraph("1. Remaining Useful Life (RUL) Predictions", section_heading))
    rul_table_data = [
        [
            Paragraph("<b>Quantile Metric</b>", body_style),
            Paragraph("<b>Estimated Cycles</b>", body_style),
            Paragraph("<b>Interpretation / Operational Context</b>", body_style)
        ],
        [
            Paragraph("Conservative RUL (Planning)", body_style),
            Paragraph(f"<b>{conservative_rul:.1f}</b>", body_style),
            Paragraph("Governing threshold for maintenance scheduling and dispatch alerts", body_style)
        ],
        [
            Paragraph("Quantile p10 (Lower Bound)", body_style),
            Paragraph(f"{p10:.1f}", body_style),
            Paragraph("90% confidence that true residual life exceeds this duration", body_style)
        ],
        [
            Paragraph("Quantile p50 (Median)", body_style),
            Paragraph(f"{p50:.1f}", body_style),
            Paragraph("Expected median cycles to functional degradation threshold", body_style)
        ],
        [
            Paragraph("Quantile p90 (Upper Bound)", body_style),
            Paragraph("Open (> p10)" if censored_at_cap else f"{p90:.1f}", body_style),
            Paragraph(
                "Censored at the RUL cap; no finite upper bound is reported"
                if censored_at_cap
                else "10% probability that remaining life exceeds this duration",
                body_style,
            )
        ],
        [
            Paragraph("80% Empirical Interval", body_style),
            Paragraph(
                f"> {p10:.1f} cycles (upper open)"
                if censored_at_cap
                else f"[{p10:.1f} - {p90:.1f}]",
                body_style,
            ),
            Paragraph(
                "Interval breadth is unbounded above because the calibrated upper bound reaches the RUL cap."
                if censored_at_cap
                else f"Interval breadth: {(p90 - p10):.1f} cycles (coverage calibrated ~80%)",
                body_style,
            )
        ]
    ]
    rul_table = Table(rul_table_data, colWidths=[150, 110, 280])
    rul_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(rul_table)
    elements.append(Spacer(1, 12))

    # 5. Top Contributing Sensors
    elements.append(Paragraph("2. Top Contributing Telemetry Sensors", section_heading))
    top_sensors = data.get("top_sensors", ["s2", "s3", "s4"])
    sensor_rows = [
        [
            Paragraph("<b>Rank</b>", body_style),
            Paragraph("<b>Sensor Channel</b>", body_style),
            Paragraph("<b>Subsystem / Description</b>", body_style)
        ]
    ]
    sensor_map = {
        "s2": "Total temperature at LPC outlet (Low Pressure Compressor)",
        "s3": "Total temperature at HPC outlet (High Pressure Compressor)",
        "s4": "Total temperature at LPT outlet (Low Pressure Turbine)",
        "s7": "Total pressure at HPC outlet (Combustion chamber inlet)",
        "s8": "Physical fan speed (Core rotational dynamics)",
        "s9": "Physical core speed (HPC/HPT spool velocity)",
        "s11": "Static pressure at HPC outlet",
        "s12": "Ratio of fuel flow to static HPC outlet pressure",
        "s13": "Corrected fan speed",
        "s14": "Corrected core speed",
        "s15": "Bypass ratio (Cooling & propulsive flow balance)",
        "s17": "Bleed enthalpy / Bleed flow rate",
        "s20": "High-pressure turbine coolant bleed",
        "s21": "Low-pressure turbine coolant bleed"
    }
    for idx, s in enumerate(top_sensors[:5], start=1):
        desc = sensor_map.get(s, "Telemetry channel degradation signature")
        sensor_rows.append([
            Paragraph(f"#{idx}", body_style),
            Paragraph(f"<b>{s}</b>", body_style),
            Paragraph(desc, body_style)
        ])
    
    sensor_table = Table(sensor_rows, colWidths=[50, 110, 380])
    sensor_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(sensor_table)
    elements.append(Spacer(1, 12))

    # 6. Rule Decision Trace
    elements.append(Paragraph("3. Model & Rule Evaluation Trace", section_heading))
    trace_items = data.get("decision_trace", [])
    if not trace_items:
        trace_items = ["rule_engine:evaluation_completed", f"severity_classified:{severity}"]
    
    trace_paragraphs = []
    for item in trace_items:
        trace_paragraphs.append([Paragraph(f"&bull; <code>{item}</code>", code_style)])
    
    trace_table = Table(trace_paragraphs, colWidths=[540])
    trace_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(trace_table)
    elements.append(Spacer(1, 12))

    # 7. Recommended Action Box
    elements.append(Paragraph("4. Recommended Maintenance Action", section_heading))
    rec_action = data.get(
        "recommended_action",
        f"Schedule standard inspection for suspected {data.get('suspected_subsystem', 'HPC')} degradation."
    )
    rec_table = Table([[Paragraph(f"<b>Action:</b> {rec_action}", body_style)]], colWidths=[540])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#3b82f6")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(rec_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def _severity_color(sev: str) -> str:
    mapping = {
        "CRITICAL": "#dc2626",
        "HIGH": "#ea580c",
        "MEDIUM": "#d97706",
        "LOW": "#2563eb",
        "HEALTHY": "#16a34a"
    }
    return mapping.get(sev, "#0f172a")
