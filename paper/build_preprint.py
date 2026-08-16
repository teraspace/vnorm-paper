from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "vnorm_preprint.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="PaperTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=19,
    leading=23,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#17324D"),
    spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="Subtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#506070"),
    spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    textColor=colors.HexColor("#17324D"),
    spaceBefore=12,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="BodyPaper",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=13.5,
    alignment=TA_LEFT,
    spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="SmallPaper",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=8,
    leading=10,
    textColor=colors.HexColor("#506070"),
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Callout",
    parent=styles["BodyText"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#17324D"),
    backColor=colors.HexColor("#EAF3F7"),
    borderColor=colors.HexColor("#A8C7D5"),
    borderWidth=0.6,
    borderPadding=8,
    spaceBefore=6,
    spaceAfter=10,
))


def P(text, style="BodyPaper"):
    return Paragraph(text, styles[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E0E6"))
    canvas.line(0.7 * inch, 0.55 * inch, 7.8 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6A7782"))
    canvas.drawString(0.7 * inch, 0.35 * inch, "VNorm - preliminary preprint")
    canvas.drawRightString(7.8 * inch, 0.35 * inch, f"{doc.page}")
    canvas.restoreState()


story = []
story.append(Spacer(1, 0.35 * inch))
story.append(P("VNorm: A BatchNorm-Compatible Parametric Activation Function", "PaperTitle"))
story.append(P("Preliminary research release", "Subtitle"))
story.append(P("Carlos M. Patino Machado", "Subtitle"))
story.append(P("Independent Researcher | Tetraspace LLC", "Subtitle"))
story.append(P(
    "<b>Abstract.</b> We introduce VNorm, a parametric activation function designed to be used after BatchNorm. "
    "VNorm combines a learnable feature-wise threshold and denominator with a learnable negative-path coefficient. "
    "In a controlled CIFAR-100 multilayer perceptron benchmark with BatchNorm, VNorm is compared with ReLU, CELU, "
    "SiLU, and PReLU across three random seeds. VNorm obtains the highest mean test accuracy among the evaluated "
    "activations in this preliminary study. The result motivates a focused view of VNorm as a BatchNorm-compatible "
    "activation rather than as a universal replacement for every normalization strategy.",
))
story.append(P(
    "<b>Keywords:</b> activation functions; BatchNorm; CIFAR-100; multilayer perceptron; reproducible machine learning",
    "SmallPaper",
))

story.append(P("1. Introduction", "Section"))
story.append(P(
    "Activation functions determine how neural networks transform intermediate representations. ReLU remains a "
    "strong baseline because of its simplicity, while smooth and parametric alternatives can provide different "
    "optimization behavior. This work studies a deliberately narrow question: can a learnable activation complement "
    "the standardized representations produced by BatchNorm?"
))
story.append(P(
    "VNorm is proposed as a component in the sequence BatchNorm -> VNorm. The method is not presented as a claim "
    "that normalization and activation should always be fused, nor as a universal winner across all architectures. "
    "The initial release focuses on a reproducible CIFAR-100 benchmark and makes the operating regime explicit."
))
story.append(P(
    "The main contributions of this preliminary release are: (1) a compact parametric activation definition; "
    "(2) a reproducible benchmark against common activation functions; and (3) an empirical observation that VNorm "
    "is particularly competitive when placed after BatchNorm."
))

story.append(P("2. VNorm", "Section"))
story.append(P(
    "For an input feature x and channel parameters gamma, D, and alpha, VNorm is defined as:"
))
story.append(P(
    "u = (x - gamma) / (abs(D) + epsilon)<br/>"
    "g = sigmoid(u)<br/>"
    "VNorm(x) = x * g + alpha * x * (1 - g)",
    "Callout",
))
story.append(P(
    "The threshold gamma and denominator D are learned per feature. The current benchmark uses a scalar alpha, "
    "initialized to 0.01. The absolute value in the denominator guarantees a positive scale up to the small epsilon "
    "stabilizer. The public implementation is intentionally small and uses standard PyTorch autograd."
))
story.append(P(
    "The recommended placement is after BatchNorm, where the input distribution has already been standardized by the "
    "preceding layer. This placement is part of the method's intended operating regime and is evaluated explicitly "
    "in the benchmark."
))

story.append(P("3. Experimental Setup", "Section"))
story.append(P(
    "We use CIFAR-100 with 45,000 training examples, 5,000 validation examples sampled from the original training "
    "set, and the standard 10,000-example test set. Images are normalized using the CIFAR-100 channel statistics."
))
story.append(P(
    "The model is a two-hidden-layer MLP. Each hidden layer uses a linear transformation followed by BatchNorm1d "
    "and the evaluated activation. The output layer has 100 classes. All activations use the same optimizer, "
    "weight decay, cosine learning-rate schedule, batch size, initialization protocol, data split, and 30-epoch budget."
))
story.append(P(
    "The benchmark evaluates three seeds: 1, 7, and 42. The best validation checkpoint from each run is evaluated "
    "once on the test set. The public repository contains the implementation and the benchmark configuration."
))

story.append(P("4. Results", "Section"))
table_data = [
    [P("Activation", "SmallPaper"), P("Mean test accuracy", "SmallPaper"), P("Std. dev.", "SmallPaper"), P("Parameters", "SmallPaper")],
    [P("VNorm"), P("26.31%"), P("0.83"), P("833,189")],
    [P("SiLU"), P("25.85%"), P("0.20"), P("833,508")],
    [P("ReLU"), P("25.57%"), P("0.45"), P("833,508")],
    [P("PReLU"), P("25.32%"), P("0.49"), P("833,510")],
    [P("CELU"), P("25.18%"), P("0.18"), P("833,508")],
]
table = Table(table_data, colWidths=[1.55 * inch, 1.7 * inch, 1.2 * inch, 1.35 * inch])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF3F7")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C6CF")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 0.12 * inch))
story.append(P(
    "VNorm obtains the highest mean test accuracy in this benchmark: 26.31% +/- 0.83%. It exceeds ReLU by 0.74 "
    "percentage points and SiLU by 0.46 percentage points. The parameter count is not a capacity advantage: VNorm "
    "has slightly fewer trainable parameters than the non-parametric baselines in this configuration."
))
story.append(P(
    "VNorm reaches its best validation checkpoint early in the three observed runs, at epochs 7, 8, and 7. This "
    "early peak is useful evidence about optimization behavior but also indicates that longer training without "
    "regularization or early stopping can overfit."
))

story.append(PageBreak())
story.append(P("5. Discussion", "Section"))
story.append(P(
    "The results support a focused empirical claim: VNorm is competitive with standard activations and performs best "
    "in the evaluated BatchNorm -> activation configuration. The result should not be interpreted as evidence that "
    "VNorm dominates every activation, architecture, dataset, or normalization strategy."
))
story.append(P(
    "The observed interaction suggests that BatchNorm and VNorm may play complementary roles. BatchNorm controls the "
    "distribution entering the activation, while VNorm provides learned feature-wise thresholding, scaling, and a "
    "learned negative path. Establishing causality for this interaction requires a factorial ablation with and without "
    "BatchNorm, which is reserved for the next revision."
))

story.append(P("6. Limitations and Next Experiments", "Section"))
story.append(P(
    "This preliminary release has several limitations. It evaluates one dataset and one MLP family. The benchmark "
    "uses three seeds, which is appropriate for an initial signal but does not establish universal statistical claims. "
    "The current results also use the two-hidden-layer benchmark; a one-hidden-layer confirmation is planned."
))
story.append(P(
    "The next experiments are intentionally limited: (1) repeat the same protocol with one hidden layer; (2) run a "
    "small BatchNorm ablation; and (3) evaluate the same public activation in the existing compact CNN. No unpublished "
    "theory or private extension is required for those experiments."
))

story.append(P("7. Reproducibility and Availability", "Section"))
story.append(P(
    "The public repository contains the VNorm v1 implementation, the CIFAR-100 benchmark, dependency information, "
    "and this preliminary paper. Results are written to CSV by the benchmark script. The code is designed to run on "
    "CPU or CUDA environments with PyTorch and torchvision installed."
))
story.append(P(
    "This first release intentionally excludes private research directions and unpublished extensions. Those materials "
    "are outside the scope of this paper."
))

story.append(P("8. Conclusion", "Section"))
story.append(P(
    "We presented VNorm, a compact parametric activation designed for use after BatchNorm. In a controlled CIFAR-100 "
    "MLP benchmark, VNorm achieved the highest mean test accuracy among five evaluated activations. The result is an "
    "initial empirical basis for studying VNorm as a BatchNorm-compatible activation, with further work needed to "
    "characterize the interaction and its generality."
))

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=letter,
    rightMargin=0.7 * inch,
    leftMargin=0.7 * inch,
    topMargin=0.65 * inch,
    bottomMargin=0.75 * inch,
    title="VNorm: A BatchNorm-Compatible Parametric Activation Function",
    author="Carlos M. Patino Machado",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
