"""
Build the fictional sample catalogue shipped with GiftxAI.

Evergreen & Co. does not exist; every product and price is made up so the demo
has something real to retrieve from. Run from the repo root:

    pip install reportlab
    python scripts/make_sample_catalogue.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "public" / "sample" / "evergreen-holiday-catalogue.pdf"

SECTIONS: list[tuple[str, list[tuple[str, float, str]]]] = [
    ("Kitchen & Home", [
        ("Copper Espresso Machine", 649.99, "15-bar pump, steam wand, brushed copper body. Two-year warranty."),
        ("Cast Iron Dutch Oven 6 qt", 189.00, "Enamelled cast iron in forest green. Oven safe to 260 C."),
        ("Smart Kettle Pro", 129.50, "Five temperature presets, keep-warm for 60 minutes, app control."),
        ("Olive Wood Serving Board", 54.00, "Hand-finished olive wood, 45 cm, with juice groove."),
        ("Holiday Spice Collection", 32.00, "Twelve jars: cinnamon, clove, star anise, nutmeg and more."),
        ("Linen Table Runner", 38.00, "Stonewashed linen, 220 cm, in winter berry red."),
        ("Ceramic Mug Set of 4", 46.00, "Speckled glaze, 350 ml, dishwasher safe."),
    ]),
    ("Tech & Gadgets", [
        ("Noise-Cancelling Headphones X9", 349.00, "40-hour battery, adaptive noise cancelling, USB-C."),
        ("E-Reader Paperlight 7", 189.99, "7-inch glare-free screen, warm light, waterproof IPX8."),
        ("Instant Film Camera Mini", 89.00, "Prints credit-card size photos in 90 seconds. Includes 10 shots."),
        ("Smart Watch Active 3", 279.00, "GPS, heart rate, sleep tracking, seven-day battery."),
        ("Portable Projector Lumen", 429.00, "1080p, 300 ANSI lumens, built-in speaker, 2.5-hour battery."),
        ("Wireless Charging Pad Duo", 59.00, "Charges a phone and earbuds at once, 15 W."),
        ("Bluetooth Speaker Pebble", 79.00, "Waterproof, 12-hour playtime, pairs in stereo."),
    ]),
    ("Toys & Games", [
        ("Wooden Train Set Deluxe", 119.00, "64 pieces of beech wood track, bridge and three engines. Ages 3+."),
        ("STEM Robot Builder Kit", 149.00, "Build five robots, code them from a tablet. Ages 8+."),
        ("Plush Reindeer Large", 42.00, "60 cm soft plush with embroidered eyes. Ages 0+."),
        ("Family Board Game Night Bundle", 74.00, "Three strategy games for 2 to 6 players."),
        ("1000-Piece Winter Village Puzzle", 24.00, "Recycled board, poster included."),
        ("Remote Control Snow Truck", 68.00, "All-terrain tyres, 25-minute run time, rechargeable."),
    ]),
    ("Wellness & Self-Care", [
        ("Cashmere Throw Blanket", 295.00, "100% Mongolian cashmere, 130 x 180 cm, oatmeal."),
        ("Aromatherapy Diffuser Stone", 64.00, "Ultrasonic diffuser with three oils: pine, cedar, orange."),
        ("Spa Gift Set Evergreen", 58.00, "Bath salts, body oil, candle and a cotton robe."),
        ("Weighted Blanket 7 kg", 139.00, "Glass bead fill, cooling bamboo cover."),
        ("Yoga Mat Natural Rubber", 88.00, "5 mm, non-slip, with carry strap."),
        ("Soy Candle Trio", 36.00, "Fir, clove and vanilla. 40-hour burn each."),
    ]),
    ("Fashion & Accessories", [
        ("Leather Weekender Bag", 389.00, "Full-grain leather, brass hardware, fits a 15-inch laptop."),
        ("Merino Wool Scarf", 68.00, "Extra-fine merino, 180 cm, charcoal or burgundy."),
        ("Silver Snowflake Pendant", 145.00, "Sterling silver on an 18-inch chain, gift boxed."),
        ("Touchscreen Gloves Leather", 52.00, "Lambskin with cashmere lining."),
        ("Minimalist Wallet Slim", 45.00, "RFID blocking, holds eight cards."),
    ]),
    ("Gift Sets & Bundles", [
        ("Premium Holiday Hamper", 249.00, "Includes: Copper Espresso Machine sampler beans, Holiday Spice Collection, Soy Candle Trio, Olive Wood Serving Board and a bottle of sparkling cider."),
        ("Cosy Night In Bundle", 179.00, "Includes: Weighted Blanket 7 kg, Ceramic Mug Set of 4 and Soy Candle Trio."),
        ("Little Explorer Bundle", 99.00, "Includes: Plush Reindeer Large, 1000-Piece Winter Village Puzzle and a picture book."),
        ("Stocking Stuffer Pack", 29.00, "Includes: lip balm, hand warmer pair, chocolate coins and a mini puzzle."),
    ]),
]

INK = HexColor("#1C1714")
MUTED = HexColor("#6B6159")
ACCENT = HexColor("#E0563F")
GOLD = HexColor("#B8862E")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("Evergreen & Co. Holiday Gift Catalogue (fictional sample)")
    W, H = A4
    m = 56

    # cover
    c.setFillColor(HexColor("#FBF6EF"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, H - 14, W, 14, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 34)
    c.drawString(m, H - 200, "Evergreen & Co.")
    c.setFont("Helvetica", 18)
    c.setFillColor(ACCENT)
    c.drawString(m, H - 232, "Holiday Gift Catalogue 2026")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 11)
    y = H - 290
    for line in (
        "Christmas presents for everyone on your list: kitchen, tech, toys,",
        "wellness, fashion and ready-made gift sets.",
        "",
        "Free gift wrapping on every order over $75.",
        "Order by December 18 for delivery before Christmas.",
        "",
        "This catalogue is fictional sample data for the GiftxAI demo.",
    ):
        c.drawString(m, y, line)
        y -= 17
    total = sum(len(items) for _, items in SECTIONS)
    c.setFillColor(GOLD)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(m, 90, f"{total} gifts across {len(SECTIONS)} departments")
    c.showPage()

    for title, items in SECTIONS:
        y = H - m
        c.setFillColor(ACCENT)
        c.rect(0, H - 8, W, 8, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(m, y - 10, title)
        y -= 46
        for name, price, desc in items:
            if y < 110:
                c.showPage()
                y = H - m
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 12.5)
            c.drawString(m, y, name)
            c.setFillColor(ACCENT)
            c.drawRightString(W - m, y, f"${price:,.2f}")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 10)
            words, line, yy = desc.split(), "", y - 16
            for w in words:
                trial = (line + " " + w).strip()
                if c.stringWidth(trial, "Helvetica", 10) > W - 2 * m - 90:
                    c.drawString(m, yy, line)
                    yy -= 13
                    line = w
                else:
                    line = trial
            c.drawString(m, yy, line)
            c.setStrokeColor(HexColor("#E8DFD3"))
            c.line(m, yy - 12, W - m, yy - 12)
            y = yy - 34
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8.5)
        c.drawString(m, 40, "Evergreen & Co. · fictional sample catalogue · prices in USD")
        c.showPage()
    c.save()
    print(f"wrote {OUT} with {total} products")


if __name__ == "__main__":
    main()
