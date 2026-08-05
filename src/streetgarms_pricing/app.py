"""Street Garms pricing app (Streamlit)."""
import base64
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from streetgarms_pricing.features.build import TIME_COL, prepare
from streetgarms_pricing.models.train import MODEL_PATH

DATASET = "data/interim/sales_combined.csv"
CONDITIONS = ["Fantastic", "Great", "Like New", "Brand New", "Unknown"]
GENDERS = ["Mens", "Womens"]
PRICING_PLATFORM = "shopify"   # channel the model prices for (a model input that moves the number)
THIN = 5                       # fewer brand+type comps than this -> manual-review state
BAR_HEADROOM = 1.25            # range-bar axis max = hi * this
COMPS_SHOWN = 8
# confidence tiers, richest first: (min comps, dot colour, label, copy)
TIERS = (
    (20, "#3fae72", "High confidence", "Plenty of similar sales to lean on."),
    (THIN, "#f0c341", "Low confidence", "Some similar sales — treat as guidance."),
)


# ---------- cached data / model ----------
@st.cache_resource
def get_model():
    """Load the persisted bundle -> (fitted pipeline, 90% conformal band factor)."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["conformal_factor"]


@st.cache_data
def get_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, format="mixed")
    return prepare(df)


def comparables(data, brand, ptype) -> pd.DataFrame:
    mask = (data["brand"] == brand) & (data["product_type"] == ptype)
    cols = [TIME_COL, "sold_price", "product_name", "platform"]
    return data.loc[mask, cols].sort_values(TIME_COL, ascending=False)


def comp_name(brand: str, product_name) -> str:
    """Show 'Brand Product Name', but don't double the brand when the name (Vinted /
    Depop titles) already leads with it."""
    name = str(product_name)
    return name if name.lower().startswith(str(brand).lower()) else f"{brand} {name}"


def logo_tag(height: int = 44) -> str:
    """Inline the logo as a data URI so it sits inside the custom HTML header.
    (The browser decodes AVIF; st.image can't be embedded in st.markdown.)"""
    p = Path("assets/logosg.avif")
    if not p.exists():
        return '<div class="sg-logo"><span>Street Garms</span></div>'   # fallback: text logo
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/avif;base64,{b64}" alt="Street Garms" style="height:{height}px;display:block;">'


# ---------- styling ----------
st.set_page_config(page_title="Street Garms — Price Estimator", page_icon="🏷️", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,400;0,600;0,700;0,800;0,900;1,800;1,900&display=swap');
.stApp { background:#f4f4f2; }
html, body, [class*="css"], .stMarkdown, input, select, button { font-family:'Archivo','Helvetica Neue',sans-serif; }
#MainMenu, header[data-testid="stHeader"], footer { visibility:hidden; }
.block-container { padding-top:1rem; max-width:900px; }
h1.sg { font-weight:900; font-style:italic; text-transform:uppercase; font-size:40px; line-height:1; margin:.2rem 0 .3rem; }
h2.sg { font-weight:900; font-style:italic; text-transform:uppercase; font-size:22px; letter-spacing:.02em; margin:1rem 0 .4rem; }
.sg-logo { background:#0c0c0c; color:#fff; font-weight:900; font-style:italic; font-size:22px; padding:8px 16px; transform:skewX(-8deg); text-transform:uppercase; display:inline-block; }
.sg-logo span { display:inline-block; transform:skewX(8deg); }
.stButton>button, div[data-testid="stFormSubmitButton"] button {
  background:#2e6b4b !important; color:#fff !important; border:none !important; border-radius:0 !important;
  font-weight:700; font-size:14px; letter-spacing:.1em; text-transform:uppercase; padding:.9rem 2.8rem !important;
}
.stButton>button:hover, div[data-testid="stFormSubmitButton"] button:hover { background:#1f4c34 !important; color:#fff !important; }
/* form fields: readable + solid white / bordered so they stand out on the cream page */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { font-size:16px; color:#111; }
[data-baseweb="input"], [data-baseweb="base-input"],
div[data-baseweb="select"] > div {
  background:#fff !important; border:1px solid #c9c9c3 !important; border-radius:0 !important;
}
[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within { border-color:#2e6b4b !important; }
ul[role="listbox"], div[data-baseweb="menu"], div[data-baseweb="popover"] { background:#fff !important; }
ul[role="listbox"] li, div[data-baseweb="menu"] li, div[data-baseweb="popover"] li { font-size:15px; color:#111; background:#fff; }
label p { font-size:12px !important; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#55554f; }
/* the form itself = a white card (contains title / brand / type / size / colour / condition) */
div[data-testid="stForm"] { background:#fff !important; border:1px solid #e3e3e0 !important; border-radius:0 !important; }
</style>""", unsafe_allow_html=True)

data = get_data()

st.markdown(f"""
<div style="background:#2e6b4b;color:#fff;text-align:center;font-size:12px;font-weight:600;
letter-spacing:.08em;text-transform:uppercase;padding:8px;margin:-1rem -1rem 1rem;">Internal staff tool — not customer-facing</div>
<div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;">
  {logo_tag()}
  <div style="display:flex;flex-direction:column;">
    <span style="font-weight:800;font-size:14px;letter-spacing:.14em;text-transform:uppercase;">Resale Price Estimator</span>
  </div>
</div>
<h1 class="sg">Price an item</h1>
<p style="color:#55554f;margin-top:0;">Enter what you know about the item. The estimate is built from comparable
sales across streetgarms.com, Depop, Vinted and eBay.</p>
""", unsafe_allow_html=True)

# ---------- form ----------
brands = sorted(data["brand"].dropna().unique())
ptypes = sorted(data["product_type"].dropna().unique())
with st.form("item"):
    title = st.text_input("Item title", "Stone Island Nylon Metal Watro Jacket")
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", brands, index=brands.index("Stone Island") if "Stone Island" in brands else 0)
    ptype = c2.selectbox("Product type", ptypes, index=ptypes.index("jacket") if "jacket" in ptypes else 0,
                         format_func=str.capitalize)
    c3, c4, c5, c6 = st.columns(4)
    size = c3.text_input("Size", "L")
    colour = c4.text_input("Colour", "Black")
    condition = c5.selectbox("Condition", CONDITIONS)
    gender = c6.selectbox("Gender", GENDERS)
    go = st.form_submit_button("Estimate price")

# ---------- results ----------
if go:
    comps = comparables(data, brand, ptype)
    n = len(comps)

    if n < THIN:
        st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e0;padding:28px;margin-top:8px;">
          <div style="display:inline-block;background:#0c0c0c;color:#f0c341;font-weight:900;font-style:italic;
          font-size:17px;text-transform:uppercase;padding:8px 16px;transform:skewX(-8deg);"><span style="display:inline-block;transform:skewX(8deg);">Not enough comps to price</span></div>
          <p style="color:#55554f;margin:14px 0 0;max-width:58ch;">Only {n} comparable {html.escape(brand)} {html.escape(ptype.capitalize())} sales in the data, so any number would
          be a guess. Price this one manually — or widen the search by trying a broader product type.</p>
        </div>""", unsafe_allow_html=True)
    else:
        pipeline, factor = get_model()
        row = prepare(pd.DataFrame([{
            "brand": brand, "product_type": ptype, "size": size, "colour": colour,
            "gender": gender, "condition_grade": condition, "platform": PRICING_PLATFORM,
            "product_name": title,
        }]))
        price = float(pipeline.predict(row)[0])
        lo, hi = price / factor, price * factor          # factor = exp(90% log-residual quantile)
        median = comps["sold_price"].median()
        bar_max = hi * BAR_HEADROOM
        dot, clabel, ccopy = next((t[1:] for t in TIERS if n >= t[0]), TIERS[-1][1:])

        def bar_pct(v):
            return f"{v / bar_max * 100:.1f}%"

        st.markdown(f"""<div style="background:#0c0c0c;color:#fff;padding:30px;margin-top:8px;">
          <div style="display:flex;justify-content:space-between;gap:36px;flex-wrap:wrap;">
            <div>
              <div style="font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#8a8a86;">Estimated price</div>
              <div style="display:flex;align-items:baseline;gap:12px;"><span style="font-weight:900;font-style:italic;font-size:70px;line-height:.9;">£{price:.0f}</span><span style="font-size:12px;text-transform:uppercase;color:#8a8a86;">list price</span></div>
              <div style="font-size:13px;color:#b9b9b2;margin-top:6px;">90% confidence range <b style="color:#fff;">£{lo:.0f} – £{hi:.0f}</b></div>
            </div>
            <div style="min-width:230px;flex:1;">
              <div style="display:flex;align-items:center;gap:8px;"><span style="width:9px;height:9px;border-radius:50%;background:{dot};display:inline-block;"></span><span style="font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:{dot};">{clabel}</span></div>
              <p style="font-size:13px;color:#b9b9b2;margin:6px 0;">{ccopy}</p>
              <div style="height:6px;background:#2a2a27;position:relative;margin-top:10px;">
                <div style="position:absolute;top:0;bottom:0;left:{bar_pct(lo)};width:{(hi - lo) / bar_max * 100:.1f}%;background:#3fae72;"></div>
                <div style="position:absolute;top:-4px;bottom:-4px;left:{bar_pct(price)};width:2px;background:#fff;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:11px;color:#8a8a86;margin-top:4px;"><span>£0</span><span>£{bar_max:.0f}</span></div>
            </div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#e3e3e0;border:1px solid #e3e3e0;">
          <div style="background:#fff;padding:16px 20px;"><div style="font-size:10.5px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#8a8a86;">Comparable sales</div><div style="font-weight:800;font-size:22px;">{n}</div></div>
          <div style="background:#fff;padding:16px 20px;"><div style="font-size:10.5px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#8a8a86;">Median comp</div><div style="font-weight:800;font-size:22px;">£{median:.0f}</div></div>
        </div>""", unsafe_allow_html=True)

        rows_html = "".join(
            '<div style="display:grid;grid-template-columns:110px 80px 1fr 90px;padding:12px 20px;'
            'border-bottom:1px solid #f0f0ee;font-size:14px;align-items:center;">'
            f'<span style="font-family:monospace;font-size:13px;color:#55554f;">{r[TIME_COL].date()}</span>'
            f'<span style="font-weight:800;">£{r["sold_price"]:.0f}</span>'
            f'<span>{html.escape(comp_name(brand, r["product_name"]))}</span>'
            f'<span style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#55554f;">{r["platform"]}</span></div>'
            for _, r in comps.head(COMPS_SHOWN).iterrows())
        st.markdown('<h2 class="sg">Comparable recent sales</h2>', unsafe_allow_html=True)
        st.markdown(f"""<div style="background:#fff;border:1px solid #e3e3e0;">
          <div style="display:grid;grid-template-columns:110px 80px 1fr 90px;background:#fbfbfa;border-bottom:1px solid #e3e3e0;padding:12px 20px;font-size:10.5px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#8a8a86;"><span>Sold</span><span>Price</span><span>Item</span><span>Platform</span></div>
          {rows_html}</div>""", unsafe_allow_html=True)
        st.caption("Estimates are guidance, not a fixed price — condition, provenance and season all move the number.")
