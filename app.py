# -*- coding: utf-8 -*-
"""
Hydrotech Toos Predictor — Streamlit app (RTL)

▸ دو حالت آغاز کار:
  • استفاده از دیتای پیش‌فرض (۱۳۹۴-۱۳۹۷)
  • ورود دیتای دلخواه کاربر (ویزارد: انتخاب سال‌ها → جدول)

▸ متن‌های فارسی راست‌چین.

▸ نمایش لوگوی «logo.png»
  • صفحه‌های «شروع» و «ورود داده‌ها»: بالا ـ سمت چپ (مطلق، با Base64 تا لود شود)
  • صفحهٔ تحلیل: داخل سایدبار

▸ پس از تأیید داده، جداول «خام / فیچرها / نرمال»، پیش‌بینی و RMSE عین نسخهٔ اصلی است.
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import base64, os


# --------------------------------------------------
# Helper: default dataset
# --------------------------------------------------
def load_default_dataset() -> pd.DataFrame:
    data = {
        "year": [1394, 1395, 1396, 1397],
        "y":    [151.4, 152.6, 152.9, 150.1],
        "x1":   [4.5, 4.5, 4.5, 3.8],
        "x3":   [20.4, 19.5, 20.7, 19.9],
        "x4":   [71.2, 78.6, 67.5, 181.3],
        "ccc":  [3027, 3358, 3356, 3416],
        "c":    [248492065, 285549812, 328134000, 377068789],
        "x2":   [1.22e-5, 1.18e-5, 1.02e-5, 9.06e-6],
    }
    return (
        pd.DataFrame(data)
        .drop(columns=["ccc", "c"])
        .set_index("year")
    )


# --------------------------------------------------
# Session helpers
# --------------------------------------------------
def safe_rerun():
    (st.experimental_rerun if hasattr(st, "experimental_rerun") else st.rerun)()


def init_state():
    defaults = {
        "step": "start",      # start | input_data | run_app
        "year_list": [],
        "user_df": None,
        "edit_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
st.set_page_config(page_title="Hydrotech Toos Predictor", layout="wide")


# --------------------------------------------------
# Global RTL style + logo helpers
# --------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
    }
    #logo-fixed {
        position:absolute;
        top:12px;
        left:12px;
        z-index:100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def inline_logo(path: str = "logo.png", width: int = 210) -> str:
    """Return <img> tag with base64-encoded logo if file exists."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"<img id='logo-fixed' src='data:image/png;base64,{b64}' width='{width}'>"


# show logo on the two first pages
if st.session_state.step in {"start", "input_data"}:
    st.markdown(inline_logo(), unsafe_allow_html=True)


# ==================================================
# STEP 0 — Start / choose mode
# ==================================================
if st.session_state.step == "start":
    st.title("🔰 آغاز کار")

    if st.button("🚀 شروع با دیتای پیش‌فرض"):
        st.session_state.user_df = load_default_dataset()
        st.session_state.step = "run_app"
        safe_rerun()

    st.markdown("---")
    st.subheader("یا دیتای خود را وارد کنید ↓")
    c1, c2 = st.columns(2)
    sy = c1.number_input("سال شروع", 1300, 1500, 1394, 1, format="%i")
    ey = c2.number_input("سال پایان", 1300, 1500, 1397, 1, format="%i")

    if sy > ey:
        st.error("سال شروع باید کوچکتر یا مساوی سال پایان باشد.")
    if st.button("مرحلهٔ بعد", disabled=sy > ey):
        st.session_state.year_list = list(range(int(sy), int(ey) + 1))
        st.session_state.step = "input_data"
        safe_rerun()

# ==================================================
# STEP 1 — Data-entry table
# ==================================================
elif st.session_state.step == "input_data":
    st.title("📥 ورود داده‌ها — جدول")

    years = st.session_state.year_list
    header_map = {
        "مصرف (y)":                   "y",
        "هدر رفت واقعی (x1)":         "x1",
        "نسبت قیمت آب به هزینه خانوار (x2)": "x2",
        "دمای میانگین (x3)":          "x3",
        "بارش سالانه (x4)":           "x4",
    }
    disp_cols = list(header_map.keys())

    if st.session_state.edit_df is None:
        st.session_state.edit_df = (
            pd.DataFrame(index=years, columns=disp_cols, dtype=float)
            .rename_axis("سال")
        )

    edited = st.data_editor(
        st.session_state.edit_df,
        num_rows="fixed",
        hide_index=False,
        use_container_width=True,
        key="data_editor",
    )
    st.session_state.edit_df = edited

    col_save, col_back = st.columns(2)

    with col_save:
        if st.button("ذخیره و ادامه ➡️"):
            if edited.isnull().any().any():
                st.error("دیتا کامل وارد نشده است؛ لطفاً تمامی سلول‌ها را پر کنید.")
            else:
                try:
                    st.session_state.user_df = (
                        edited.rename(columns=header_map).astype(float)
                    )
                    st.session_state.step = "run_app"
                    safe_rerun()
                except Exception as exc:
                    st.error(f"خطا در پردازش داده‌ها: {exc}")
                    st.session_state.edit_df = None
                    safe_rerun()

    with col_back:
        if st.button("🔙 بازگشت"):
            st.session_state.step = "start"
            st.session_state.edit_df = None
            safe_rerun()

# ==================================================
# STEP 2 — Analysis & UI
# ==================================================
elif st.session_state.step == "run_app":
    df_raw = st.session_state.user_df

    # ---------- preprocessing ----------
    def preprocess(df_in: pd.DataFrame):
        df = df_in.copy()
        for col in df.columns:
            df[f"{col}^2"]   = df[col] ** 2
            df[f"log_{col}"] = np.log(df[col])
            df[f"1/{col}"]   = 1 / df[col]
        norm = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            norm[col] = df[col] / df[col].mean()
        return df, norm

    def select_best(df):
        targets = ["y", "1/y", "log_y", "y^2"]
        bases   = ["x1", "x2", "x3", "x4"]
        best_map, scores = {}, {}
        for b in bases:
            cand = [b, f"{b}^2", f"log_{b}", f"1/{b}"]
            vals = {c: np.mean([abs(df[c].corr(df[t])) for t in targets]) for c in cand}
            best = max(vals, key=vals.get)
            best_map[b], scores[b] = best, vals[best]
        return best_map, scores

    def t_apply(v, t):
        return v ** 2 if t.endswith("^2") else (
            np.log(v) if t.startswith("log_") else (1 / v if t.startswith("1/") else v)
        )

    def t_revert(v, tg):
        if tg == "y":      return v
        if tg == "1/y":    return np.inf if v == 0 else 1 / v
        if tg == "log_y":  return np.exp(v)
        if tg == "y^2":    return np.nan if v < 0 else np.sqrt(v)
        return v

    def predict(df0, mapping, inp, tg, mdl):
        raw   = [t_apply(inp[b], mapping[b]) for b in mapping]
        means = [df0[mapping[b]].mean() for b in mapping]
        norm  = [raw[i] / means[i] for i in range(len(raw))]
        pnorm = mdl.predict(pd.DataFrame([norm], columns=list(mapping.values())))[0]
        return pnorm * df0[tg].mean()

    def rmse_all(df0, mapping, mdls):
        out = {}
        for t, m in mdls.items():
            preds = [
                t_revert(
                    predict(df0, mapping,
                            {b: r[b] for b in ["x1", "x2", "x3", "x4"]},
                            t, m),
                    t
                )
                for _, r in df0.iterrows()
            ]
            out[t] = np.sqrt(((df0["y"] - np.array(preds)) ** 2).mean())
        return out

    feature_df, norm_df = preprocess(df_raw)
    best_map, corr = select_best(feature_df)
    sel_feats = list(best_map.values())
    models = {
        t: LinearRegression().fit(norm_df[sel_feats], norm_df[t])
        for t in ["y", "1/y", "log_y", "y^2"]
    }
    rmse = rmse_all(feature_df, best_map, models)

    # ---------- UI ----------
    st.sidebar.image("logo.png", width=210)   # logo inside sidebar

    st.title("📊 پیش بینی سرانه مصرف ")

    st.sidebar.header("🔢 ورودی مدل")
    inputs = {
        "x1": st.sidebar.number_input("x1: هدررفت واقعی", 0.0, value=float(feature_df["x1"].mean())),
        "x2": st.sidebar.number_input("x2: نسبت قیمت آب به هزینه خانوار", 1e-9, value=float(feature_df["x2"].mean()), format="%.8f"),
        "x3": st.sidebar.number_input("x3: دمای میانگین", value=float(feature_df["x3"].mean())),
        "x4": st.sidebar.number_input("x4: بارش سالانه", value=float(feature_df["x4"].mean())),
    }
    target = st.sidebar.selectbox("🎯 متغیر هدف", ["y", "1/y", "log_y", "y^2"])
    show_corr = st.sidebar.checkbox("📈 نمایش نقشهٔ همبستگی")

    # --- datasets view
    st.markdown("---")
    st.subheader("📑 داده‌ها")
    for lbl, view in zip(
        ["📄 خام", "⚙️ فیچرها", "🔧 نرمال"],
        [feature_df[["y", "x1", "x2", "x3", "x4"]], feature_df, norm_df],
    ):
        with st.expander(lbl):
            st.dataframe(view, use_container_width=True)

    # --- feature summary
    st.markdown("---")
    st.subheader("🔍 انتخاب فیچرها و قدرت همبستگی")
    desc = {
        "x1": "هدر رفت واقعی",
        "x2": "نسبت قیمت آب به هزینه خانوار",
        "x3": "دمای میانگین",
        "x4": "بارش سالانه",
    }
    st.table(
        pd.DataFrame(
            {
                "توضیح": desc,
                "تبدیل منتخب": best_map,
                "میانگین |ρ|": {b: f"{corr[b]:.3f}" for b in best_map},
            }
        ).T
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(corr.keys(), corr.values())
    ax.set_ylabel("Mean |ρ|")
    ax.set_xlabel("Feature")
    st.pyplot(fig)

    # --- optional heatmap
    if show_corr:
        st.markdown("---")
        st.subheader("📊 نقشهٔ همبستگی (فیچرهای کامل)")
        tg = ["y", "1/y", "log_y", "y^2"]
        others = [c for c in feature_df.columns if c not in tg]
        cm = pd.DataFrame({t: feature_df[others].corrwith(feature_df[t]) for t in tg}).T
        fig_h, ax_h = plt.subplots(figsize=(10, 4))
        sns.heatmap(cm, annot=True, cmap="coolwarm", ax=ax_h)
        st.pyplot(fig_h)

    # --- prediction
    pred_norm = predict(feature_df, best_map, inputs, target, models[target])
    pred_y = t_revert(pred_norm, target)

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("پیش‌بینی y", f"{pred_y:.4f}")
    c2.metric(f"پیش‌بینی {target}", f"{pred_norm:.4f}")

    st.markdown("#### ⚖️ دقت مدل‌ها (RMSE در مقیاس y)")
    st.table(pd.DataFrame(rmse, index=["RMSE"]).T)

    coef = models[target].coef_
    st.code(
        f"{target}_norm = "
        + " + ".join(f"{coef[i]:+.4f}*{sel_feats[i]}" for i in range(len(coef)))
        + f" {models[target].intercept_:+.4f}",
        language="python",
    )

    if st.button("🔄 بازگشت به صفحهٔ شروع"):
        st.session_state.step = "start"
        safe_rerun()
