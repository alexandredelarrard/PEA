"""Render saved multi-horizon IC and variable effects as one self-contained offline report."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from plotly.offline import get_plotlyjs

from src.modelling.utils.artifacts import file_identity
from src.modelling.utils.assessment import json_value, write_json
from src.utils.modelling_runs import read_run


def _payload(run: Path, assessment_directory: Path | None = None) -> dict[str, Any]:
    training = read_run(run)
    assessment_directory = assessment_directory or run / "assessment"
    state = json.loads((assessment_directory / "metadata.json").read_text(encoding="utf-8"))
    if state["run_id"] != training["run_id"]:
        raise ValueError("Report and training run identities differ")
    analysis = {}
    for horizon in training["horizons"]:
        output = assessment_directory / f"h{horizon}"

        daily = pd.read_parquet(output / "daily.parquet")
        temporal = {}
        for population in ("validation", "backtest"):
            temporal[population] = {}
            for model in [*training["model_types"], "ensemble"]:
                rows = daily.loc[(daily.population == population) & (daily.model == model)].sort_values("date")
                if rows.empty or rows.date.duplicated().any():
                    raise ValueError(f"h{horizon}: incomplete or duplicate temporal chart series")
                temporal[population][model] = {name: rows[name].tolist() for name in ("date", "ic", "rolling_ic", "rolling_ir", "paired", "universe")}
        effects = {name: {} for name in training["model_types"]}
        windows = state["windows"][str(horizon)]
        if set(windows) != {"0", *(str(fold) for fold in training["folds"][str(horizon)])}:
            raise ValueError("Report requires all original folds and cutoff effects")
        for fold in windows:
            folder = output / f"fold{fold}"
            for family in training["model_types"]:
                curves = json.loads((folder / family / "curves.json").read_text(encoding="utf-8"))
                if not curves.get("features"):
                    raise ValueError("A required model has no retained variable effects")
                for values in curves["features"].values():
                    if fold == "0":
                        values["scatter"] = {key: values["scatter"][key] for key in ("x", "shap", "raw_x") if key in values["scatter"]}
                    else:
                        values.pop("scatter", None)
                effects[family][fold] = curves
        for family, folds in effects.items():
            retained = set(folds["0"]["features"])
            if any(set(curves["features"]) != retained for curves in folds.values()):
                raise ValueError(f"h{horizon} {family}: original-fold effect feature inventory differs")
        analysis[str(horizon)] = {"temporal": temporal, "effects": effects}
    return json_value(
        {
            "run_id": training["run_id"],
            "horizons": training["horizons"],
            "families": training["model_types"],
            "training_as_of": training["training_as_of"],
            "frontier": state["observation_frontier"],
            "trained_at_utc": training["training_completed_at_utc"],
            "assessed_at_utc": state["assessment_completed_at_utc"],
            "effective_fold_test_ends": {h: evidence["effective_fold_test_end"] for h, evidence in state["horizons"].items()},
            "analysis": analysis,
        }
    )


def _document(payload: dict[str, Any]) -> str:
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saved-model assessment</title><style>
:root{color-scheme:light;--ink:#183042;--muted:#586a78;--line:#d9e3e8;--accent:#007f86}
*{box-sizing:border-box}body{margin:0;background:#f2f6f8;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1480px;margin:auto;padding:32px}header{padding:12px 4px 20px}h1{font-size:30px;letter-spacing:-.7px;margin:4px 0}
h2{font-size:23px;margin:0 0 18px}h3{font-size:17px;margin:20px 0 10px}.eyebrow{color:var(--accent);font-size:12px;font-weight:750;letter-spacing:1.6px;text-transform:uppercase}
section{background:white;border:1px solid var(--line);border-radius:12px;padding:25px;margin:16px 0}
.controls{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0 14px}label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:650;color:var(--muted)}
select{font:14px system-ui;color:var(--ink);background:white;border:1px solid #b8cbd4;border-radius:5px;padding:9px;min-width:125px;max-width:100%}
label.wide{flex:1;min-width:220px}.note{color:var(--muted);font-size:13px}.chart{height:480px}.importance-scroll{max-height:500px;overflow-y:auto;border:1px solid #e7eef2;border-radius:6px}
#report-status{min-height:22px;color:var(--muted)}@media(max-width:700px){main{padding:12px}section{padding:15px}.chart{height:400px}h1{font-size:24px}}
</style></head><body><main>
<header><div class="eyebrow">Saved original models · fixed feature pools</div><h1>Model assessment</h1><p id="scope" class="note"></p><div id="report-status" role="status" aria-live="polite">Loading saved charts…</div></header>
<section id="time-section"><h2>IC and IC-IR through time</h2><div class="controls">
<label for="horizon-time">Horizon<select id="horizon-time"></select></label>
<label for="period">Sample<select id="period"><option value="validation">Fold tests · original models</option><option value="backtest">Post-2022 assessment · cutoff model</option></select></label>
<label for="stat">Statistic<select id="stat"><option value="ic">Daily IC</option><option value="rolling_ic">Smoothed IC</option><option value="rolling_ir">Smoothed IC-IR</option></select></label>
</div><div id="temporal" class="chart"></div><p class="note">IC is daily finite-pair Spearman correlation. IC-IR = mean(IC) / population std(IC) × √(252 / horizon). Smoothing requires the full configured number of observed sessions; missing or constant statistics remain gaps. IC-IR is a model statistic.</p></section>
<section id="effects-section"><h2>Importance and the shape of a variable’s effect</h2>
<div class="controls"><label for="horizon-effects">Horizon<select id="horizon-effects"></select></label><label for="family">Model<select id="family"></select></label></div>
<h3>Fixed cutoff model importance</h3><p id="importance-note" class="note"></p><div class="importance-scroll"><div id="importance"></div></div>
<div class="controls"><label for="feature" class="wide">Variable<select id="feature"></select></label><label for="effect-kind">Effect chart<select id="effect-kind"><option value="signed">Signed SHAP dependence</option><option value="pdp">Partial dependence · fixed cutoff model</option></select></label></div>
<div id="effect" class="chart"></div><p id="effect-note" class="note"></p>
<p class="note">PDP replaces one variable while keeping the other sampled inputs fixed. Correlated-variable combinations may be unsupported. These are model responses, with conditional historical interpretation because source vintages and classifications are retrospective.</p></section>
</main><script id="assessment-data" type="application/json">__PAYLOAD__</script>
<script>__PLOTLY__</script><script>
"use strict";
const D=JSON.parse(document.getElementById("assessment-data").textContent),$=id=>document.getElementById(id);
const colors=["#007f86","#cc6c32","#7163a8","#162e43","#779957","#b44e77"],names={elasticnet:"Linear (elastic net)",ridge:"Linear (ridge)",lgbm:"LightGBM",random_forest:"Random forest",ensemble:"Ensemble"};
const settings={responsive:true,displaylogo:false,toImageButtonOptions:{format:"png",scale:2}};
function options(node,values,label){const old=node.value;node.replaceChildren(...values.map(value=>{const option=document.createElement("option");option.value=value;option.textContent=label(value);return option;}));if(values.includes(old))node.value=old;}
function layout(title,y){return {title:{text:title,font:{size:16}},font:{family:"system-ui,sans-serif",color:"#183042"},paper_bgcolor:"white",plot_bgcolor:"white",margin:{l:80,r:25,t:65,b:65},legend:{orientation:"h",y:-.2},xaxis:{showgrid:false},yaxis:{title:{text:y},zerolinecolor:"#aabdc8"},hovermode:"closest"};}
function selected(){const h=$("horizon-effects").value,m=$("family").value;return {h,m,folds:D.analysis[h].effects[m],linear:["elasticnet","ridge"].includes(m)};}
async function temporal(){const h=$("horizon-time").value,p=$("period").value,s=$("stat").value,series=D.analysis[h].temporal[p];const traces=Object.entries(series).map(([m,v],i)=>({x:v.date,y:v[s],name:names[m]||m,type:"scatter",mode:v[s].filter(Number.isFinite).length===1?"markers":"lines",connectgaps:false,line:{color:colors[i],width:m==="ensemble"?2.4:1.4},customdata:v.paired.map((n,j)=>[n,v.universe[j]]),hovertemplate:"%{x|%Y-%m-%d}<br>%{y:.4f}<br>finite pairs %{customdata[0]} / %{customdata[1]}<extra>"+(names[m]||m)+"</extra>"}));return Plotly.react("temporal",traces,layout("h"+h+" · "+(p==="validation"?"Fold tests":"Post-2022 assessment"),s==="rolling_ir"?"IC-IR":"IC"),settings);}
async function importance(){const {h,m,folds,linear}=selected(),features=folds["0"].features;const sorted=Object.keys(features).sort((a,b)=>features[b].importance-features[a].importance||a.localeCompare(b));options($("feature"),sorted,x=>x);$("effect-kind").options[0].textContent=linear?"Signed linear contribution":"Signed SHAP dependence";$("importance-note").textContent=linear?"Absolute saved coefficient on training-standardized inputs. Hover shows signed beta. Zero coefficients remain selectable.":"Mean absolute SHAP on the deterministic cutoff backtest sample; all retained variables remain selectable.";const view=layout((names[m]||m)+" · h"+h, "");view.height=Math.max(360,sorted.length*23+80);view.margin={l:Math.min(420,Math.max(170,Math.max(...sorted.map(x=>x.length))*6)),r:30,t:50,b:45};view.xaxis={title:{text:linear?"|standardized coefficient|":"Mean |SHAP|"}};view.yaxis={autorange:"reversed"};return Plotly.react("importance",[{x:sorted.map(x=>features[x].importance),y:sorted,type:"bar",orientation:"h",marker:{color:"#007f86"},customdata:linear?sorted.map(x=>features[x].coefficient):undefined,hovertemplate:linear?"%{y}<br>|beta| %{x:.6g}<br>Saved beta %{customdata:+.6g}<extra></extra>":"%{y}<br>%{x:.6g}<extra></extra>"}],view,settings);}
async function effect(){const {h,m,folds,linear}=selected(),feature=$("feature").value,kind=$("effect-kind").value,F=folds["0"].features[feature],traces=[];let unit=linear?"Signed linear contribution":"Signed SHAP";if(kind==="pdp"){unit="Mean raw model prediction";traces.push({x:F.pdp.x||[],y:F.pdp.mean||[],name:"Fixed cutoff model",mode:"lines+markers",line:{color:colors[0]}});}else{const scatter=F.scatter||{};traces.push({x:scatter.x||[],y:scatter.shap||[],name:"Cutoff sample",mode:"markers",showlegend:false,marker:{size:4,color:"#7f939e",opacity:.3},customdata:linear?(scatter.raw_x||[]).map(value=>[value,F.coefficient]):undefined,hovertemplate:linear?"Training-standardized input %{x:.6g}<br>Raw input %{customdata[0]:.6g}<br>Saved beta %{customdata[1]:+.6g}<br>Contribution %{y:.6g}<extra>Cutoff sample</extra>":undefined});for(const fold of Object.keys(folds).sort((a,b)=>Number(a)-Number(b))){const values=folds[fold].features[feature],curve=values.binned;traces.push({x:curve.x,y:curve.mean,name:fold==="0"?"Cutoff backtest":"Fold "+fold,mode:"lines+markers",line:{color:colors[Number(fold)],width:fold==="0"?3:1.5,dash:fold==="0"?"solid":"dot"},marker:{size:5},customdata:linear?(curve.raw_x||[]).map(value=>[value,values.coefficient]):undefined,hovertemplate:linear?"Training-standardized median %{x:.6g}<br>Raw input median %{customdata[0]:.6g}<br>Saved beta %{customdata[1]:+.6g}<br>Mean contribution %{y:.6g}<extra>%{fullData.name}</extra>":undefined});}}const view=layout(feature+" · "+(names[m]||m)+" · h"+h,unit);view.xaxis={title:{text:feature+(linear?(kind==="pdp"?" (raw input)":" (training-standardized)"):(F.categorical?" (category code)":""))},type:F.categorical?"category":"linear"};if(!traces.some(t=>(t.x||[]).length))view.annotations=[{text:"No finite observed states in this sample",xref:"paper",yref:"paper",x:.5,y:.5,showarrow:false}];$("effect-note").textContent=kind==="pdp"?"Partial dependence uses the saved fixed cutoff model and observed feature grid, in raw prediction units.":linear?"Signed linear contribution = saved coefficient × (value − training mean) / training std. Each curve uses its own saved training scaler. Missing inputs contribute zero and have no observed x/bin. Hover retains raw inputs and signed beta. Curves show the original configured folds and cutoff assessment.":"Signed SHAP in raw member-prediction units. Lines show mean signed SHAP by observed state or quantile bin for each original fold and cutoff backtest.";return Plotly.react("effect",traces,view,settings);}
async function effects(){await importance();await effect();}
function listen(ids,fn){for(const id of ids)$(id).addEventListener("change",()=>fn().catch(error=>{$("report-status").textContent=error.message;window.__reportError=error.message;}));}
async function start(){options($("horizon-time"),D.horizons.map(String),h=>"h"+h+" · sessions");options($("horizon-effects"),D.horizons.map(String),h=>"h"+h+" · sessions");options($("family"),D.families,m=>names[m]||m);if(D.families.includes("lgbm"))$("family").value="lgbm";$("scope").textContent="Run "+D.run_id+" · outcomes known by "+D.training_as_of+" · observed calendar through "+D.frontier+". Trained at "+D.trained_at_utc+" · assessed at "+D.assessed_at_utc+" · effective fold ends "+JSON.stringify(D.effective_fold_test_ends)+". Fold tests also selected LightGBM rounds by RMSE. No fitting in assessment.";listen(["horizon-time","period","stat"],temporal);listen(["horizon-effects","family"],effects);listen(["feature","effect-kind"],effect);await Promise.all([temporal(),effects()]);$("report-status").textContent="Saved charts ready · works offline";document.documentElement.dataset.ready="true";}
start().catch(error=>{$("report-status").textContent=error.message;window.__reportError=error.message;});
</script></body></html>"""
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return document.replace("__PAYLOAD__", encoded).replace("__PLOTLY__", get_plotlyjs())


def render_assessment(run: Path, config: DictConfig, *, assessment_directory: Path | None = None, report_path: Path | None = None) -> Path:
    """Read only saved evidence; no database, fitted-object loading or provider is needed."""
    assessment_directory = assessment_directory or run / "assessment"
    payload = _payload(run, assessment_directory)
    document = _document(payload)
    output = report_path or run / "report.html"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run, prefix=".report.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    write_json(
        assessment_directory / "report_manifest.json",
        {
            "report_sha256": file_identity(output)["sha256"],
            "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "horizons": payload["horizons"],
            "families": payload["families"],
            "plot_areas": ["temporal", "importance", "effect"],
        },
    )
    return output
