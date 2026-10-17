"""Render the frozen assessment as one offline HTML document, without loading models or fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from plotly.offline import get_plotlyjs

from src.modelling.utils.assessment import json_value, write_json


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> Any:
    return _read_json(path) if path.exists() else {}


def _return_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare all families on the same fully supported formation dates, retaining attrition."""
    frame = frame.copy()
    frame["year"] = frame.date.dt.year
    complete = frame.groupby("date").excess_return.apply(lambda values: values.notna().all())
    frame["common_complete"] = frame.date.map(complete)
    rows = []
    for (period, year, model), group in frame.groupby(["period", "year", "model"]):
        paired = group.loc[group.common_complete]
        rows.append(
            {
                "period": str(period),
                "year": int(str(year)),
                "model": str(model),
                "formation_dates": len(group),
                "complete_dates": int(group.excess_return.notna().sum()),
                "common_dates": len(paired),
                "selected_windows": int(group.selected.sum()),
                "supported_windows": int(group.supported.sum()),
                "mean_60d_stock_return": paired.cohort_return.mean(),
                "mean_60d_spy_return": paired.market_return.mean(),
                "mean_excess": paired.excess_return.mean(),
                "excess_win_fraction": (paired.excess_return > 0).mean(),
            }
        )
    return rows


def render_assessment(run: Path, config: DictConfig) -> Path:
    """Deterministic artifact-only rendering; every retained tree feature must be represented."""
    artifacts = run / "_out"
    ledger = _read_json(artifacts / "selection.json")
    evaluation = _read_json(artifacts / "eval" / "result.json")
    if not ledger["frozen"] or evaluation["frozen_at"] != ledger["frozen_at"]:
        raise ValueError("Report requires matching frozen development and sealed evaluation")
    explanations = {}
    for family in ["lgbm", "random_forest"]:
        explanations[family] = {str(fold): _read_json(artifacts / "eval" / f"{family}_{fold}" / "curves.json") for fold in range(6)}
        required = set(ledger["selected"][family]["features"])
        if any(set(item["features"]) != required for item in explanations[family].values()):
            raise ValueError(f"{family}: explanation coverage does not match retained features")
    daily = {name: pd.read_parquet(artifacts / "eval" / f"daily_{name}.parquet").to_dict("records") for name in ["development", "sealed"]}
    coverage = pd.concat([pd.read_parquet(artifacts / "eval" / f"q{fold}.parquet") for fold in range(6)], ignore_index=True)
    coverage = coverage.groupby(["year", "feature"], as_index=False)[["observed", "rows"]].sum()
    coverage["coverage"] = coverage.observed / coverage.rows
    payload = {
        "selection": ledger,
        "baseline": _read_json(artifacts / "b0" / "result.json"),
        "final": _read_json(artifacts / "final" / "result.json"),
        "evaluation": evaluation,
        "explanations": explanations,
        "daily": daily,
        "coverage": coverage.to_dict("records"),
        "market": _return_summary(pd.read_parquet(artifacts / "eval" / "matched_returns.parquet")),
        "spy_calendar_years": _read_json(artifacts / "eval" / "market_calendar_years.json"),
        "linear": _read_json(artifacts / "eval" / "linear.json"),
        "research": _optional_json(artifacts / "research.json"),
        "period_interpretation": _optional_json(artifacts / "period_interpretation.json"),
        "upstream_quality": _optional_json(artifacts / "upstream_quality.json"),
        "validation": _optional_json(artifacts / "validation.json"),
        "eligibility": {key: value for key, value in _read_json(artifacts / "eligibility.json").items() if key not in {"coverage", "date_counts"}},
        "target": str(config.model.target_type),
    }
    encoded = json.dumps(json_value(payload), separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    output = run / "report.html"
    maximum_path = max(len(str(path.resolve())) for path in artifacts.rglob("*") if path.is_file())
    if maximum_path >= 260:
        raise ValueError(f"Artifact path exceeds the portable Windows path budget: {maximum_path}")
    document = _TEMPLATE.replace("__PLOTLY__", get_plotlyjs()).replace("__PAYLOAD__", encoded)
    output.write_text(document, encoding="utf-8")
    write_json(
        artifacts / "report_manifest.json",
        {
            "report_sha256": hashlib.sha256(document.encode()).hexdigest(),
            "bytes": output.stat().st_size,
            "model_count": 4,
            "retained_tree_features": {family: sorted(value["0"]["features"]) for family, value in explanations.items()},
            "selection_frozen_at": ledger["frozen_at"],
            "max_artifact_path": maximum_path,
            "inline_assets": True,
            "sections": ["protocol", "selection", "temporal", "market", "explanation", "diagnosis", "interpretation versus research"],
        },
    )
    return output


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>h60 ensemble · sealed validation</title>
<style>
:root{color-scheme:dark;--bg:#091321;--card:#112135;--ink:#edf3fa;--muted:#a9bad0;--accent:#62d5c5;--border:#2b4058}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,Segoe UI,sans-serif}
main{max-width:1450px;margin:auto;padding:48px 32px}h1{font-size:clamp(30px,4vw,54px);line-height:1.13;letter-spacing:-1px;max-width:1000px}h2{font-size:29px;margin:0 0 18px}h3{font-size:20px}section{margin:40px 0;padding:30px;background:var(--card);border:1px solid var(--border);border-radius:18px}
.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.15em;text-transform:uppercase;font-size:12px}.sub,.note{color:var(--muted)}.note{font-size:14px}.tag{display:inline-block;padding:5px 12px;border:1px solid var(--border);border-radius:30px;margin:4px 8px 4px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}.metric{background:var(--bg);border:1px solid var(--border);padding:18px;border-radius:12px}.value{font-size:28px;font-weight:700;color:var(--accent)}
.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:end;margin:18px 0}label{display:flex;flex-direction:column;gap:5px;font-size:14px;color:var(--muted)}select{max-width:100%;padding:10px;border-radius:7px;background:var(--bg);color:var(--ink);border:1px solid #66829d;min-width:160px}select:focus,a:focus{outline:3px solid var(--accent)}.plot{min-height:420px}.wide{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--border);white-space:nowrap}th{color:var(--muted);font-weight:600}a{color:var(--accent)}.callout{padding:18px;border-left:4px solid var(--accent);background:var(--bg);margin:20px 0}.warning{border-color:#e8ba72}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}#researchText{white-space:pre-line}.nav{display:flex;gap:18px;flex-wrap:wrap}.nav a{text-decoration:none;font-size:14px}code{font-size:13px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.empty{color:#e8ba72} @media(max-width:800px){main{padding:24px 12px}section{padding:18px}.split{grid-template-columns:1fr}.plot{min-height:350px}}
</style><script>__PLOTLY__</script></head><body><main>
<div class="eyebrow">Research assessment · fixed parameters · h60 only</div>
<h1>Which signals generalize beyond the training period?</h1>
<p class="sub">Separate feature selection for LightGBM, LightGBM RF and elastic-net; a frozen cutoff ensemble evaluated from January 2022.</p>
<nav class="nav" aria-label="Report sections"><a href="#protocol">Protocol</a><a href="#selection">Selection</a><a href="#temporal">IC / IR</a><a href="#market">Market</a><a href="#explanation">Effects</a><a href="#diagnosis">Diagnosis</a><a href="#research">Interpretation versus research</a></nav>
<section id="protocol"><h2>1. What the evidence measures</h2><div id="protocolTags"></div>
<p>Models learn the magnitude-preserving cross-sectional <b>z-score target at 60 sessions</b>. IC is daily cross-sectional Spearman correlation. IC-IR is pooled mean daily IC divided by its population standard deviation, multiplied by √(252/60). It is not a portfolio return information ratio.</p>
<p class="note">The target producer defines a standardized residual after forward factor-return subtraction and cross-sectional neutralization against loadings, momentum, size and industry groups. IC measures this residual label. Actual top-decile total return versus SPY is a separate assessment.</p><div id="upstreamRisk" class="callout warning"></div>
<div class="callout">Development begins in 1999. Five forward validation windows cover 2000–02, 2003–06, 2007–12, 2013–18 and 2019–21; every used training outcome is observable by 31 December 2021. Selection uses development only. The latest period is sealed until all pools and cutoff models are frozen.</div>
<p class="note">Overlapping h60 outcomes are dependent. The intervals below use 60- and 120-session moving blocks within original folds. No target, hyperparameter, seed, ensemble-rule or new monotonic-direction search is performed. Selection performance is exploratory; sealed performance is the final assessment.</p><div id="validationStatus" class="callout warning"></div>
<details><summary>Scope, bias and practical limits</summary><p>Stored historical predictors and the current stock roster are used. This run does not certify historical S&amp;P 500 membership, delisted coverage, every vendor vintage or every producer's point-in-time contract. These limits affect investability claims. Raw/adjusted levels and share-unit controls with uncertified adjustment vintages are excluded. Conditional sector-specific missingness is retained and measured. The final cutoff model ages through the sealed period; no later retraining is hidden in its results.</p></details></section>
<section id="selection"><h2>2. Each family earns its own feature pool</h2><div id="selectionCards" class="grid"></div><p class="note">Wide baseline plus at most ten pruning rounds. Exact zero held-out importance is removed first; otherwise bottom ceil(20%). Stop after two rounds fail to beat the best pooled OOF IC-IR. Existing configured pools can win. Trees use date-balanced original-fold SHAP; linear uses standardized absolute coefficients. Undefined or missing candidate dates cannot improve the objective.</p><div id="selectionTable" class="wide"></div><details><summary>Exact frozen pools and fixed controls</summary><pre id="frozenPools"></pre></details></section>
<section id="temporal"><h2>3. IC and IC-IR through time</h2><div class="controls"><label>Population<select id="period"><option value="sealed">Sealed · 2022 onward</option><option value="development">Development · original forward folds</option></select></label><label>Rolling statistic<select id="stat"><option value="rolling_ic">252-session rolling mean IC</option><option value="rolling_ir">252-session rolling IC-IR</option><option value="ic">Daily IC</option></select></label></div><div id="metricCards" class="grid"></div><div id="icPlot" class="plot"></div><div id="yearTable" class="wide"></div><details><summary>Uncertainty, folds and prediction attrition</summary><pre id="uncertainty"></pre></details><p class="note">These time series use predictions from the complete feature-only universe, then paired finite labels. Development selection metrics above use the fixed labelled OOF population. Additional contemporaneous rows can change ensemble cross-sectional scaling, so both populations are explicitly identified. Rolling windows crossing a fold boundary include a model refresh. First-fold training is limited by the approved 1999 start.</p></section>
<section id="market"><h2>4. Highest predicted decile versus SPY</h2><p>At each formation date, select the highest predicted <b>ceil(N/10)</b> stocks for each member and the ensemble; break ties by ticker, then assign equal weights. Membership is committed before future labels or prices are attached. Match each realized 60-session total-return window with SPY's <code>equity_tr</code> on identical dates.</p><div id="marketTable" class="wide"></div><p class="note">Gross overlapping 60-session stock-picking diagnostics; no compounding into a portfolio curve, turnover or trading costs. Formation-year 2022 windows can end in 2023. All reported means compare models on common fully supported formation dates. Missing selected windows remain fixed missing weights and are disclosed. Every interior daily return must be valid and compounded returns must reconcile with endpoints.</p><div id="spyCalendar" class="note"></div></section>
<section id="explanation"><h2>5. Importance and the shape of a variable's effect</h2><div class="controls"><label>Tree family<select id="family"><option value="lgbm">LightGBM</option><option value="random_forest">RF · bagged LightGBM</option></select></label><label>Retained variable<select id="feature"></select></label></div><div id="featureInfo" class="callout"></div><div id="importancePlot" class="plot"></div><div class="split"><div><h3>Signed SHAP dependence</h3><div id="shapPlot" class="plot"></div></div><div><h3>Partial dependence · fixed cutoff model</h3><div id="pdpPlot" class="plot"></div></div></div><p class="note">SHAP allocates a model prediction among inputs; it is not a causal effect. PDP replaces one variable across a sampled background while keeping other inputs at observed values, then averages predicted scores. Correlated inputs may create unsupported combinations; sector-specific variables can create unrealistic all-sector interventions. Categorical grids use observed valid codes, never fractional categories. Scatter is a deterministic display sample; full sampled SHAP vectors, keys and base values are retained locally. Fold curves show original held-out models; no later model is retrospectively fitted for 2019.</p><details><summary>Linear member · signed standardized coefficients</summary><div id="linearTable" class="wide"></div></details></section>
<section id="diagnosis"><h2>6. Weak periods: coverage, cohort changes or model weakness?</h2><div id="diagnosisText"></div><div id="qualityPlot" class="plot"></div><div id="qualityTable" class="wide"></div><p class="note">Coverage is measured on the full scoring universe. Sparse sector-specific features can have valid conditional support. Coverage shifts or SHAP changes are associations; they do not by themselves prove a source failure. Confirmed data-defect claims require named dates, affected inputs and source evidence. A model can perform poorly without a demonstrated data defect. Current-roster coverage and model age remain explicit alternative explanations.</p></section>
<section id="research"><div class="eyebrow">Final interpretation step</div><h2>7. Interpretation versus research</h2><p>For the variable selected above: is there a measured relationship, is its direction stable across the five original folds, does the economics support it, and would a future monotonic experiment be defensible?</p><div id="researchText" class="callout"></div><div id="researchSources"></div><p class="note">An existing imposed direction can force a PDP shape; it is not independent evidence that the data learned that direction. A SHAP slope alone cannot prove a monotonic constraint violation. New constraint proposals are future research only: predeclare them, rerun purged development comparisons and retain a fresh untouched evaluation period. No constraint is added from sealed outcomes in this run.</p></section>
<p class="note">Offline report · all chart code and data embedded · source outcomes and models remain local · evidence files under _out.</p>
</main><script id="assessment-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('assessment-data').textContent),names=['elasticnet','lgbm','random_forest','ensemble'];
const labels={elasticnet:'Linear',lgbm:'LightGBM',random_forest:'RF',ensemble:'Ensemble'}, colors={elasticnet:'#e6b86a',lgbm:'#62d5c5',random_forest:'#aa9cf7',ensemble:'#f28b99'};
const $=id=>document.getElementById(id),fmt=(x,n=3)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x).toFixed(n),pct=x=>x===null||x===undefined?'—':fmt(x*100,2)+'%';
function text(id,value){$(id).textContent=value}
function table(id,headers,rows){const el=$(id);el.replaceChildren();const t=document.createElement('table'),head=t.createTHead().insertRow();headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.append(th)});const body=t.createTBody();rows.forEach(row=>{const tr=body.insertRow();row.forEach(v=>{tr.insertCell().textContent=v??'—'})});el.append(t)}
function plot(id,traces,title,ytitle,extra={}){Plotly.react(id,traces,{paper_bgcolor:'#112135',plot_bgcolor:'#112135',font:{color:'#c9d7e6'},title:{text:title,font:{size:16}},margin:{t:60,l:70,r:25,b:70},legend:{orientation:'h'},xaxis:{gridcolor:'#2b4058',automargin:true},yaxis:{title:ytitle,gridcolor:'#2b4058',zerolinecolor:'#66829d',automargin:true},...extra},{responsive:true,displaylogo:false})}
function initialize(){
 const tags=[D.eligibility.rows.toLocaleString()+' development rows',D.selection.features.length+' eligible predictors','Outcomes known by '+D.selection.outcome_cutoff,'Sealed formation end '+D.evaluation.formation_end];tags.forEach(v=>{const s=document.createElement('span');s.className='tag';s.textContent=v;$('protocolTags').append(s)});
 text('validationStatus',D.validation.verdict?'Independent validation: '+D.validation.verdict+' · '+(D.validation.summary||'See diagnostic evidence.'):'Independent validation is pending; model and report completeness are not yet a readiness verdict.');
 text('upstreamRisk',(D.upstream_quality.paragraphs||['Historical vintage assessment has not been recorded. A time split alone does not certify point-in-time inputs or an investable historical universe.']).join('\n'));
 const rows=[];names.filter(n=>n!=='ensemble').forEach(n=>{const f=D.selection.families[n],b=f.best;const c=document.createElement('div');c.className='metric';c.innerHTML='<div></div><div class="value"></div><div class="note"></div>';c.children[0].textContent=labels[n];c.children[1].textContent=b.features.length+' inputs';c.children[2].textContent='Winner '+b.id+' · OOF IC-IR '+fmt(b.metrics.ic_ir);$('selectionCards').append(c);rows.push([labels[n],'configured baseline',D.baseline.features[n].length,fmt(D.baseline.metrics[n].mean_ic),fmt(D.baseline.metrics[n].ic_ir),String(D.baseline.metrics[n].eligible)]);f.candidates.forEach(r=>rows.push([labels[n],r.id+(r.id===b.id?' · winner':''),r.features.length,fmt(r.metrics.mean_ic),fmt(r.metrics.ic_ir),String(r.metrics.eligible)]))});
 table('selectionTable',['Model','Candidate','Features','Pooled IC','Pooled IC-IR','Complete objective'],rows);text('frozenPools',JSON.stringify({frozen_at:D.selection.frozen_at,selected:D.selection.selected,controls:D.selection.controls},null,2));
 table('marketTable',['Period','Year','Model','Common / all dates','Supported / selected windows','Mean stock h60','Mean SPY h60','Mean excess','Excess wins'],D.market.map(r=>[r.period,r.year,labels[r.model],r.common_dates+' / '+r.formation_dates,r.supported_windows+' / '+r.selected_windows,pct(r.mean_60d_stock_return),pct(r.mean_60d_spy_return),pct(r.mean_excess),pct(r.excess_win_fraction)]));
 text('spyCalendar','Separate calendar-year SPY total-return context: 2022 '+pct(D.spy_calendar_years['2022'])+'; 2023 '+pct(D.spy_calendar_years['2023'])+'. These calendar returns have a different horizon from the matched h60 averages.');
 const L=D.linear;table('linearTable',['Variable','Signed coefficient / training SD','Training mean','Training SD'],L.features.map((f,i)=>[f,L.coefficient[i],L.mean[i],L.std[i]]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).map(r=>[r[0],fmt(r[1],6),fmt(r[2],6),fmt(r[3],6)]));
 const interpretation=D.period_interpretation;const passages=interpretation.paragraphs||['Period attribution has not been reviewed yet. The tables identify weak windows; they do not establish a source defect or a causal explanation.'];passages.forEach(v=>{const p=document.createElement('p');p.textContent=v;$('diagnosisText').append(p)});
 const quality={};D.coverage.forEach(r=>{if(!quality[r.feature])quality[r.feature]={};quality[r.feature][r.year]=r.coverage});const changes=Object.entries(quality).filter(([f,v])=>v['2018']!==undefined&&v['2019']!==undefined).map(([f,v])=>[f,v['2018'],v['2019'],v['2019']-v['2018']]).sort((a,b)=>a[3]-b[3]);table('qualityTable',['Largest 2019 coverage contractions','2018','2019','Change'],changes.slice(0,20).map(r=>[r[0],pct(r[1]),pct(r[2]),pct(r[3])]));
 const yrs=[...new Set(D.coverage.map(r=>r.year))].sort(),fs=changes.slice(0,20).map(r=>r[0]);plot('qualityPlot',[{type:'heatmap',x:yrs,y:fs,z:fs.map(f=>yrs.map(y=>quality[f][y]??null)),zmin:0,zmax:1,colorscale:'Viridis',colorbar:{title:'Support'}}],'Coverage of variables with largest 2019 contractions','',{height:650,margin:{t:60,l:330,r:50,b:60}});
 $('period').addEventListener('change',temporal);$('stat').addEventListener('change',temporal);$('family').addEventListener('change',features);$('feature').addEventListener('change',feature);temporal();features();document.documentElement.dataset.reportReady='true';
}
function temporal(){const p=$('period').value,s=$('stat').value,metrics=D.evaluation.metrics[p];$('metricCards').replaceChildren();names.forEach(n=>{const m=metrics[n],el=document.createElement('div');el.className='metric';el.innerHTML='<div></div><div class="value"></div><div class="note"></div>';el.children[0].textContent=labels[n];el.children[1].textContent=fmt(m.mean_ic)+' IC';el.children[2].textContent='IC-IR '+fmt(m.ic_ir)+' · '+m.n_days+' dates';$('metricCards').append(el)});plot('icPlot',names.map(n=>{const a=D.daily[p].filter(r=>r.model===n);return {x:a.map(r=>r.date),y:a.map(r=>r[s]),name:labels[n],mode:'lines',line:{color:colors[n],width:n==='ensemble'?2.5:1.4}}}),p==='sealed'?'Frozen cutoff model · sealed evaluation':'Original fold models · development',s==='rolling_ir'?'IC-IR':'Spearman IC');const rows=[];names.forEach(n=>Object.entries(metrics[n].years).forEach(([yr,m])=>rows.push([yr,labels[n],fmt(m.mean_ic),fmt(m.ic_ir),m.n_days])));rows.sort((a,b)=>Number(a[0])-Number(b[0]));table('yearTable',['Formation year','Model','Mean IC','IC-IR','Paired dates'],rows);text('uncertainty',JSON.stringify(metrics,null,2))}
function features(){const f=$('family').value,values=D.explanations[f]['0'].features,prior=$('feature').value;$('feature').replaceChildren();Object.keys(values).sort((a,b)=>values[b].importance-values[a].importance).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;$('feature').append(o)});if(values[prior])$('feature').value=prior;feature()}
function feature(){const family=$('family').value,name=$('feature').value,E=D.explanations[family],F=E['0'].features[name],all=E['0'].features,ranked=Object.keys(all).sort((a,b)=>all[b].importance-all[a].importance),top=ranked.slice(0,25).reverse();
 text('featureInfo',name+' · final mean |SHAP| '+fmt(F.importance,6)+' · observed '+F.observed+'/'+F.sample_rows+' sampled rows · '+(F.categorical?'unordered category':('existing imposed direction '+F.constraint))+' · additivity max error '+E['0'].additivity_max_error.toExponential(2));
 plot('importancePlot',[{type:'bar',orientation:'h',y:top,x:top.map(n=>all[n].importance),marker:{color:top.map(n=>n===name?'#f28b99':'#62d5c5')}}],'Final holdout-sample importance · top 25 (all retained variables available above)','Variable',{height:650,xaxis:{title:'Date-weighted mean absolute SHAP'},margin:{l:350,t:60,r:30,b:60}});
 const traces=[{x:F.scatter.x,y:F.scatter.shap,type:'scattergl',mode:'markers',name:'Sealed sample',marker:{color:'#62d5c5',opacity:.35,size:5}},{x:F.binned.x,y:F.binned.mean,mode:'lines+markers',name:'Sealed binned mean',line:{color:'#f28b99',width:3}}];for(let fold=1;fold<=5;fold++){const a=E[String(fold)].features[name];traces.push({x:a.binned.x,y:a.binned.mean,mode:'lines',name:'Original fold '+fold,visible:'legendonly'})}plot('shapPlot',traces,'Signed contribution versus observed variable','Contribution to raw h60 z-score prediction');
 plot('pdpPlot',[{x:F.pdp.x||[],y:F.pdp.mean||[],mode:'lines+markers',line:{color:'#aa9cf7',width:3},name:'PDP'}],'Observed-domain PDP · '+F.pdp.trend.shape,'Average raw h60 z-score prediction');
 const folds=[1,2,3,4,5].map(i=>{const a=E[String(i)].features[name];return 'Fold '+i+': '+a.shap_trend.shape+'; rho '+fmt(a.shap_trend.rho)+'; observed '+a.observed+'/'+a.sample_rows});
 const R=D.research[name]||{},constraint=F.categorical?'Categorical codes are unordered; an ordinal monotonic constraint is inappropriate.':F.constraint?'The current direction '+F.constraint+' is imposed. A directional PDP therefore cannot independently validate that economic prior.':'No direction is currently imposed. A stable PDP/SHAP association needs economic and horizon-specific support before a future constrained experiment.';
 text('researchText',name+'\nMeasured final PDP: '+F.pdp.trend.shape+'.\n'+folds.join('\n')+'\n\n'+constraint+'\n\n'+(R.interpretation||'No feature-specific research conclusion has been entered. No new constraint is prescribed from this sealed sample.')+'\n'+(R.future_constraint||'Future protocol: predeclare a direction, compare against this unchanged model on purged development folds, and reserve a fresh untouched evaluation period.'));
 $('researchSources').replaceChildren();const sources=R.sources||[];if(!sources.length){const p=document.createElement('p');p.className='empty';p.textContent='Feature-specific primary research support: not established in this report.';$('researchSources').append(p)}sources.forEach(s=>{const p=document.createElement('p'),a=document.createElement('a');a.textContent=s.title;a.href=s.url;a.target='_blank';a.rel='noopener noreferrer';p.append(a);if(s.limit)p.append(document.createTextNode(' · '+s.limit));$('researchSources').append(p)})
}
initialize();
</script></body></html>"""
