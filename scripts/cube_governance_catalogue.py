"""
cube_governance_catalogue.py  (scripts/)
--------------------------------------------------------------------------------------------
The PROSE half of `cube_part_governance`, in the same shape `cube_feature_catalogue.py` holds
for `cube_part_fundamentals`: characteristic -> (family, what, why, tail).

⚠ WHY THIS IS A SECOND DICT RATHER THAN MORE ENTRIES IN `CATALOGUE`. The fundamentals
generator asserts BOTH directions against ONE table and exits non-zero on either failure
(`cube_fundamentals_evidence.py`: `return 1 if (missing or unused) else 0`). Adding 85
governance names to the dict it reads would put 85 entries in its `unused` list and fail every
future fundamentals run. Both directions are only assertable PER PART, so the catalogue is
per part and `cube_feature_catalogue.CATALOGUES` is the registry that maps one to the other.

⚠ KEYED ON THE CHARACTERISTIC -- the column minus its `f_` prefix and minus any
`_vs_peers` / `_vs_hist` / `_xs` suffix -- which is the convention `CATALOGUE` already
documents. A peer leg and its raw parent are one quantity seen two ways and the economics
belong to the quantity, so there are 85 entries here for 102 columns and NOT 102 entries.
`cube_feature_catalogue.split()` is the function that does the stripping; nothing here should
reimplement it.

⚠ NO NUMBERS THAT COULD GO STALE, same rule as the fundamentals catalogue. Figures that do
appear are properties of a SOURCE or a DEFINITION, not of a build -- Item 402(k) starting at
the 2008 proxy season, the 21.9% spurious-turnover rate of raw string matching, the 1,095-day
level horizon -- and each names what it measured.
"""
from __future__ import annotations

#: characteristic -> (family, what it does, why it is right, how to read the tail)
GOVERNANCE: dict[str, tuple[str, str, str, str]] = {}


def add(name: str, family: str, what: str, why: str, tail: str) -> None:
    GOVERNANCE[name] = (family, what, why, tail)


# ------------------------------------------------------- legacy proxy levels (panel.py) #
add("ceo_pay_ratio", "proxy levels",
    "CEO-to-median-employee pay ratio exactly as the proxy discloses it.",
    "Dodd-Frank s.953(b) created the disclosure, so the field is structurally absent before "
    "FY2017 and is left NaN there rather than back-filled. Reported, never recomputed: Item "
    "402(u) lets a filer annualise a partial-year CEO or use whoever served at year end, so "
    "the disclosed ratio and total/median legitimately disagree after a mid-year change.",
    "The high tail is retail and hospitality, where a large hourly workforce sets the "
    "denominator -- a workforce-composition fact as much as a pay fact, which is why the "
    "feature also ships peer-relative.")
add("ceo_equity_pay_pct", "proxy levels",
    "Share of CEO total compensation delivered as equity. Alignment with owners rather than "
    "with salary.",
    "Domain-gated to [0, 1] and NaN'd outside it, never clipped: a value above 1 means the "
    "DENOMINATOR is wrong (RCL's 2021 proxy reports $3,042,000 of stock awards against a "
    "$645,000 total), and clipping to 1.0 would publish 'all of this CEO's pay was equity' "
    "as though it had been measured.",
    "The low tail is real and not an error -- BRK-B sits at 0 across 22 of 29 proxies because "
    "Buffett takes no equity. A zero is a pay-philosophy fact, so it must survive every "
    "staleness and constant-series check aimed at frozen data.")
add("pct_independent_directors", "proxy levels",
    "Independent share of the board.",
    "Domain-gated to [0, 1]: ABT's 2012 proxy reads 1.273, i.e. 14 independent directors on "
    "11 seats, which says the count and the denominator came from different tables. Left as a "
    "hole rather than clipped, because that filing tells us nothing about that year.",
    "Compressed near the top of the range by NYSE and Nasdaq listing rules, which mandate a "
    "majority. The informative tail is the low one: controlled companies claiming the "
    "listing-standard exemption.")
add("pct_female_directors", "proxy levels",
    "Female share of the board.",
    "Derived by the extractor from per-director rows, not from a filer-stated summary. Its "
    "IC flips sign under all three encodings, which is why it carries no monotone constraint "
    "and no peer leg -- it is carried as a characteristic, not as a signal.",
    "A secular time trend dominates the cross-section (near zero in the 1990s), so a pooled "
    "model reads the date as much as the board. Difference it before trusting a level.")
add("board_size", "proxy levels",
    "Number of board seats. Bloat versus lean.",
    "Kept raw AND peer-relative: 12.4% of its cross-sectional variance is between-sector, the "
    "highest in the panel, against 9.0% for profitMargins on the identical measure -- so "
    "'large for this industry' is a coherent question where 'large' alone is a sector proxy.",
    "The high tail is banks and insurers, which carry committee-heavy boards by regulation. "
    "Reaches 4,428 days (12.1 years) stale on BX before the level horizon; a partnership that "
    "files proxies irregularly loses the most and loses fiction, not information.")
add("avg_board_tenure", "proxy levels",
    "Mean director tenure in years. Entrenchment versus freshness.",
    "The ONLY field in the panel that earns a self-history leg: a trailing five-year self-z "
    "beat both alternatives AND held its sign across both halves of the sample (+0.0128 then "
    "+0.0270), where raw and peer each flip. 'This board is more entrenched than it has been "
    "in five years' is the economically right shape for a deterioration signal.",
    "The high tail is founder-adjacent and family-controlled boards. A long tenure is not "
    "automatically bad -- it is bad relative to the firm's own history, which is what the "
    "`_vs_hist` leg measures and the raw level cannot.")
add("insider_ownership_pct", "proxy levels",
    "Directors-and-officers-as-a-group ownership share from the proxy. Skin in the game.",
    "The most SIGN-STABLE signal in the panel (IC +0.0182 then +0.0138 across halves), which "
    "is the only reason it keeps a peer leg on a 7.7% between-sector share -- the lowest of "
    "the four that have one. ⚠ For a dual-class filer the disclosed figure is VOTING power, "
    "not economic ownership, so the two are not the same quantity across the panel.",
    "The high tail is founder-controlled firms and is the informative end. This is the field "
    "the unbounded forward-fill damaged most: JNJ carried 7,671 daily cells over 30.5 years "
    "built on 2 of its 31 proxies, and the level horizon removes 18.89% of its cells -- the "
    "largest bite anywhere in the panel, and it is removing fill, not measurement.")
add("control_wedge", "proxy levels",
    "Insider voting power minus insider economic ownership, per proxy. Zero is one share, "
    "one vote; 0.47 is a group controlling 47 percentage points more of the vote than of the "
    "equity.",
    "The number the D12 defect was accidentally measuring. The extraction used to be told "
    "three times to read percent-of-class and never the voting-power column beside it, and on "
    "59 filings it read the voting column anyway; asking for BOTH columns and differencing "
    "them turns that failure into the feature. It measures the wedge between control and "
    "cost-bearing: a founder with 61% of the votes on 14% of the equity bears one seventh of "
    "the cost of a decision they can impose unilaterally. Ships RAW ONLY -- the zero is the "
    "thesis, so peer-standardising or re-ranking it would throw away the one value that "
    "matters most, the same argument that keeps `ceo_pay_vs_revenue_growth` raw.",
    "Zero for every single-class filer, which is ~91% of the archive, and that zero is a "
    "DERIVATION: one class means one vote per share, so percent of class IS percent of voting "
    "power. A null `dual_class_shares` (8 of 12,343 filings) yields null, not zero. The high "
    "tail is the dual-class founder cohort. A NEGATIVE value is impossible for a group holding "
    "super-voting shares and is blanked with a tally entry -- it is the detector for the model "
    "putting the two legs in each other's fields.")
add("ceo_tenure", "proxy levels",
    "Years the CEO has led the firm: the current calendar year minus the point-in-time "
    "`ceo_since_year`, so it accrues DAILY rather than stepping at each proxy.",
    "Guarded against a start year in the future (which would read negative). Expired on the "
    "age of `ceo_since_year`'s filing, not on the tenure value's own freshness: a tenure "
    "accruing off a 2006 proxy is not a fresh measurement of a long tenure, it is an "
    "unverified assumption that the same person still holds the job.",
    "Its IC flips sign under all three encodings, so it ships raw with no monotone constraint. "
    "The high tail mixes founder-CEOs (where long tenure is alignment) with entrenchment "
    "(where it is the opposite), and the level cannot tell them apart -- `founder_ceo` is the "
    "column that separates them.")
add("ceo_pay_growth", "proxy levels",
    "Year-on-year PCT change in CEO total compensation, filing to filing.",
    "Nulled across a CEO change, using the SAME `names.ceo_identity_changed` key as the "
    "`ceo_comp_growth_1y` family so the two can never disagree about a transition. A raw "
    "string comparison would manufacture 356 spurious turnovers out of 1,625 (21.9%) from "
    "filers' own spelling drift alone. Infinities from a $0 prior year are dropped, not kept.",
    "A PCT change with no positivity floor under the denominator, bounded only by a "
    "cross-sectional 1%/99% trim -- so it is heavier-tailed than the log-based "
    "`ceo_comp_growth_1y` by construction. The two run at Spearman 0.958 and are near "
    "rank-identical; prefer the log leg where a magnitude matters.")
add("ceo_pay_vs_revenue_growth", "proxy levels",
    "CEO pay growth MINUS TTM revenue growth: pay racing ahead of the business.",
    "Inherits the pay leg's CEO-change guard and its staleness horizon by subtraction, so it "
    "needs neither of its own. ⚠ It mixes bases -- a PCT pay change against a PCT revenue "
    "change -- where `pay_revenue_gap` does the same comparison in log space on both legs. "
    "Both ship; they are not interchangeable.",
    "The high tail is the misalignment case the feature exists to find: a package growing "
    "while the business shrinks. Read it alongside `pay_up_revenue_down`, which flags the same "
    "state as a binary and is far less sensitive to a small revenue denominator.")
add("founder_ceo", "proxy levels",
    "1 when the CEO is a founder of the firm.",
    "Ships RAW because both peer encodings were degenerate: a peer z divides by the standard "
    "deviation of a Bernoulli draw over ~7 peers, so `_vs_peers` read as 'how many of my "
    "peers are also founder-led' and survived on 32% of cells; and `rank(pct=True)` of a "
    "binary is a two-valued affine rescale whose scale wobbles with the day's base rate.",
    "Binary, measured sector share 5.8% -- firm-specific, so there is no peer norm to "
    "standardise against. Founder-led firms behave differently (long-termism, skin in the "
    "game) and the raw indicator says so directly.")
add("say_on_pay_support", "proxy levels",
    "Most recent say-on-pay approval share AS THE PROXY REPORTS IT.",
    "A FRACTION with an absolute meaning -- 0.60 is a near-revolt at any firm in any sector -- "
    "so it ships raw and is never re-ranked per day. Sector share 2.8%, the lowest in the "
    "panel. Domain-gated to [0, 1]; that gate is inert on the live archive (0 of 4,675 values) "
    "and its first catch was a test fixture generating the column in percent.",
    "⚠ NOT independent of `sop_dissent`. The proxy reports the PRIOR year's meeting, so this "
    "is a one-year-lagged near-complement of the 8-K-sourced dissent figure, not a second "
    "measurement of it. Live range 0.034 .. 1.0 with a median of 0.94: the mass sits at "
    "near-unanimous approval and the whole signal is in the thin low tail.")

# --------------------------------------------------- exec pay, phase 4 (pay_features.py) #
add("log_ceo_total_comp", "exec pay",
    "log1p of CEO total compensation from the Summary Compensation Table. The pay LEVEL.",
    "Logged because a dollar level spanning six figures to nine is a scale, and every "
    "downstream consumer (the peer winsorisation, a linear sleeve, a monotone constraint) "
    "reads a log-dollar amount as a comparable quantity and a raw one as a size proxy. Gated "
    "on comp > 0, so a placeholder zero is NaN rather than a large negative.",
    "The high tail is real and is the point: mega-grants and one-off retention awards. Note "
    "the SCT `Total` column changed definition twice (equity excluded pre-2006, FAS 123R "
    "expense FY2006-08, grant-date fair value FY2009+), so a long time series of this level "
    "crosses two regime boundaries even though the identity total = sum(parts) holds in all "
    "three.")
add("ceo_comp_growth_1y", "exec pay",
    "log(comp_t / comp_{t-1}) in CEO total compensation, filing to filing.",
    "Triple-guarded where the legacy leg is not: both comps must be positive, the filings must "
    "be ADJACENT, and the CEO must be the same person by `ceo_identity`. A partial-year "
    "incoming CEO's package is not organic pay growth, and that is the whole reason the family "
    "exists.",
    "Symmetric in log space, so a halving and a doubling are equal and opposite -- which a "
    "percentage change is not. Compare with `ceo_pay_growth`: Pearson 0.516 but Spearman "
    "0.958, i.e. the two differ almost only in the tails, and this is the leg to use when the "
    "magnitude rather than the ordering is what matters.")
add("ceo_turnover_flag", "exec pay",
    "1 when the proxy names a different CEO than the prior filing did.",
    "Decided by `ceo_identity`, never by raw string equality -- `Timothy D. Cook` -> "
    "`Timothy Cook` -> `Tim Cook` is one CEO across three filings and a text comparison calls "
    "it two changes. The same key that nulls the growth legs raises this flag, so a nulled "
    "growth observation always has a visible reason.",
    "Binary and sparse (~1 in 8 filings). Its value is as a REGIME marker for the pay "
    "features rather than as a signal: every pay-growth column is NaN wherever this is 1, so "
    "it is the column that explains their missingness.")
add("ceo_pay_slice", "exec pay",
    "CEO total comp as a share of the top-five NEO total -- the academic CEO Pay Slice, from "
    "the per-NEO child table rather than from a summary.",
    "Built from `def14a_executive_comp` because that is the only source with the exact "
    "denominator. ⚠ That table carries THREE fiscal years per accession (Item 402(c) requires "
    "three), so the grain is (ticker, accession, name, fiscal_year) and taking it as one row "
    "per NEO per filing would triple-count the denominator.",
    "Bounded near [0.2, 1.0] by construction -- the CEO is one of the five and is normally the "
    "largest. The high tail is power concentration inside the C-suite, which is the intended "
    "reading; a value at 1.0 usually means the child table captured only the CEO and should be "
    "treated as a coverage failure, not as total dominance.")
add("ceo_pay_slice_delta_1y", "exec pay",
    "Year-on-year change in the CEO pay slice.",
    "A difference of two bounded shares, so it is already centred on zero and needs no peer "
    "relativisation -- peer-z-scoring a difference is a second relativisation of the same "
    "quantity.",
    "Symmetric around zero. A large positive is the C-suite tilting toward the CEO in one "
    "year, which is a governance event; both tails are informative and the feature carries no "
    "monotone constraint.")
add("pay_revenue_gap", "exec pay",
    "CEO pay growth minus revenue growth, both in LOG space.",
    "The base matters and is not a rounding question: a doubled package is +1.00 as a "
    "percentage change and +0.69 as a log growth, so mixing bases would read +31pp of "
    "spurious misalignment on exactly the extreme observations the family exists to find. The "
    "revenue leg therefore goes through log1p before the subtraction.",
    "Positive = pay outrunning the business. The tail is dominated by small revenue "
    "denominators, which is why the binary `pay_up_revenue_down` and its severity leg ship "
    "alongside rather than instead: the flag fires on the STATE, immune to the denominator.")
add("pay_up_revenue_down", "exec pay",
    "1 when CEO pay rose and revenue fell in the same year. The misalignment flag.",
    "Deterministic from the two signs, so it cannot be distorted by a small denominator the "
    "way the continuous gap can. A NaN on either leg gives a NaN flag, never a false 0 -- "
    "'we could not measure this' and 'no misalignment' are opposite claims.",
    "Binary and genuinely rare, which is what makes it informative; a flag that fires on a "
    "third of the panel would be a business-cycle indicator rather than a governance one.")
add("pay_up_revenue_down_severity", "exec pay",
    "The size of the pay-versus-revenue gap in the years the flag fires, and 0 otherwise.",
    "A hinge, not a mask: it keeps the flag's immunity to sign-flipping while restoring the "
    "magnitude the flag discards, so a 5% divergence and a 90% one are distinguishable where "
    "the binary saturates.",
    "Zero-inflated by design -- the mass at 0 is every non-firing year and is a real state, "
    "not missingness. Read the flag for frequency and this for depth.")
add("pay_revenue_peer_misalignment", "exec pay",
    "The pay-versus-revenue gap expressed against the peer basket.",
    "Peer-relative because pay-setting is done by a compensation consultant against a sector "
    "benchmark, so 'my package grew faster than my business AND faster than my peers' "
    "packages did' is a different and stronger claim than the raw gap.",
    "Centred on zero with the peer-z clip at +/-8. Extreme values are dominated by small "
    "peer baskets; the dispersion floor at 0.10 of the day's standard deviation is what stops "
    "a quiet basket manufacturing a large z.")
add("pay_return_gap", "exec pay",
    "CEO pay growth minus the 252-day trailing total shareholder return, in log space.",
    "The performance leg the revenue comparison cannot supply: revenue can grow while owners "
    "lose money. Built from `close_total`, the total-return series, so buybacks and dividends "
    "are inside the comparison rather than outside it.",
    "⚠ THE ONLY COLUMN IN THE PART THAT MOVES DAILY. Its pay leg is an annual step function "
    "and its return leg moves every session, so the difference re-prices continuously between "
    "proxies -- do not read a change in it as a governance event without checking which leg "
    "moved.")
add("pay_up_return_down", "exec pay",
    "1 when CEO pay rose and the trailing shareholder return was negative.",
    "The sharper half of the pay-for-performance test: a board can defend paying up through a "
    "revenue dip far more easily than through a year in which owners lost money.",
    "Binary, and more frequent than its revenue twin because returns are negative more often "
    "than revenue falls. Same NaN discipline: unmeasurable is never 0.")
add("pay_up_return_down_severity", "exec pay",
    "The size of the pay-versus-return gap in the years that flag fires, and 0 otherwise.",
    "Same hinge construction as the revenue severity leg, for the same reason -- magnitude "
    "without the sign-flip sensitivity of a ratio.",
    "Zero-inflated by design. The deep tail is a large package granted into a large drawdown, "
    "which is the single most legible governance failure in the whole panel.")
add("pay_return_peer_misalignment", "exec pay",
    "The pay-versus-return gap expressed against the peer basket.",
    "Separates a firm-specific failure from a sector-wide drawdown: paying up through a year "
    "when the whole sector fell is a different fact from doing it when only you fell.",
    "Centred on zero, clipped at +/-8. In a sector-wide crash the raw gap fires on everyone "
    "and this leg on nobody -- which is the correct behaviour and the reason both ship.")

# ------------------------------------------- provisions and auditor (provisions_features.py) #
add("classified_board_added", "provisions",
    "1 in the year a staggered (classified) board was ADOPTED.",
    "A TRANSITION, not a level. A classified board is a fact about a company; adopting one is "
    "an act by a board, and it is the act that carries information. The NULL rows are dropped "
    "BEFORE the diff, so a silent proxy year can neither manufacture a transition nor hide one.",
    "Binary and very sparse -- adoptions are rare in the modern era, which is why the "
    "aggregate counts ship alongside every individual flag.")
add("classified_board_removed", "provisions",
    "1 in the year a staggered board was removed. Declassification is the improvement leg.",
    "Same tri-state discipline as its `_added` twin. Both directions ship separately because "
    "adopting and removing are not one signed variable: the populations that do each are "
    "different.",
    "Binary and sparse, but the commoner direction of the pair -- declassification campaigns "
    "were a sustained governance-activist theme.")
add("dual_class_added", "provisions",
    "1 in the year a dual-class share structure appeared.",
    "Ships only in the `_added` direction; a dual-class structure is almost never unwound, so "
    "a `_removed` twin would be a near-constant zero and a constant column is worse than an "
    "absent one.",
    "Binary and extremely sparse. ⚠ A firing usually means an IPO or a recapitalisation rather "
    "than an existing board taking a new action, so read it as a structure marker rather than "
    "as a governance decision.")
add("majority_voting_removed", "provisions",
    "1 in the year majority voting for director elections was dropped.",
    "`majority_voting` is deliberately TRI-STATE -- NULL when the proxy is silent -- because "
    "inferring FALSE from silence flipped it 21.2% year-over-year on a bylaw that does not "
    "change. The transition is computed only between two DISCLOSED years.",
    "Binary and sparse. The tri-state discipline is what makes it trustworthy: nearly all of "
    "the apparent churn in this bylaw was an artefact of reading silence as a value.")
add("majority_voting_added", "provisions",
    "1 in the year majority voting for directors was adopted.",
    "Same tri-state gate as its twin. The improvement direction, and the commoner one: "
    "plurality-to-majority was a decade-long governance shift.",
    "Binary, sparse, and time-clustered rather than spread evenly -- a pooled model can read "
    "the era from it, so prefer it inside the aggregate counts.")
add("poison_pill_removed", "provisions",
    "1 in the year a shareholder-rights plan (poison pill) lapsed or was withdrawn.",
    "⚠ Its `_added` twin fires ZERO times on the live archive and is deliberately NOT "
    "exported -- a constant-0 column is worse than an absent one, because it looks like "
    "evidence. `poison_pill` is tri-state for the same reason as `majority_voting`: inferring "
    "FALSE from silence made it read TRUE for 0.1% of rows.",
    "Binary and one-directional in practice. The asymmetry is real: pills are adopted in "
    "response to a specific threat and rarely disclosed in the proxy that adopts them.")
add("ceo_became_board_chair", "provisions",
    "1 in the year the CEO also took the board chair. Deterioration.",
    "Board-leadership structure ships as transitions for the same reason the bylaws do -- the "
    "ACT is the information. Combining the CEO and chair roles removes the board's most "
    "direct check on the executive.",
    "Binary, sparse, and one of the more legible deterioration events in the panel; its "
    "reversal ships as `ceo_stopped_being_chair`.")
add("ceo_stopped_being_chair", "provisions",
    "1 in the year the CEO gave up the board chair. Improvement.",
    "The other direction of the same event, as its own column rather than as a sign, because "
    "the firms that split the roles are not the firms that combine them.",
    "Binary and sparse. Frequently co-fires with `independent_chair_added` -- the two are "
    "usually one board decision seen from two sides, which is why both feed the aggregate "
    "counts rather than being multiplied together.")
add("independent_chair_lost", "provisions",
    "1 in the year an independent board chair ceased to be independent, or the role lapsed.",
    "Tracks the chair's INDEPENDENCE, which is a distinct fact from whether the CEO holds the "
    "chair -- a non-CEO insider chair is neither an independent chair nor a combined role.",
    "Binary and sparse. This is the column that catches the founder-returns-as-chair case, "
    "which `ceo_became_board_chair` misses entirely.")
add("independent_chair_added", "provisions",
    "1 in the year an independent chair was installed. Improvement.",
    "Same independence basis as its `_lost` twin.",
    "Binary, sparse, and time-clustered with the post-2003 governance reforms.")
add("lead_independent_director_lost", "provisions",
    "1 in the year the lead-independent-director role disappeared.",
    "The lead independent director is the structural substitute for an independent chair when "
    "the roles are combined, so losing it is a real deterioration even at a firm whose formal "
    "structure did not change.",
    "Binary and sparse. Most informative at firms with a combined CEO/chair, where it is the "
    "only remaining independent counterweight.")
add("lead_independent_director_added", "provisions",
    "1 in the year the lead-independent-director role was created.",
    "The improvement direction, and typically the board's answer to pressure over a combined "
    "CEO/chair rather than an independent action.",
    "Binary, sparse, and often the year AFTER a say-on-pay or director-election revolt -- "
    "worth reading against `sop_dissent_gt_20` rather than in isolation.")
add("governance_deterioration_count", "provisions",
    "Count of the deterioration flags firing in a given proxy year.",
    "ONE EVENT, ONE POINT -- no weights. A weighting scheme would encode an untested claim "
    "about the relative severity of a classified board versus a combined chair; the tree "
    "models can learn that themselves from the individual flags, which all ship.",
    "A small integer, overwhelmingly 0. The whole distribution above 1 is a handful of "
    "firm-years in which a board made several entrenching changes at once, which is exactly "
    "the state the aggregate exists to surface.")
add("governance_improvement_count", "provisions",
    "Count of the improvement flags firing in a given proxy year.",
    "Same unweighted construction as its deterioration twin, so the two are directly "
    "comparable and their difference is meaningful.",
    "A small integer, overwhelmingly 0, and time-clustered around the major reform waves.")
add("net_governance_change", "provisions",
    "Improvement count minus deterioration count.",
    "Ships alongside both legs rather than replacing them, because a 0 produced by no events "
    "and a 0 produced by one change in each direction are different states and the difference "
    "alone cannot tell them apart.",
    "Symmetric, tightly massed at 0. Both tails are informative; there is no monotone "
    "constraint because the sign convention is a labelling choice, not a measured direction.")
add("board_independence_delta_1y", "provisions",
    "Year-on-year change in the independent share of the board.",
    "A difference of two gated shares, so the [0, 1] domain gate on the level protects it too. "
    "Gated on PROVENANCE: a pair with an interpolated leg is rejected, because two thirds of "
    "the pairs in the comparable busyness delta had one and the feature was substantially a "
    "measurement of the fill rather than of the board.",
    "Centred on zero and dominated by the granularity of a board: on an 11-seat board the "
    "smallest possible move is ~9 percentage points, so the distribution is lumpy rather than "
    "smooth and small nonzero values are impossible rather than rare.")
add("board_independence_drop_10pp", "provisions",
    "1 when the independent share fell by 10 percentage points or more in one year.",
    "A threshold on the delta rather than a second measurement of it: on a typical board one "
    "seat is about 9 points, so a 10-point bar is 'more than one seat' expressed in a unit "
    "that is comparable across board sizes.",
    "Binary and sparse. Fires on genuine board restructurings and on post-acquisition board "
    "reconstitution, which are different causes with a similar signature.")
add("board_busyness", "provisions",
    "Mean number of OTHER public-company board seats held per director.",
    "DERIVED from the per-director child table after a per-person fill, not read from the "
    "filer's own scalar and not interpolated. The parent scalar already is the mean of its own "
    "children (correlation 1.000, median absolute difference 0.00), so the gain is entirely in "
    "filling the CHILD table first -- `other_public_company_boards` is present on only 29.3% "
    "of director rows and the median filing carrying any covers 67% of its own board.",
    "The high tail is the ISS 'overboarded' concern -- directors spread thin. Read alongside "
    "`pct_overboarded`, which counts the directors over a bar instead of averaging, and so "
    "survives a board with one extremely busy member.")
add("board_busyness_delta_1y", "provisions",
    "Year-on-year change in mean other-board seats per director.",
    "The number that justifies deriving the level at all: on the interpolated basis this "
    "feature rejected 65.5% of its pairs as having a filled leg, so it was substantially a "
    "measurement of the fill. Re-based on the derivation it rejects ~22.5% and the clean "
    "population multiplies by 2.25x.",
    "Centred on zero. Still gated on provenance, so its coverage is materially lower than the "
    "level's -- that gap is the honest cost of not differencing interpolated values.")
add("auditor_is_big4", "provisions",
    "1 if the engaged audit firm is one of the Big Four.",
    "Resolved from the CANONICAL firm name rather than the raw string, so the ~40 spellings, "
    "legacy names and post-merger forms of the same firm collapse to one entity. A raw-string "
    "test would read a rebrand as an auditor change.",
    "Binary and heavily skewed toward 1 in the S&P 500 -- the informative side is the small "
    "non-Big-Four population, which is where an audit-quality concern would actually live.")
add("auditor_tenure", "provisions",
    "Years the current audit firm has been engaged.",
    "Also canonical-firm-based, so a rebrand does not reset the clock. Long auditor tenure is "
    "the classic independence-impairment concern and the reason the EU mandated rotation.",
    "⚠ LEFT-CENSORED for any engagement that predates the disclosure, so the level is a LOWER "
    "BOUND on the true tenure rather than a measurement. `auditor_tenure_censored` is the flag "
    "that says which rows those are; using the level without it treats a floor as a fact.")
add("auditor_tenure_censored", "provisions",
    "1 when auditor tenure is left-censored -- the engagement predates the archive.",
    "Shipped as its own column rather than encoded as a NaN in the level, because the level IS "
    "informative on those rows, just as a bound. Nulling them would discard real information; "
    "publishing them unflagged would understate long tenures systematically.",
    "Binary, and concentrated in the early history where the archive itself is shallow -- so "
    "it correlates with the date and should not be read as a firm characteristic.")
add("auditor_changed", "provisions",
    "1 in the year the audit firm changed.",
    "Canonical-firm comparison, which is what makes this trustworthy: on raw strings a rebrand "
    "or a spelling change fires this flag and there is no genuine event behind it.",
    "Binary and sparse. An auditor change is a low-frequency, high-information event; the "
    "cause matters enormously (a fee dispute and a disagreement over accounting look "
    "identical here) and the proxy alone cannot distinguish them.")
add("ceo_is_board_chair", "provisions",
    "1 when the CEO currently also chairs the board.",
    "The one LEVEL in this module, and it ships as a plain column because of D16: none of the "
    "five proposed interaction products are built, and this leg existed only inside one of "
    "them. Every leg of every product ships separately and the tree models find the products "
    "themselves -- no column in the part has `_x_` in its name and a test asserts it.",
    "Binary, and the most common single governance concern in the US market. Read the two "
    "transition columns for the ACT and this for the standing STATE.")

# --------------------------------------------------- board quality, phase 6 (directors.py) #
add("board_turnover", "board quality",
    "Share of board seats that changed occupant since the prior proxy.",
    "Keyed on `(ticker, name)` AS FILED, which is a measured choice: the looser "
    "`lastname|firstinitial` bucket collides on 261 of 12,239 boards with people who are "
    "demonstrably different humans (ADM 1995 has both `Martin L. Andreas` and "
    "`Michael D. Andreas`), and each collision would hide a real seat change.",
    "The high tail is genuine board reconstitution -- post-activist settlements, post-merger "
    "boards. A moderate level is healthy refreshment, so neither tail is unambiguously good "
    "and the feature carries no monotone constraint.")
add("pct_long_tenured", "board quality",
    "Share of directors with 15 or more years of tenure.",
    "What a board AVERAGE structurally cannot express. A mean tenure of 8 years is produced "
    "both by a uniformly mid-tenure board and by a board split between founders and new "
    "arrivals, and only the second is an entrenchment concern.",
    "Bounded [0, 1]. The high tail is family- and founder-controlled boards. Prefer this to "
    "`avg_board_tenure` when the question is entrenchment rather than experience.")
add("pct_overboarded", "board quality",
    "Share of directors holding 3 or more OTHER public-company boards -- the ISS overboarding "
    "bar.",
    "A count over a threshold rather than an average, so one extremely busy director cannot "
    "carry the whole board's reading the way it can in `board_busyness`. The bar is ISS's, not "
    "ours, which makes the feature comparable to how index investors actually vote.",
    "Bounded [0, 1] and mostly low. The informative tail is the high one and it tends to be "
    "prestige boards -- the directors most in demand are the most overcommitted.")
add("board_tenure_dispersion", "board quality",
    "Standard deviation of director tenure across the board.",
    "Distinguishes a board refreshed in WAVES from one refreshed continuously, which no "
    "measure of the average can. A staggered-refresh board and a cliff-refresh board can share "
    "a mean tenure exactly.",
    "The high tail is a board with a founder cohort plus recent arrivals and nothing between "
    "-- a succession risk that reads as healthy on every average. Denominator-sensitive on "
    "very small boards.")
add("board_age_dispersion", "board quality",
    "Standard deviation of director age.",
    "Ships RAW despite a 7.20% between-sector variance share, which is the near miss in this "
    "family: the 7.7% floor for a peer leg belongs to `insider_ownership_pct` and is granted "
    "only because that field is the most sign-stable signal in the panel. A board's age spread "
    "has no such record to lean on, so the leg was deliberately NOT taken.",
    "Low values are a generational monoculture, high ones a genuinely mixed board. Depends on "
    "`avg_director_age`'s child-table coverage, which is the one field where the derivation "
    "also paid a coverage dividend (+250 filings neither filed nor interpolable).")
add("oldest_director_age", "board quality",
    "Age of the oldest serving director.",
    "A retirement-policy pressure proxy: most boards have a stated retirement age, so this "
    "measures how close the board is to having to act, and whether it has waived its own rule.",
    "The high tail is boards that have granted an explicit waiver, which is itself a "
    "governance signal. Coverage tracks director-age disclosure, which is voluntary and "
    "uneven -- a missing value here is a disclosure fact, not an age fact.")

# ------------------------------------------------ director pay, phase 6 (director_comp.py) #
add("ceo_to_director_pay_ratio", "director pay",
    "CEO total compensation over MEDIAN non-employee-director total compensation.",
    "A power-concentration measure: executive pay is what the board GRANTS, director pay is "
    "what it TAKES, and the ratio between them is a measure of the board's own stake in the "
    "relationship. Measured p10-p90 spread of 19.2x to 79.5x around a median of 43.5x -- the "
    "dispersion of a real signal rather than of a rounding artefact.",
    "⚠ A RATIO WHOSE ABSOLUTE LEVEL IS THE THESIS, so it ships raw and is never re-centred on "
    "a sector mean -- 80x is entrenchment at any firm in any sector, and peer-relativising it "
    "would encode a whole sector's excess as normal. Ford read exactly 0 for 2,769 consecutive "
    "trading days from a 2011 CEO leg extracted as 0 welded to a director leg that vanished "
    "for a decade; the 1,095-day level horizon is what now bounds that.")
add("director_cash_fee_pct", "director pay",
    "Cash share of median director pay.",
    "One of only two peer legs this family earns, and earned on the same between-sector "
    "variance-share measure that refuted every other candidate: 11.94%, against the surviving "
    "legacy legs' 7.7%-12.4%. That is the economically right shape too -- a director's "
    "cash-versus-equity mix is set by a compensation consultant against a sector benchmark, so "
    "'how much more cash than my sector' is a real question.",
    "⚠ THE ONE DELIBERATE CLIP IN THE WHOLE GOVERNANCE PANEL. Its breach is 1.001-1.093 from a "
    "summation artefact rather than a broken denominator, so it is clipped where every other "
    "out-of-domain value is NaN'd; the exception lives beside the code that computes it.")
add("director_equity_pay_pct", "director pay",
    "Equity share of median director pay. Director alignment with owners.",
    "The family's other peer leg, at 8.17% between-sector share. ⚠ Checked against its cash "
    "twin before shipping both: pooled correlation -0.51 (mean per-date cross-sectional -0.47, "
    "never beyond -0.64), because cash + equity sums to a median 0.98 but a p10 of only 0.66 "
    "-- option awards, pension accruals and 'other' take a third of the pay bill on many "
    "boards. Two complements would have been one column twice over.",
    "The high tail is boards paid predominantly in stock, the alignment case. Note the pair "
    "does NOT sum to 1, and treating the two as complements would reintroduce exactly the "
    "r = 1.0 defect the cube has already paid for once.")
add("log_median_director_pay", "director pay",
    "log1p of median non-employee-director total compensation.",
    "The MEDIAN rather than the mean, deliberately: a lead-director or committee-chair "
    "retainer is a fat tail on a ten-person board. Logged for the same reason as "
    "`log_ceo_total_comp` -- a dollar level spanning $149k (p10) to $360k (p90) is a scale.",
    "⚠ ITS COVERAGE IS A REGIME STAIRCASE, NOT AN EXTRACTION GAP: Item 402(k) created the "
    "Director Compensation Table in the 2006 Reg S-K overhaul, so the field starts at the 2008 "
    "proxy season by regulation (5.7% in 1995-99 rising to 91.3% in 2020-26) and is left NaN "
    "before it, never filled. Six tickers have zero rows here and that IS a parser defect -- "
    "APP, CRH, IBM, PANW, VTRS, WDAY -- so a 91%-covered family missing IBM entirely is not "
    "91% covered.")

# ---------------------------------------- say-on-pay dissent (vote_dissent_features.py) #
add("sop_dissent", "say-on-pay dissent",
    "(against + abstain) / valid votes on the say-on-pay proposal, from the Item 5.07 8-K.",
    "The only numbers in EDGAR certified by the OWNERS rather than reported by the company. "
    "ABSTENTION IS OPPOSITION on an advisory vote -- an abstention is a deliberate withholding "
    "-- which is why ISS reads against + abstain and why the `against / valid` twin was "
    "retired at r = 0.9949. BROKER NON-VOTES ARE EXCLUDED from the denominator: a broker "
    "non-vote is an absent instruction, not an opinion.",
    "Heavily massed near zero -- most packages pass overwhelmingly -- so the entire signal is "
    "in the thin high tail. Expires on the 548-day EVENT horizon: a 2019 vote is not evidence "
    "about 2026, and a feature that keeps reporting it asserts an opinion nobody expressed.")
add("sop_dissent_excess_10", "say-on-pay dissent",
    "max(0, sop_dissent - 0.10). A hinge above the 10% opposition bar.",
    "LINEAR in the tail where the companion flag saturates, so a 35% revolt is distinguishable "
    "from a 21% one. The hinge keeps the flag's threshold interpretation and restores the "
    "magnitude it discards.",
    "Zero for a normal vote and that mass at 0 is a real state, not missingness. Read with "
    "`sop_dissent_gt_10` for frequency and this for depth.")
add("sop_dissent_excess_20", "say-on-pay dissent",
    "max(0, sop_dissent - 0.20). A hinge above the 20% bar.",
    "20% is the ISS 'significant opposition' bar, the level at which a board is expected to "
    "respond publicly, so the threshold is the market's rather than ours.",
    "Zero-inflated and much sparser than the 10% hinge. Anything above zero here is a firm "
    "whose board owed shareholders an explanation.")
add("sop_dissent_gt_10", "say-on-pay dissent",
    "1 when say-on-pay dissent exceeds 10%.",
    "Ships RAW and is never peer-z-scored or percentile-ranked: the cross-sectional rank of a "
    "binary is a re-encoding of its base rate, and a peer z of a flag over ~7 peers reads as "
    "'how many of my peers also tripped this' -- close to the opposite of the intended signal.",
    "Binary. Membership in `RAW_FLAG_FIELDS` is the ONLY thing that keeps it out of the "
    "z-scoring path, so a new flag not listed there silently becomes a z-score.")
add("sop_dissent_gt_20", "say-on-pay dissent",
    "1 when say-on-pay dissent exceeds 20% -- the ISS significant-opposition bar.",
    "The threshold that carries an institutional consequence rather than a statistical one: "
    "crossing it triggers engagement obligations and adverse vote recommendations the "
    "following year.",
    "Binary and sparse. Often the best single predictor of the NEXT year's "
    "`lead_independent_director_added` or pay-structure change -- boards respond to it.")
add("sop_dissent_delta_1y", "say-on-pay dissent",
    "Year-on-year change in say-on-pay dissent.",
    "Already a difference centred on zero, so it gets no peer leg -- peer-relativising a "
    "difference is a second relativisation of the same quantity.",
    "Symmetric around zero, and both tails matter: rising dissent is deterioration, falling "
    "dissent after a revolt is a board that responded. Expires on the event horizon like its "
    "parent.")

# -------------------------------------- board election dissent (vote_dissent_features.py) #
add("board_dissent_mean", "election dissent",
    "Mean per-nominee dissent across the director slate.",
    "Built from `nominee_votes_json`, filled on 100% of the 7,884 election rows, so the "
    "per-nominee grain is real rather than reconstructed. ⚠ WITHHELD IS ALREADY INSIDE "
    "`votes_against`: 20.0% of election filings run a plurality standard and print 'Withheld' "
    "instead of 'Against', and the extractor writes it into `votes_against` either way -- "
    "adding a withhold term would double-count exactly those filings.",
    "Massed near zero; uncontested director elections are near-unanimous. The high tail is a "
    "board under coordinated opposition rather than a single unpopular nominee, which is what "
    "`board_dissent_max` isolates.")
add("board_dissent_median", "election dissent",
    "Median per-nominee dissent across the slate.",
    "Robust to the single-targeted-director case that pulls the mean, so the pair separates "
    "'the whole board is unpopular' from 'one nominee is'.",
    "Nearly always below the mean, and the GAP between them is the informative quantity -- a "
    "wide gap is a targeted campaign.")
add("board_dissent_max", "election dissent",
    "The worst single nominee's dissent. The targeted-director signal.",
    "An extremum rather than a central tendency, because activist campaigns and ISS adverse "
    "recommendations target individuals -- typically a compensation-committee chair -- and any "
    "average dilutes that by the size of the board.",
    "The highest-magnitude column of the family by construction. Read against the mean: a high "
    "max with a low mean is one director in trouble, and both high is a board in trouble.")
add("board_dissent_p90", "election dissent",
    "90th percentile of per-nominee dissent.",
    "A softer extremum than the max -- less sensitive to a single outlier nominee while still "
    "reading the opposed end of the slate. On a ten-person board it is close to the "
    "second-worst nominee.",
    "Sits between the median and the max by construction, so it is only informative alongside "
    "them; on very small boards it collapses onto the max.")
add("board_dissent_breadth_10", "election dissent",
    "Share of nominees drawing more than 10% dissent. BREADTH rather than depth.",
    "A different question from any magnitude measure: two directors at 30% and six at 2% is a "
    "targeted campaign, while all eight at 12% is a systemic objection to the board, and the "
    "two can share a mean.",
    "Bounded [0, 1] and lumpy on the size of the board -- on a nine-seat board the possible "
    "values are ninths. Massed at 0.")
add("board_dissent_breadth_20", "election dissent",
    "Share of nominees above 20% dissent.",
    "The stricter breadth bar, matching say-on-pay's significant-opposition threshold so the "
    "two families are read on a comparable scale.",
    "Bounded [0, 1], far sparser than the 10% version. Anything materially above 0 is a "
    "board-wide revolt, which is rare enough to be worth inspecting individually.")
add("board_pct_nominees_below_70_support", "election dissent",
    "Share of nominees drawing under 70% support.",
    "Expressed on the SUPPORT basis rather than the dissent basis, deliberately: 70% support "
    "is the conventional line below which a director's mandate is questioned, and stating it "
    "as support is what keeps the two bases from being confused for one another.",
    "Bounded [0, 1], very sparse. A firing nominee is one whose re-election is a live "
    "governance question rather than a formality.")
add("ceo_director_dissent", "election dissent",
    "Dissent against the CEO's OWN board seat.",
    "The CEO-specific leg, and the one election field that earns a peer leg. Shareholders vote "
    "on the CEO as a director, which makes this the most direct available referendum on the "
    "individual rather than on the package or the board.",
    "Massed near zero. The high tail is the strongest single-name governance signal in the "
    "panel: institutional owners voting against the chief executive personally.")
add("ceo_excess_dissent", "election dissent",
    "CEO dissent minus the board mean: is the CEO specifically the target?",
    "Built in the SAME pass as the board median from the SAME ballot -- reading the election "
    "rows twice would be both wasteful and a chance for the two legs to drift apart. Already "
    "a difference centred on zero, so no peer leg.",
    "Symmetric around zero. Positive is a CEO doing worse than their own board, which is the "
    "targeted case; negative is a CEO shielded by a slate that is more unpopular than they are.")
add("management_dissent_spread", "election dissent",
    "Executive-officer nominees' dissent minus the non-employee nominees'.",
    "Separates opposition to MANAGEMENT from opposition to the board as a whole. Its peer leg "
    "was deliberately dropped: measured coverage was 2.2% on 102 tickers, because `min_peers=3` "
    "cannot be met on a field present in only 17.7% of ballots.",
    "The sparsest field in the family -- most slates have one or no executive nominee besides "
    "the CEO. Read the coverage before reading the value.")
add("ceo_vs_nonemployee_dissent", "election dissent",
    "CEO dissent minus the non-employee-director mean.",
    "EXACT where `ceo_excess_dissent` is approximate: the non-employee bucket excludes the CEO "
    "by construction, so this difference has no overlap between its two legs, while the board "
    "mean contains the CEO's own vote.",
    "Symmetric around zero and the cleaner of the two CEO-versus-board measures. Prefer it "
    "where the contamination matters, i.e. on small boards where the CEO is a large share of "
    "the mean.")
add("board_dissent_mean_delta_1y", "election dissent",
    "Year-on-year change in mean board dissent.",
    "A difference of differences, centred on zero, so it gets no peer leg.",
    "Symmetric. Rising board-wide dissent is the deterioration reading; the level says how bad "
    "it is and the delta says which way it is moving.")
add("board_dissent_median_delta_1y", "election dissent",
    "Year-on-year change in median board dissent.",
    "The robust twin of the mean delta, for the same targeted-versus-systemic reason as the "
    "levels.",
    "Symmetric and quieter than the mean delta; a move here is a shift in the whole slate "
    "rather than in one nominee.")
add("board_dissent_max_delta_1y", "election dissent",
    "Year-on-year change in the worst nominee's dissent.",
    "Tracks an extremum across years, so it can move because a different director became the "
    "worst rather than because any director's support changed -- a real limitation, and the "
    "reason the mean and median deltas ship alongside.",
    "The noisiest delta in the family by construction. Corroborate it against the breadth "
    "delta before reading a single year's move as an event.")
add("board_dissent_breadth_10_delta_1y", "election dissent",
    "Year-on-year change in the share of nominees above 10% dissent.",
    "The breadth delta, which is the one that distinguishes a campaign WIDENING from a "
    "campaign deepening -- the magnitude deltas cannot.",
    "Lumpy on board size, like its level. Symmetric around zero and massed there.")
add("ceo_dissent_delta_1y", "election dissent",
    "Year-on-year change in dissent against the CEO's board seat.",
    "The CEO-specific delta. No peer leg, being already a difference.",
    "Symmetric. A large positive is institutional owners turning on a chief executive within "
    "one year, which is about as sharp a governance deterioration signal as the archive holds.")

# --------------------------------- auditor ratification dissent (vote_dissent_features.py) #
add("auditor_vote_dissent", "auditor dissent",
    "(against + abstain) / valid votes on auditor ratification.",
    "The cheapest available protest vote, which is exactly why abstentions belong in the "
    "numerator. Retired its `against / valid` twin at r = 0.9867, where this leg is a STRICT "
    "SUPERSET -- 254 cells and 6 distinct values more, and not one cell where only the twin "
    "existed. ⚠ Under NYSE Rule 452 auditor ratification is a ROUTINE matter where brokers may "
    "vote uninstructed, so auditor rows carry broker non-votes on only 30.4% of rows against "
    "nearly always for say-on-pay -- a shared denominator would make the two incomparable.",
    "Massed very close to zero; ratification normally passes overwhelmingly. Carries a real "
    "SECULAR TREND (cross-sectional median 0.012 -> 0.047, a 3.9x rise as proxy advisers "
    "turned on audit firms), so a pooled model can read the date from the level -- difference "
    "it or use the delta leg if that matters.")
add("auditor_dissent_gt_05", "auditor dissent",
    "1 when auditor dissent exceeds 5%.",
    "The bar is deliberately TIGHTER than say-on-pay's 10%: auditor ratification normally "
    "passes with overwhelming support, so 5% opposition is already the tail here where "
    "say-on-pay routinely runs to 10%. Using one threshold for both would make the auditor "
    "flag near-constant.",
    "Binary, in `RAW_FLAG_FIELDS`, so never z-scored. Its base rate rises through the sample "
    "with the underlying secular trend.")
add("auditor_dissent_gt_10", "auditor dissent",
    "1 when auditor dissent exceeds 10%.",
    "The severe bar for this family. Two thresholds ship rather than one because the "
    "distribution is so compressed that 5% and 10% select genuinely different populations.",
    "Binary and sparse. A firing is a serious institutional objection to the audit firm -- "
    "typically a tenure or non-audit-fee concern.")
add("auditor_vote_dissent_delta_1y", "auditor dissent",
    "Year-on-year change in auditor ratification dissent.",
    "Already a difference, so no peer leg. It is also the natural way to use this family given "
    "the secular trend in the level: a change is comparable across eras where a level is not.",
    "Symmetric around zero and tightly massed. The informative tail is a sharp one-year jump, "
    "which usually follows a restatement or a fee disclosure rather than a slow drift.")
