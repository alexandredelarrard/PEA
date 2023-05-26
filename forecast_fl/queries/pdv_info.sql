SELECT vs.CODE_PDV AS COD_SITE, rpdv.CODE_VOCATION AS VOCATION,
    rpdv.CODE_BASE, rpdv.SURFACE_TOTALE, rpdv.NB_CAISSES, rpdv.NOM_PDV,
    sum(vs.MNT_VENTE) AS YEAR_SELL
FROM ventes_standards vs JOIN ref_point_de_vente rpdv ON vs.CODE_PDV = rpdv.CODE_PDV
where vs.DATE_VENTE > DATEADD(month, -6, GETDATE())
group by vs.CODE_PDV, rpdv.CODE_VOCATION, rpdv.CODE_BASE, rpdv.SURFACE_TOTALE, rpdv.NB_CAISSES, rpdv.NOM_PDV

-- SELECT rpdv.CODE_PDV AS COD_SITE, rpdv.CODE_VOCATION AS VOCATION, rpdv.CODE_BASE,
--        rpdv.SURFACE_TOTALE, rpdv.NB_CAISSES, rpdv.NOM_PDV
-- FROM ref_point_de_vente rpdv
