SELECT vg.DATE_VENTE AS DATE, vg.CODE_PDV, vg.CODE_PRGE, vg.MNT_VENTE, vg.QTE_VENTE, vg.POIDS_VENTE_ESTIM, vg.FRACTION_PROMO,
       rpg.LIB_PRGE, rpg.CODE_NOMENCLATURE, rpg.POIDS_NET_UC,
       rpdv.CODE_VOCATION, rpdv.CODE_BASE,
       rpdv.CODE_PAYS, rpdv.CODE_POSTAL, rpdv.LIB_VILLE, rpdv.LIB_ADRS, rpdv.GPS_X, rpdv.GPS_Y,
       rpdv.SURFACE_TOTALE, rpdv.NB_CAISSES, rpdv.DATE_ACTIVATION_TECH, rpdv.DATE_DESACTIVATION_TECH
FROM ventes_generiques vg LEFT JOIN ref_produit_generique rpg ON vg.CODE_PRGE = rpg.CODE_PRGE
LEFT JOIN ref_point_de_vente rpdv ON vg.CODE_PDV = rpdv.CODE_PDV
