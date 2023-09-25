class CLIMATE:
    CLIMATE_DATA_COLUMNS = [
        "p (mbar)",
        "T (degC)",
        "Tpot (K)",
        "Tdew (degC)",
        "rh (%)",
        "VPmax (mbar)",
        "VPact (mbar)",
        "VPdef (mbar)",
        "sh (g/kg)",
        "H2OC (mmol/mol)",
        "rho (g/m**3)",
        "wv (m/s)",
        "max. wv (m/s)",
        "wd (deg)",
        "Time_Stamp",
    ]

    COLUMN_KEYS = dict(
        map(
            lambda key_index: (key_index[1], key_index[0]),
            enumerate(CLIMATE_DATA_COLUMNS),
        )
    )
