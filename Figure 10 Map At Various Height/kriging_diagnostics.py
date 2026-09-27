#!/usr/bin/env python3
"""Ordinary Kriging diagnostics for WS, WPD, and WED at 10--100 m.

Run in the MapAtVariousHeight directory. The script reads the three CSV files,
fits a spherical variogram, performs leave-one-out cross-validation, exports
variogram parameters, prediction errors, standardized errors, kriging variance
summaries, and diagnostic plots.
"""
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point

warnings.filterwarnings('ignore')
BASE = Path.cwd()
OUT = BASE / 'Kriging_Diagnostics_Output'
OUT.mkdir(exist_ok=True)
SHAPEFILE = Path('/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_shape_file-with-Kashmir/Pakistan_with_Kashmir.shp')
FILES = {
    'WS': BASE / '1-Wind-Speed-at-different-height.csv',
    'WPD': BASE / '2-Wind-Power-Density-at-different-height.csv',
    'WED': BASE / '3-Wind-Energy-Density-at-different-height.csv',
}
HEIGHTS = [10,20,30,40,50,60,70,80,90,100]
VARIOGRAM = 'spherical'


def clean(df):
    df.columns = (df.columns.astype(str).str.replace('\xa0',' ',regex=False).str.replace(r'\s+',' ',regex=True).str.strip())
    for c in ['Latitude','Longitude']:
        if c not in df.columns:
            raise ValueError(f'Missing coordinate column: {c}')
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_inputs():
    if not SHAPEFILE.exists():
        raise FileNotFoundError(f'Shapefile not found: {SHAPEFILE}')
    for path in FILES.values():
        if not path.exists():
            raise FileNotFoundError(f'Input CSV not found: {path}')
    boundary = gpd.read_file(SHAPEFILE).to_crs(4326)
    return boundary, {k: clean(pd.read_csv(v)) for k,v in FILES.items()}


def fit_ok(lon, lat, val):
    return OrdinaryKriging(lon, lat, val, variogram_model=VARIOGRAM, coordinates_type='geographic', verbose=False, enable_plotting=False)


def parameters(ok):
    p = np.asarray(ok.variogram_model_parameters, dtype=float).ravel()
    # PyKrige spherical parameters are [partial sill, range, nugget].
    return {'Variogram_model': VARIOGRAM, 'Partial_sill': p[0] if len(p)>0 else np.nan, 'Range': p[1] if len(p)>1 else np.nan, 'Nugget': p[2] if len(p)>2 else np.nan}


def loo_cv(df, col):
    d = df[['Longitude','Latitude',col]].copy()
    d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna()
    rows=[]
    if len(d)<4:
        return pd.DataFrame(), {}
    for i, row in d.iterrows():
        train=d.drop(index=i)
        try:
            ok=fit_ok(train.Longitude.to_numpy(), train.Latitude.to_numpy(), train[col].to_numpy())
            pred,var=ok.execute('points', np.array([row.Longitude]), np.array([row.Latitude]))
            pred=float(np.asarray(pred).ravel()[0]); var=float(np.asarray(var).ravel()[0])
            err=pred-float(row[col])
            rows.append({'held_out_index':i,'Observed':float(row[col]),'Predicted':pred,'Error_pred_minus_observed':err,'Kriging_variance':var,'Standardized_error':err/np.sqrt(max(var,1e-12))})
        except Exception as e:
            rows.append({'held_out_index':i,'Observed':float(row[col]),'Predicted':np.nan,'Error_pred_minus_observed':np.nan,'Kriging_variance':np.nan,'Standardized_error':np.nan,'Error':str(e)})
    cv=pd.DataFrame(rows).dropna(subset=['Predicted'])
    if cv.empty: return cv, {}
    e=cv.Error_pred_minus_observed.to_numpy(); obs=cv.Observed.to_numpy(); pred=cv.Predicted.to_numpy()
    stats={'N_CV':len(cv),'Mean_error':float(np.mean(e)),'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'Mean_kriging_variance':float(cv.Kriging_variance.mean()),'Mean_standardized_error':float(cv.Standardized_error.mean()),'Std_standardized_error':float(cv.Standardized_error.std(ddof=1)),'R2_CV':float(np.corrcoef(obs,pred)[0,1]**2) if len(cv)>2 else np.nan}
    return cv, stats


def main():
    boundary, datasets=load_inputs()
    all_params=[]; all_cv=[]; all_summary=[]
    for var,df in datasets.items():
        for h in HEIGHTS:
            col=f'{var} at {h}m'
            if col not in df.columns:
                print(f'Skip missing column: {col}'); continue
            d=df[['Longitude','Latitude',col]].dropna()
            if len(d)<4:
                print(f'Skip {col}: fewer than 4 stations'); continue
            ok=fit_ok(d.Longitude.to_numpy(),d.Latitude.to_numpy(),d[col].to_numpy())
            pars=parameters(ok); pars.update({'Variable':var,'Height_m':h,'N_stations':len(d)}); all_params.append(pars)
            cv,st=loo_cv(df,col)
            st.update({'Variable':var,'Height_m':h,'N_stations':len(d),**pars}); all_summary.append(st)
            if not cv.empty:
                cv.insert(0,'Variable',var); cv.insert(1,'Height_m',h); all_cv.append(cv)
    pd.DataFrame(all_params).to_csv(OUT/'variogram_parameters.csv',index=False)
    pd.DataFrame(all_summary).to_csv(OUT/'kriging_cross_validation_summary.csv',index=False)
    pd.concat(all_cv,ignore_index=True).to_csv(OUT/'kriging_cross_validation_predictions.csv',index=False)
    with pd.ExcelWriter(OUT/'kriging_diagnostics.xlsx',engine='openpyxl') as w:
        pd.DataFrame(all_params).to_excel(w,index=False,sheet_name='variogram_parameters')
        pd.DataFrame(all_summary).to_excel(w,index=False,sheet_name='CV_summary')
        pd.concat(all_cv,ignore_index=True).to_excel(w,index=False,sheet_name='CV_predictions')
    print('\nDiagnostics saved to:',OUT.resolve())
    print(pd.DataFrame(all_summary).round(5).to_string(index=False))

if __name__=='__main__':
    try: main()
    except Exception as e: print(f'ERROR: {e}',file=sys.stderr); raise
