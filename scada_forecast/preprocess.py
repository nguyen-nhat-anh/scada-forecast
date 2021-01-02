import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import unidecode
import holidays
from lunarcalendar import Converter
from tqdm.auto import tqdm # note: tqdm version >= 4.8 only

def read_humidity(file_path):
    """
    Đọc dữ liệu độ ẩm từ file excel
    """
    df_humidity = pd.read_excel(file_path)
    df_humidity = df_humidity.pivot(index='kttv_thoidiem', columns='tenDoiTuong', values='kttv_doam')
    df_humidity = df_humidity.resample('H').mean().interpolate()
    df_humidity = df_humidity.add_prefix('Độ ẩm ')
    return df_humidity

def hour_rounder(t):
    """
    Rounds to nearest hour by adding a timedelta hour if minute >= 30
    """
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + dt.timedelta(hours=t.minute//30))

def read_temperature(file_path):
    """
    Đọc dữ liệu nhiệt độ từ file excel
    """
    df_temp = pd.read_excel(file_path, index_col=None, parse_dates=['Ngay'])
    df_temp = df_temp.drop(columns=['Quốc gia', 'Bắc', 'Trung', 'Nam'])
    df_temp['Ngay'] = df_temp['Ngay'].apply(hour_rounder)
    df_temp.set_index('Ngay', inplace=True)
    df_temp = df_temp.add_prefix('Nhiệt độ ')
    return df_temp

def create_date_col(row):
    """
    Tạo cột `date` cho dữ liệu scada
    """
    date = datetime.strptime(str(row['Ngay']), '%Y%m%d').strftime('%Y-%m-%d')
    hour = str(row['Gio']).zfill(2)
    minute = str(row['Phut'])[1:].zfill(2)
    _datetime = f'{date} {hour}:{minute}:00'
    return _datetime

def interpolate_nan_and_outlier(df, column):
    """
    Nội suy các dữ liệu thiếu và dữ liệu bất thường của cột trong dataframe
    """
    _df = df.copy()
    
    q75, q25 = np.nanpercentile(_df[column], [75, 25])
    iqr = q75 - q25
    
    _df[(_df[column] < q25 - 1.5 * iqr) | (_df[column] > q75 + 1.5 * iqr)] = np.nan # set nan for outliers
    _df[column] = _df[column].interpolate() # interpolate nan values
    return _df

def read_scada(file_path):
    """
    Đọc dữ liệu phụ tải điện scada từ file excel
    """
    df_scada = pd.read_excel(file_path, sheet_name='SCADA_QGia')
    minute_cols = [col for col in df_scada.columns if col not in ['Ngay', 'Gio']]
    df_scada = pd.melt(df_scada, id_vars=['Ngay', 'Gio'], value_vars=minute_cols, var_name='Phut', value_name='Load')
    df_scada['Date'] = df_scada.apply(create_date_col, axis=1)
    df_scada = df_scada[['Date', 'Load']].copy()
    df_scada['Date']= pd.to_datetime(df_scada['Date'])
    df_scada = df_scada.sort_values(by='Date', ignore_index=True)
    df_scada.set_index('Date', inplace=True)
    
    df_scada = interpolate_nan_and_outlier(df_scada, column='Load')
    df_scada = df_scada.resample('5min').mean().interpolate()
    return df_scada

def merge_dataframes(df_scada, df_temp=None, df_humidity=None):
    """
    Gộp dữ liệu scada với dữ liệu nhiệt độ, độ ẩm (nếu có)
    """
    if (df_temp is None) and (df_humidity is None):
        return df_scada
    
    if (df_temp is not None) and (df_humidity is not None):
        df = pd.merge(df_humidity, df_temp, how='inner', left_index=True, right_index=True)
    elif df_temp is not None:
        df = df_temp
    elif df_humidity is not None:
        df = df_humidity
    df = df.resample('5min').mean().interpolate()
    return pd.merge(df, df_scada, how='inner', left_index=True, right_index=True)

def get_holiday(t):
    """
    Trích xuất ngày nghỉ lễ từ date
    """
    date = t.date()
    holiday = holidays.VN().get(date)
    if holiday is None:
        holiday = "No"
    return holiday

def get_lunar_calendar_features(t):
    """
    Trích xuất các feature âm lịch (tháng âm, ngày âm, tháng nhuận) từ date
    """
    date = t.date()
    lunar = Converter.Solar2Lunar(date)
    return pd.Series({'LunarMonth': lunar.month,
                      'LunarDayOfMonth': lunar.day, 
                      'LeapMonth': lunar.isleap})

def add_calendar_features(df, use_lunar=True, use_holiday=True):
    """
    Thêm các feature giờ, thứ, tháng, ngày nghỉ lễ, lịch âm, v.v. vào dataframe
    """
    df['Hour'] = df.index.to_series().dt.hour
    df['DayOfWeek'] = df.index.to_series().dt.dayofweek
    df['Month'] = df.index.to_series().dt.month
    df['DayOfYear'] = df.index.to_series().dt.dayofyear
    df['Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int) # Saturday or Sunday
    
    if use_lunar or use_holiday:
        tqdm.pandas()
    
    if use_lunar:
        lunar_features = df.index.to_series().progress_apply(get_lunar_calendar_features)
        df = df.merge(lunar_features, left_index=True, right_index=True)
        df['LeapMonth'] = df['LeapMonth'].astype(int)
        
    if use_holiday:
        df['Holiday'] = df.index.to_series().progress_apply(get_holiday) # Running this will take a lot of time
        df['IsHoliday'] = (df['Holiday'] != 'No').astype(int)
    return df

def prepare_data(df, hour_steps, use_temp=True, use_humidity=True, use_lunar=True, use_holiday=True):
    """
    Chuẩn bị các feature để đưa vào mô hình
    """
    df['LogLoad'] = np.log(df['Load'])
    
    # Lag features
    df['Lag1h'] = df['LogLoad'].shift(hour_steps)
    df['Lag1d'] = df['LogLoad'].shift(24 * hour_steps)
    df['Lag1w'] = df['LogLoad'].shift(7 * 24 * hour_steps)

    df['Lag1h_Rmean1h'] = df['Lag1h'].rolling(hour_steps).mean()
    df['Lag1d_Rmean1h'] = df['Lag1d'].rolling(hour_steps).mean()
    df['Lag1w_Rmean1h'] = df['Lag1w'].rolling(hour_steps).mean()

    df['Lag1h_Rmax1h'] = df['Lag1h'].rolling(hour_steps).max()
    df['Lag1d_Rmax1h'] = df['Lag1d'].rolling(hour_steps).max()
    df['Lag1w_Rmax1h'] = df['Lag1w'].rolling(hour_steps).max()
    
    # Features lists
    calendar_features = ['Hour', 'DayOfWeek', 'Month', 'DayOfYear', 'Weekend']
    lag_features = ['Lag1h', 'Lag1d', 'Lag1w', 
                    'Lag1h_Rmean1h', 'Lag1d_Rmean1h', 'Lag1w_Rmean1h', 
                    'Lag1h_Rmax1h', 'Lag1d_Rmax1h', 'Lag1w_Rmax1h']
    humidity_features = []
    temperature_features = []
    load_features = ['Load', 'LogLoad']
    
    # Holiday
    if use_holiday:
        df['HolidayCat'] = df['Holiday'].map({
            "No": 0,
            "International New Year's Day": 1, 
            "Vietnamese New Year's Eve": 2,
            "Vietnamese New Year": 3, 
            "The second day of Tet Holiday": 4,
            "The third day of Tet Holiday": 5,
            "The forth day of Tet Holiday": 6,
            "The fifth day of Tet Holiday": 7, 
            "Hung Kings Commemoration Day": 8,
            "Liberation Day/Reunification Day": 9,
            "International Labor Day": 10,
            "Independence Day": 11
        })
        df.drop(columns=['Holiday'], inplace=True)
        calendar_features += ['HolidayCat', 'IsHoliday']
    
    # Lunar
    if use_lunar:
        calendar_features += ['LunarMonth', 'LunarDayOfMonth', 'LeapMonth']
        
    # Humidity
    if use_humidity:
        humidity_features += [col for col in df if col.startswith('Độ ẩm')]
        for feature in humidity_features:
            df[feature] /= 100.0
        
        df.rename(columns={s: unidecode.unidecode(s).replace(' ', '').replace('(', '').replace(')', '')
                           for s in humidity_features}, inplace=True)
        humidity_features = [unidecode.unidecode(s).replace(' ', '').replace('(', '').replace(')', '')
                             for s in humidity_features]
        
    # Temperature
    if use_temp:
        temperature_features += [col for col in df if col.startswith('Nhiệt độ')]
        
        df.rename(columns={s: unidecode.unidecode(s).replace(' ', '').replace('(', '').replace(')', '')
                           for s in temperature_features}, inplace=True)
        temperature_features = [unidecode.unidecode(s).replace(' ', '').replace('(', '').replace(')', '') 
                                for s in temperature_features]
        
    # Result
    df = df[temperature_features + humidity_features + lag_features + calendar_features + load_features].copy()
    categorical_cols = list(set(calendar_features) - set(['LeapMonth', 'IsHoliday', 'Weekend']))
    numeric_cols = list(set(temperature_features + humidity_features + lag_features + calendar_features) - set(categorical_cols))
    target_col = 'LogLoad'
    
    features_dict = {
        'calendar_features': calendar_features,
        'lag_features': lag_features,
        'humidity_features': humidity_features,
        'temperature_features': temperature_features,
        'load_features': load_features
    }
        
    return df, features_dict, categorical_cols, numeric_cols, target_col