import argparse
import numpy as np
import tensorflow as tf

from scada_forecast.preprocess import read_humidity, read_temperature, read_scada
from scada_forecast.preprocess import get_lag_features, merge_dataframes, add_calendar_features, prepare_data

from scada_forecast.model import train_model, inference

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_args():
    parser = argparse.ArgumentParser('Dự báo phụ tải SCADA trước 1 giờ')
    parser.add_argument('--task', type=str, default='train', help='train hay inference')
    parser.add_argument('--scada_path', type=str, default='data/scada/Dữ liệu SCADA Phụ tải 26.08.2020.xlsx', help='File excel scada')
    parser.add_argument('--scada_sheet', type=str, default='SCADA_QGia', help='Chọn sheet cho file excel scada')
    parser.add_argument('--use_humidity', action='store_true', default=False, help='Có sử dụng dữ liệu độ ẩm hay không')
    parser.add_argument('--humidity_path', type=str, default='data/scada/DoAm.xlsx', help='File excel độ ẩm')
    parser.add_argument('--use_temperature', action='store_true', default=False, help='Có sử dụng dữ liệu nhiệt độ hay không')
    parser.add_argument('--temperature_path', type=str, default='data/NhietDoQuaKhu.xlsx', help='File excel nhiệt độ')
    parser.add_argument('--use_lunar', action='store_true', default=False, help='Có sử dụng các feature âm lịch hay không')
    parser.add_argument('--use_holiday', action='store_true', default=False, help='Có sử dụng các feature ngày nghỉ lễ hay không')
    parser.add_argument('--ckpt_path', type=str, default='tmp/ckpt', help='Đường dẫn lưu model')
    # Flag for training
    parser.add_argument('--n_val', type=int, default=100 * 24 * 12, help='Số điểm dữ liệu dùng cho tập validation (default 100 ngày)')
    parser.add_argument('--n_test', type=int, default=100 * 24 * 12, help='Số điểm dữ liệu dùng cho tập test (default 100 ngày cuối)')
    # Flag for inference
    parser.add_argument('--forecast_horizon', type=int, default=1, 
                        help='Số điểm dự báo (bắt đầu từ thời điểm sau 1h đổ về trước) (default chỉ dự báo tại 1 thời điểm sau 1h)')
    
    args = parser.parse_args()
    return args
    
def main(args):
    df_humidity = None
    df_temperature = None
    df_scada = None
    hour_steps = int(60 / 5)
    input_width = hour_steps
    categorical_unique_values_dict = {
        'Month': list(range(1, 12 + 1)),
        'DayOfYear': list(range(1, 365 + 1)),
        'DayOfWeek': list(range(7)),
        'Hour': list(range(24)),
        'LunarMonth': list(range(1, 12 + 1)),
        'LunarDayOfMonth': list(range(1, 30 + 1)),
        'HolidayCat': list(range(12))
    }
    
    # TIEN XU LY
    print('Doc du lieu scada...')
    df_scada = read_scada(args.scada_path, sheet_name=args.scada_sheet)
    if args.task == 'train':
        df_scada = get_lag_features(df_scada, hour_steps, clip_df=False)
    if args.task == 'inference':
        df_scada = get_lag_features(df_scada, hour_steps, clip_df=True, 
                                    input_width=input_width, forecast_horizon=args.forecast_horizon)
    
    if args.use_humidity:
        print('Doc du lieu do am...')
        df_humidity = read_humidity(args.humidity_path)
    if args.use_temperature:
        print('Doc du lieu nhiet do...')
        df_temperature = read_temperature(args.temperature_path)
        
    df = merge_dataframes(df_scada, df_temp=df_temperature, df_humidity=df_humidity)
    print('Them cac feature thoi gian (gio, thu, ngay, thang)...')
    df = add_calendar_features(df, use_lunar=args.use_lunar, use_holiday=args.use_holiday)
    
    print('Chuan bi du lieu cho mo hinh...')
    df, features_dict, categorical_cols, numeric_cols, target_col = prepare_data(df, 
                                                                                 train_flag=(args.task=='train'),
                                                                                 use_temp=args.use_temperature, 
                                                                                 use_humidity=args.use_humidity, 
                                                                                 use_lunar=args.use_lunar,
                                                                                 use_holiday=args.use_holiday)
    
    dtype_dict = {col: df[col].dtype for col in categorical_cols + numeric_cols}
    chosen_features = (features_dict['calendar_features'] + features_dict['lag_features'] + 
                       features_dict['temperature_features'] + features_dict['humidity_features'])
    
    # TRAIN MO HINH
    if args.task == 'train':
        print('Training...')
        train_model(df, input_width, args.n_val, args.n_test, hour_steps,
                    dtype_dict, categorical_unique_values_dict, 
                    chosen_features, categorical_cols, 
                    numeric_cols, target_col, args.ckpt_path)
        print('Hoan thanh.')
        
    # DU BAO
    if args.task == 'inference':
        print('Inference...')
        forecast_df = inference(df, input_width, args.ckpt_path,
                                dtype_dict, categorical_unique_values_dict, 
                                chosen_features, categorical_cols, numeric_cols)
        print(forecast_df)
    
    
if __name__ == '__main__':
    args = get_args()
    main(args)