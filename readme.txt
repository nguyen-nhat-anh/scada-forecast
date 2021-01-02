Project Structure:
- data: luu cac file du lieu
- scada_forecast: luu source code
- tmp: luu checkpoint mo hinh


Example usage:

# train mo hinh (ngan gon)
python main.py --task train --use_humidity --use_temperature --use_lunar --ckpt_path tmp/ckpt
# train mo hinh (viet day du)
python main.py --task train --scada_path data/scada/Dữ liệu SCADA Phụ tải 26.08.2020.xlsx 
--use_humidity --humidity_path data/scada/DoAm.xlsx 
--use_temperature --temperature_path data/NhietDoQuaKhu.xlsx 
--use_lunar --use_holiday --n_val 28800 --n_test 28800 --ckpt_path tmp/ckpt

# du bao
python main.py --task inference --use_humidity --use_temperature --use_lunar --ckpt_path tmp/ckpt