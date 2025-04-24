from flask import Flask, request, render_template, redirect, url_for, flash, send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os
import time
import json
import re
import logging
from dotenv import load_dotenv
import uuid
import pandasql as ps
from fuzzywuzzy import process
import networkx as nx
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')  # Fallback if not in .env
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# SQL Server connection string
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')

# Directories
UPLOAD_FOLDER = '/home/ubuntu/workflow/uploads'
OUTPUT_FOLDER = '/home/ubuntu/workflow/outputs'
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}

# Chunk size for large data processing
CHUNK_SIZE = 10000

# Pricing and limits
FREE_DAILY_LIMIT = 1000  # Dòng tối đa/ngày cho người dùng miễn phí
PRO_COST_PER_1000_ROWS = 0.01  # Chi phí cho 1000 dòng với người dùng Pro

# Track daily usage for free users
daily_usage = defaultdict(lambda: defaultdict(int))  # {user_id: {date: rows_processed}}
        
# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Clean old files
def clean_old_files(directory, max_age_days=7):
    """Xóa các tệp có tuổi lớn hơn max_age_days."""
    max_age_seconds = max_age_days * 24 * 60 * 60
    current_time = time.time()
    try:
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error cleaning old files in {directory}: {str(e)}")

# Generate unique filename
def generate_unique_filename(base_path, filename):
    """Tạo tên file duy nhất bằng cách thêm hậu tố (n)."""
    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    new_path = os.path.join(base_path, new_filename)
    
    while os.path.exists(new_path):
        new_filename = f"{base_name}({counter}){ext}"
        new_path = os.path.join(base_path, new_filename)
        counter += 1
    
    return new_filename

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, email, is_subscribed=False, balance=0.0):
        self.id = id
        self.email = email
        self.is_subscribed = is_subscribed
        self.balance = balance

@login_manager.user_loader
def load_user(user_id):
    """Tải thông tin người dùng từ cơ sở dữ liệu."""
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, is_subscribed, balance FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        if user_data:
            return User(user_data[0], user_data[1], user_data[2], user_data[3])
        return None
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}")
        return None
    finally:
        conn.close()

# Data processing utility functions
def allowed_file(filename):
    """Kiểm tra định dạng file hợp lệ."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def load_data(source_config, chunked=False):
    """Tải dữ liệu từ tệp (CSV, Excel, JSON)."""
    file_path = source_config['path']
    if not os.path.exists(file_path):
        raise Exception(f"Tệp không tồn tại: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if chunked and ext == '.csv':
            return pd.read_csv(file_path, chunksize=CHUNK_SIZE)
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        else:
            raise Exception("Định dạng tệp không được hỗ trợ")
    except Exception as e:
        raise Exception(f"Lỗi khi tải tệp: {str(e)}")

def save_data(df, dest_config):
    """Lưu dữ liệu vào tệp theo định dạng được chỉ định."""
    try:
        file_path = dest_config['path']
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            df.to_csv(file_path, index=False)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif ext == '.json':
            df.to_json(file_path, orient='records', lines=True)
        else:
            raise Exception("Định dạng tệp đầu ra không được hỗ trợ")
    except Exception as e:
        raise Exception(f"Lỗi khi lưu tệp: {str(e)}")

def check_columns(df, required_columns, removed_columns, trans_type, step):
    """Kiểm tra cột cần thiết có tồn tại trong DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            for prev_step, prev_trans in removed_columns:
                if col in prev_trans['columns']:
                    raise Exception(
                        f"Cột '{col}' không tồn tại trong biến đổi '{trans_type}' ở bước {step + 1} "
                        f"vì đã bị xóa bởi biến đổi '{prev_trans['type']}' ở bước {prev_step + 1}"
                    )
        raise Exception(
            f"Các cột không tồn tại trong biến đổi '{trans_type}' ở bước {step + 1}: {', '.join(missing_cols)}"
        )

def simple_levenshtein(s1, s2):
    """Tính khoảng cách Levenshtein đơn giản."""
    if len(s1) < len(s2):
        return simple_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Transformation functions
def apply_select_columns(df, config, step, removed_columns):
    """Chọn các cột được chỉ định."""
    columns = [c.strip() for c in config['columns'].split(',')]
    check_columns(df, columns, removed_columns, 'select_columns', step)
    return df[columns]

def apply_filter(df, config, step, removed_columns):
    """Lọc dữ liệu theo điều kiện."""
    condition = config['condition']
    words = re.findall(r'\b\w+\b', condition)
    required_columns = [col for col in words if col in df.columns]
    check_columns(df, required_columns, removed_columns, 'filter', step)
    return df.query(condition)

def apply_python_script(df, config, step, removed_columns):
    """Thực thi mã Python tùy chỉnh."""
    code = config['code']
    if any(kw in code for kw in ['__import__', 'os.', 'sys.', 'eval', 'exec']):
        raise Exception("Mã Python chứa từ khóa bị hạn chế")
    local_vars = {'df': df}
    exec(code, {'__builtins__': {}}, local_vars)
    return local_vars['df']

def apply_remove_nulls(df, config, step, removed_columns):
    """Xóa các hàng có giá trị null trong cột được chỉ định."""
    columns = [c.strip() for c in config['columns'].split(',')]
    check_columns(df, columns, removed_columns, 'remove_nulls', step)
    return df.dropna(subset=columns)

def apply_deduplicate(df, config, step, removed_columns):
    """Xóa các hàng trùng lặp dựa trên cột được chỉ định."""
    columns = [c.strip() for c in config['columns'].split(',')]
    check_columns(df, columns, removed_columns, 'deduplicate', step)
    return df.drop_duplicates(subset=columns)

def apply_replace_values(df, config, step, removed_columns):
    """Thay thế giá trị trong cột được chỉ định."""
    column = config['column']
    check_columns(df, [column], removed_columns, 'replace_values', step)
    df[column] = df[column].replace(config['old_value'], config['new_value'])
    return df

def apply_join(df, config, step, removed_columns):
    """Thực hiện join với tệp khác."""
    join_df = load_data({'path': config['path']})
    join_keys = config['join_key'].split(',')
    check_columns(df, join_keys, removed_columns, 'join', step)
    for key in join_keys:
        if key not in join_df.columns:
            raise Exception(f"Khóa join không tồn tại trong tệp join: {key}")
    return df.merge(join_df, on=join_keys, how=config['join_type'])

def apply_aggregate(df, config, step, removed_columns):
    """Tổng hợp dữ liệu theo nhóm."""
    group_by = [c.strip() for c in config['group_by'].split(',')]
    agg_column = config['agg_column']
    check_columns(df, group_by + [agg_column], removed_columns, 'aggregate', step)
    return df.groupby(group_by)[agg_column].agg(config['agg_func']).reset_index()

def apply_pivot(df, config, step, removed_columns):
    """Tạo bảng pivot."""
    index = config['pivot_index'].split(',')
    columns = config['pivot_columns'].split(',')
    values = [config['pivot_values']]
    check_columns(df, index + columns + values, removed_columns, 'pivot', step)
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=config['pivot_aggfunc']
    ).reset_index()

def apply_drop_columns(df, config, step, removed_columns):
    """Xóa các cột được chỉ định."""
    columns = [c.strip() for c in config['columns'].split(',')]
    check_columns(df, columns, removed_columns, 'drop_columns', step)
    valid_cols = [col for col in columns if col in df.columns]
    if valid_cols:
        removed_columns.append((step, {'type': 'drop_columns', 'columns': valid_cols}))
        return df.drop(columns=valid_cols), removed_columns
    return df, removed_columns

def apply_split_records(df, config, user_output_dir, step, removed_columns):
    """Chia dữ liệu thành nhiều tệp dựa trên số dòng tối đa."""
    max_rows = int(config.get('max_rows', 1000))
    output_prefix = config.get('output_prefix', 'split_')
    condition = config.get('condition', None)
    output_files = []

    if condition:
        words = re.findall(r'\b\w+\b', condition)
        required_columns = [col for col in words if col in df.columns]
        check_columns(df, required_columns, removed_columns, 'split_records', step)
        temp_df = df.query(condition)
        remaining_df = df[~df.index.isin(temp_df.index)]
    else:
        temp_df = df
        remaining_df = pd.DataFrame()

    num_splits = max(1, (len(temp_df) + max_rows - 1) // max_rows)
    for i in range(num_splits):
        split_df = temp_df[i * max_rows:(i + 1) * max_rows]
        if not split_df.empty:
            split_filename = generate_unique_filename(user_output_dir, f"{output_prefix}{i+1}.csv")
            split_path = os.path.join(user_output_dir, split_filename)
            split_df.to_csv(split_path, index=False)
            output_files.append(split_path)

    return remaining_df if not remaining_df.empty else df, output_files

def apply_merge_records(df, config, step, removed_columns):
    """Hợp nhất dữ liệu từ nhiều tệp."""
    input_files = config['input_files'].split(',')
    merge_type = config.get('merge_type', 'concat')
    join_key = config.get('join_key', None)

    dfs = [load_data({'path': f}) for f in input_files]
    if merge_type == 'concat':
        return pd.concat(dfs, ignore_index=True)
    elif merge_type == 'merge':
        if not join_key:
            raise Exception("Cần chỉ định cột khóa cho merge")
        join_keys = join_key.split(',')
        check_columns(df, join_keys, removed_columns, 'merge_records', step)
        for d in dfs:
            for key in join_keys:
                if key not in d.columns:
                    raise Exception(f"Khóa join không tồn tại: {key}")
        result = dfs[0]
        for other_df in dfs[1:]:
            result = result.merge(other_df, on=join_keys, how='outer')
        return result

def apply_route_on_attribute(df, config, user_output_dir, step, removed_columns):
    """Định tuyến dữ liệu dựa trên giá trị cột."""
    column = config['route_column']
    check_columns(df, [column], removed_columns, 'route_on_attribute', step)
    routes = json.loads(config['routes'])
    output_files = []

    for value, filename in routes.items():
        route_df = df[df[column] == value]
        if not route_df.empty:
            route_filename = generate_unique_filename(user_output_dir, filename)
            route_path = os.path.join(user_output_dir, route_filename)
            route_df.to_csv(route_path, index=False)
            output_files.append(route_path)
    return df[~df[column].isin(routes.keys())], output_files

def apply_enrich_data(df, config, step, removed_columns):
    """Bổ sung dữ liệu từ tệp khác."""
    enrich_df = load_data({'path': config['enrich_file']})
    join_key = config['join_key'].split(',')
    columns = config['columns'].split(',')
    check_columns(df, join_key, removed_columns, 'enrich_data', step)

    for key in join_key:
        if key not in enrich_df.columns:
            raise Exception(f"Khóa join không tồn tại trong tệp bổ sung: {key}")
    for col in columns:
        if col not in enrich_df.columns:
            raise Exception(f"Cột không tồn tại trong tệp bổ sung: {col}")

    return df.merge(enrich_df[join_key + columns], on=join_key, how='left')

def apply_convert_format(df, config, user_output_dir, step, removed_columns):
    """Chuyển đổi định dạng tệp."""
    temp_format = config['format']
    temp_filename = generate_unique_filename(user_output_dir, f"temp_{uuid.uuid4().hex}.{temp_format}")
    temp_path = os.path.join(user_output_dir, temp_filename)

    if temp_format == 'csv':
        df.to_csv(temp_path, index=False)
    elif temp_format == 'xlsx':
        df.to_excel(temp_path, index=False)
    elif temp_format == 'json':
        df.to_json(temp_path, orient='records', lines=True)
    else:
        raise Exception("Định dạng không được hỗ trợ")

    return load_data({'path': temp_path}), [temp_path]

def apply_replace_text(df, config, step, removed_columns):
    """Thay thế văn bản bằng regex."""
    column = config['column']
    check_columns(df, [column], removed_columns, 'replace_text', step)
    df[column] = df[column].astype(str).str.replace(
        config['pattern'], config['replacement'], regex=True)
    return df

def apply_execute_sql(df, config, step, removed_columns):
    """Thực thi truy vấn SQL trên DataFrame."""
    query = config['query'].lower()
    words = re.findall(r'\b\w+\b', query)
    required_columns = [col for col in words if col in df.columns]
    check_columns(df, required_columns, removed_columns, 'execute_sql', step)
    try:
        return ps.sqldf(query, locals())
    except Exception as e:
        raise Exception(f"Lỗi truy vấn SQL: {str(e)}")

def apply_branching(df, config, user_output_dir, step, removed_columns):
    """Phân nhánh dữ liệu dựa trên điều kiện."""
    condition = config['condition']
    words = re.findall(r'\b\w+\b', condition)
    required_columns = [col for col in words if col in df.columns]
    check_columns(df, required_columns, removed_columns, 'branching', step)

    true_filename = generate_unique_filename(user_output_dir, 'true_branch.csv')
    false_filename = generate_unique_filename(user_output_dir, 'false_branch.csv')
    true_path = os.path.join(user_output_dir, true_filename)
    false_path = os.path.join(user_output_dir, false_filename)
    output_files = []

    true_df = df.query(condition)
    false_df = df[~df.index.isin(true_df.index)]

    if not true_df.empty:
        true_df.to_csv(true_path, index=False)
        output_files.append(true_path)
    if not false_df.empty:
        false_df.to_csv(false_path, index=False)
        output_files.append(false_path)

    return pd.DataFrame(), output_files

def apply_external_task_sensor(config):
    """Cảm biến tác vụ bên ngoài (mock)."""
    task_id = config['task_id']
    timeout = int(config.get('timeout', 300))
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise Exception(f"Timeout waiting for task {task_id}")
        time.sleep(1)
        break  # Mock: Thoát ngay
    logger.info(f"Task {task_id} detected")

def apply_email_notification(config):
    """Gửi email thông báo (mock)."""
    recipient = config['recipient']
    subject = config['subject']
    body = config['body']
    logger.info(f"Mock email sent to {recipient}: Subject: {subject}, Body: {body}")

def apply_file_sensor(config):
    """Cảm biến tệp (mock)."""
    file_path = config['file_path']
    timeout = int(config.get('timeout', 300))
    start_time = time.time()

    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise Exception(f"Timeout waiting for file {file_path}")
        time.sleep(1)
    logger.info(f"File {file_path} detected")

def apply_data_quality_check(df, config, step, removed_columns):
    """Kiểm tra chất lượng dữ liệu."""
    rules = json.loads(config['rules'])
    for rule, cols in rules.items():
        for col in cols:
            check_columns(df, [col], removed_columns, 'data_quality_check', step)
            if col not in df.columns:
                raise Exception(f"Cột '{col}' không tồn tại trong Data Quality Check")
            if rule == 'no_nulls' and df[col].isna().any():
                raise Exception(f"Cột '{col}' chứa giá trị null")
            elif rule == 'unique' and df[col].duplicated().any():
                raise Exception(f"Cột '{col}' chứa giá trị trùng lặp")
    return df

def apply_dynamic_split(df, config, user_output_dir, step, removed_columns):
    """Chia dữ liệu động dựa trên giá trị cột."""
    column = config['split_key']
    check_columns(df, [column], removed_columns, 'dynamic_split', step)
    output_prefix = config.get('output_prefix', 'split_')
    output_files = []

    unique_values = df[column].unique()
    for value in unique_values:
        split_df = df[df[column] == value]
        if not split_df.empty:
            safe_value = str(value).replace('/', '_').replace('\\', '_')
            split_filename = generate_unique_filename(
                user_output_dir, f"{output_prefix}{safe_value}.csv"
            )
            split_path = os.path.join(user_output_dir, split_filename)
            split_df.to_csv(split_path, index=False)
            output_files.append(split_path)

    return pd.DataFrame(), output_files

def apply_normalize_data(df, config, step, removed_columns):
    """Chuẩn hóa dữ liệu."""
    columns = [c.strip() for c in config['columns'].split(',')]
    check_columns(df, columns, removed_columns, 'normalize_data', step)
    method = config['method']
    for column in columns:
        if method == 'min_max':
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        elif method == 'z_score':
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def apply_fuzzy_match(df, config, step, removed_columns):
    """So khớp mờ (fuzzy matching)."""
    source_column = config['column']
    check_columns(df, [source_column], removed_columns, 'fuzzy_match', step)
    reference_df = load_data({'path': config['reference_file']})
    reference_column = config['reference_column']
    output_column = config['output_column']
    threshold = float(config['threshold'])

    if reference_column not in reference_df.columns:
        raise Exception(f"Cột tham chiếu '{reference_column}' không tồn tại")

    try:
        choices = reference_df[reference_column].astype(str).tolist()
        df[output_column] = df[source_column].astype(str).apply(
            lambda x: process.extractOne(x, choices, score_cutoff=threshold)[0] if x else ''
        )
    except:
        choices = reference_df[reference_column].astype(str).tolist()
        df[output_column] = df[source_column].astype(str).apply(
            lambda x: min(choices, key=lambda c: simple_levenshtein(x, c)) if x else ''
        )
    return df

def apply_data_masking(df, config, step, removed_columns):
    """Che giấu dữ liệu."""
    columns = [c.strip() for c in config['columns'].split(',')]
    check_columns(df, columns, removed_columns, 'data_masking', step)
    method = config['method']
    for column in columns:
        if method == 'hash':
            df[column] = df[column].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:10])
        elif method == 'mask':
            df[column] = df[column].astype(str).apply(lambda x: '*' * len(x))
    return df

def apply_json_flatten(df, config, step, removed_columns):
    """Làm phẳng dữ liệu JSON."""
    column = config['json_column']
    check_columns(df, [column], removed_columns, 'json_flatten', step)
    prefix = config.get('prefix', column)

    json_data = df[column].apply(json.loads)
    flattened = pd.json_normalize(json_data)
    flattened.columns = [f"{prefix}_{col}" for col in flattened.columns]
    return pd.concat([df.drop(columns=[column]), flattened], axis=1)

def apply_stream_filter(df, config, step, removed_columns):
    """Lọc dữ liệu theo luồng."""
    condition = config['condition']
    words = re.findall(r'\b\w+\b', condition)
    required_columns = [col for col in words if col in df.columns]
    check_columns(df, required_columns, removed_columns, 'stream_filter', step)
    return df.query(condition)

def apply_http_enrich(df, config, step, removed_columns):
    """Bổ sung dữ liệu từ HTTP (mock)."""
    column = config['key_column']
    check_columns(df, [column], removed_columns, 'http_enrich', step)
    url = config['url']
    output_columns = config.get('output_columns', '').split(',')

    mock_response = {val: {col: f"{col}_{val}" for col in output_columns} for val in df[column].unique()}
    for col in output_columns:
        df[col] = df[column].map(lambda x: mock_response.get(x, {}).get(col, ''))
    return df

def apply_window_function(df, config, step, removed_columns):
    """Áp dụng hàm cửa sổ."""
    function = config['function']
    partition_by = config['partition_by']
    check_columns(df, [partition_by], removed_columns, 'window_function', step)
    output_column = config.get('output_column', f"{function}_result")

    if function == 'rank':
        df[output_column] = df.groupby(partition_by)[partition_by].rank()
    elif function == 'row_number':
        df[output_column] = df.groupby(partition_by).cumcount() + 1
    elif function == 'cumsum':
        df[output_column] = df.groupby(partition_by)[partition_by].cumsum()
    return df

def apply_cross_join(df, config, step, removed_columns):
    """Thực hiện cross join."""
    join_df = load_data({'path': config['join_file']})
    df['__key'] = 1
    join_df['__key'] = 1
    result = df.merge(join_df, on='__key').drop(columns='__key')
    return result

def apply_conditional_split(df, config, user_output_dir, step, removed_columns):
    """Chia dữ liệu theo nhiều điều kiện."""
    conditions = json.loads(config['conditions'])
    output_files = []
    remaining_df = df.copy()

    for cond in conditions:
        condition = cond['condition']
        words = re.findall(r'\b\w+\b', condition)
        required_columns = [col for col in words if col in df.columns]
        check_columns(df, required_columns, removed_columns, 'conditional_split', step)
        output_filename = generate_unique_filename(user_output_dir, cond['output'])
        output_path = os.path.join(user_output_dir, output_filename)
        split_df = remaining_df.query(condition)
        if not split_df.empty:
            split_df.to_csv(output_path, index=False)
            output_files.append(output_path)
        remaining_df = remaining_df[~remaining_df.index.isin(split_df.index)]

    return remaining_df, output_files

def apply_aggregate_multiple(df, config, step, removed_columns):
    """Tổng hợp nhiều cột cùng lúc."""
    group_by = config['group_by'].split(',')
    aggregations = json.loads(config['aggregations'])
    check_columns(df, group_by + [agg['column'] for agg in aggregations], removed_columns, 'aggregate_multiple', step)

    agg_dict = {agg['column']: [agg['func']] for agg in aggregations}
    result = df.groupby(group_by).agg(agg_dict).reset_index()
    result.columns = group_by + [agg['output_column'] for agg in aggregations]
    return result

# Validation and transformation orchestration
def validate_flow(flow):
    """Kiểm tra tính hợp lệ của flow: có input/output, không có vòng lặp."""
    nodes = flow['nodes']
    connections = flow['connections']
    
    # Kiểm tra input và output
    input_nodes = [n for n in nodes if n['type'] == 'input']
    output_nodes = [n for n in nodes if n['type'] == 'output']
    if not input_nodes or not output_nodes:
        raise Exception("Flow phải có ít nhất một node Input và một node Output")
    
    # Kiểm tra vòng lặp
    G = nx.DiGraph()
    for conn in connections:
        G.add_edge(conn['source'], conn['target'])
    
    try:
        nx.find_cycle(G, orientation='original')
        raise Exception("Flow chứa vòng lặp, không được phép")
    except nx.NetworkXNoCycle:
        pass
    
    # Validate node config
    for node in nodes:
        config = node.get('config', {})
        node_type = node['type']
        if node_type == 'input' and not config.get('path'):
            raise Exception(f"Node Input {node['id']} thiếu tệp đầu vào")
        elif node_type == 'output' and not config.get('path'):
            raise Exception(f"Node Output {node['id']} thiếu tệp đầu ra")
        elif node_type in ['select_columns', 'remove_nulls', 'deduplicate', 'drop_columns']:
            if not config.get('columns'):
                raise Exception(f"Node {node_type} {node['id']} thiếu cột")
            if not all(c.strip() for c in config['columns'].split(',')):
                raise Exception(f"Node {node_type} {node['id']} có cột không hợp lệ")
        elif node_type == 'filter':
            if not config.get('condition'):
                raise Exception(f"Node Filter {node['id']} thiếu điều kiện")
            if not isinstance(config['condition'], str) or not config['condition'].strip():
                raise Exception(f"Node Filter {node['id']} có điều kiện không hợp lệ")
        elif node_type == 'python_script':
            if not config.get('code'):
                raise Exception(f"Node Python Script {node['id']} thiếu mã")
            code = config['code']
            if any(kw in code for kw in ['__import__', 'os.', 'sys.', 'eval', 'exec']):
                raise Exception(f"Node Python Script {node['id']} chứa từ khóa bị hạn chế")
        elif node_type == 'replace_values':
            if not all(config.get(k) for k in ['column', 'old_value', 'new_value']):
                raise Exception(f"Node Replace Values {node['id']} thiếu thông tin")
        elif node_type == 'join':
            if not all(config.get(k) for k in ['path', 'join_key', 'join_type']):
                raise Exception(f"Node Join {node['id']} thiếu thông tin")
            if config['join_type'] not in ['left', 'right', 'inner', 'outer']:
                raise Exception(f"Node Join {node['id']} có kiểu join không hợp lệ")
        elif node_type == 'aggregate':
            if not all(config.get(k) for k in ['group_by', 'agg_column', 'agg_func']):
                raise Exception(f"Node Aggregate {node['id']} thiếu thông tin")
            if config['agg_func'] not in ['sum', 'mean', 'count', 'min', 'max']:
                raise Exception(f"Node Aggregate {node['id']} có hàm tổng hợp không hợp lệ")
        elif node_type == 'pivot':
            if not all(config.get(k) for k in ['pivot_index', 'pivot_columns', 'pivot_values', 'pivot_aggfunc']):
                raise Exception(f"Node Pivot {node['id']} thiếu thông tin")
        elif node_type == 'split_records':
            if not config.get('max_rows') or not config.get('output_prefix'):
                raise Exception(f"Node Split Records {node['id']} thiếu thông tin")
            if not str(config['max_rows']).isdigit() or int(config['max_rows']) <= 0:
                raise Exception(f"Node Split Records {node['id']} có max_rows không hợp lệ")
        elif node_type == 'merge_records':
            if not config.get('input_files'):
                raise Exception(f"Node Merge Records {node['id']} thiếu tệp đầu vào")
        elif node_type == 'route_on_attribute':
            if not config.get('route_column') or not config.get('routes'):
                raise Exception(f"Node Route on Attribute {node['id']} thiếu thông tin")
            try:
                routes = json.loads(config['routes'])
                if not isinstance(routes, dict):
                    raise Exception(f"Node Route on Attribute {node['id']} có routes không hợp lệ")
            except json.JSONDecodeError:
                raise Exception(f"Node Route on Attribute {node['id']} có routes không hợp lệ")
        elif node_type == 'enrich_data':
            if not all(config.get(k) for k in ['enrich_file', 'join_key', 'columns']):
                raise Exception(f"Node Enrich Data {node['id']} thiếu thông tin")
        elif node_type == 'convert_format':
            if not config.get('format'):
                raise Exception(f"Node Convert Format {node['id']} thiếu định dạng")
            if config['format'] not in ['csv', 'xlsx', 'json']:
                raise Exception(f"Node Convert Format {node['id']} có định dạng không hợp lệ")
        elif node_type == 'replace_text':
            if not all(config.get(k) for k in ['column', 'pattern', 'replacement']):
                raise Exception(f"Node Replace Text {node['id']} thiếu thông tin")
        elif node_type == 'execute_sql':
            if not config.get('query'):
                raise Exception(f"Node Execute SQL {node['id']} thiếu truy vấn")
        elif node_type == 'branching':
            if not config.get('condition'):
                raise Exception(f"Node Branching {node['id']} thiếu điều kiện")
        elif node_type == 'external_task_sensor':
            if not all(config.get(k) for k in ['task_id', 'timeout']):
                raise Exception(f"Node External Task Sensor {node['id']} thiếu thông tin")
        elif node_type == 'email_notification':
            if not all(config.get(k) for k in ['recipient', 'subject', 'body']):
                raise Exception(f"Node Email Notification {node['id']} thiếu thông tin")
        elif node_type == 'file_sensor':
            if not config.get('file_path'):
                raise Exception(f"Node File Sensor {node['id']} thiếu đường dẫn tệp")
        elif node_type == 'data_quality_check':
            if not config.get('rules'):
                raise Exception(f"Node Data Quality Check {node['id']} thiếu quy tắc")
            try:
                rules = json.loads(config['rules'])
                if not isinstance(rules, dict):
                    raise Exception(f"Node Data Quality Check {node['id']} có quy tắc không hợp lệ")
            except json.JSONDecodeError:
                raise Exception(f"Node Data Quality Check {node['id']} có quy tắc không hợp lệ")
        elif node_type == 'dynamic_split':
            if not all(config.get(k) for k in ['split_key', 'output_prefix']):
                raise Exception(f"Node Dynamic Split {node['id']} thiếu thông tin")
        elif node_type == 'normalize_data':
            if not all(config.get(k) for k in ['columns', 'method']):
                raise Exception(f"Node Normalize Data {node['id']} thiếu thông tin")
            if config['method'] not in ['min_max', 'z_score']:
                raise Exception(f"Node Normalize Data {node['id']} có phương pháp không hợp lệ")
        elif node_type == 'fuzzy_match':
            if not all(config.get(k) for k in ['column', 'threshold', 'reference_file', 'reference_column', 'output_column']):
                raise Exception(f"Node Fuzzy Match {node['id']} thiếu thông tin")
            if not str(config['threshold']).replace('.', '', 1).isdigit() or float(config['threshold']) < 0 or float(config['threshold']) > 100:
                raise Exception(f"Node Fuzzy Match {node['id']} có ngưỡng không hợp lệ")
        elif node_type == 'data_masking':
            if not all(config.get(k) for k in ['columns', 'method']):
                raise Exception(f"Node Data Masking {node['id']} thiếu thông tin")
            if config['method'] not in ['hash', 'mask']:
                raise Exception(f"Node Data Masking {node['id']} có phương pháp không hợp lệ")
        elif node_type == 'json_flatten':
            if not config.get('json_column'):
                raise Exception(f"Node JSON Flatten {node['id']} thiếu cột JSON")
        elif node_type == 'stream_filter':
            if not config.get('condition'):
                raise Exception(f"Node Stream Filter {node['id']} thiếu điều kiện")
        elif node_type == 'http_enrich':
            if not all(config.get(k) for k in ['url', 'key_column']):
                raise Exception(f"Node HTTP Enrich {node['id']} thiếu thông tin")
        elif node_type == 'window_function':
            if not all(config.get(k) for k in ['function', 'partition_by']):
                raise Exception(f"Node Window Function {node['id']} thiếu thông tin")
            if config['function'] not in ['rank', 'row_number', 'cumsum']:
                raise Exception(f"Node Window Function {node['id']} có hàm không hợp lệ")
        elif node_type == 'cross_join':
            if not config.get('join_file'):
                raise Exception(f"Node Cross Join {node['id']} thiếu tệp join")
        elif node_type == 'conditional_split':
            if not config.get('conditions'):
                raise Exception(f"Node Conditional Split {node['id']} thiếu điều kiện")
            try:
                conditions = json.loads(config['conditions'])
                if not isinstance(conditions, list):
                    raise Exception(f"Node Conditional Split {node['id']} có điều kiện không hợp lệ")
                for cond in conditions:
                    if not all(k in cond for k in ['condition', 'output']):
                        raise Exception(f"Node Conditional Split {node['id']} có điều kiện thiếu thông tin")
            except json.JSONDecodeError:
                raise Exception(f"Node Conditional Split {node['id']} có điều kiện không hợp lệ")
        elif node_type == 'aggregate_multiple':
            if not all(config.get(k) for k in ['group_by', 'aggregations']):
                raise Exception(f"Node Aggregate Multiple {node['id']} thiếu thông tin")
            try:
                aggregations = json.loads(config['aggregations'])
                if not isinstance(aggregations, list):
                    raise Exception(f"Node Aggregate Multiple {node['id']} có aggregations không hợp lệ")
                for agg in aggregations:
                    if not all(k in agg for k in ['column', 'func', 'output_column']):
                        raise Exception(f"Node Aggregate Multiple {node['id']} có aggregation thiếu thông tin")
                    if agg['func'] not in ['sum', 'mean', 'count', 'min', 'max']:
                        raise Exception(f"Node Aggregate Multiple {node['id']} có hàm không hợp lệ")
            except json.JSONDecodeError:
                raise Exception(f"Node Aggregate Multiple {node['id']} có aggregations không hợp lệ")
    
    return True

def get_transformation_order(flow):
    """Xác định thứ tự thực thi transformations dựa trên connections."""
    nodes = flow['nodes']
    connections = flow['connections']
    
    G = nx.DiGraph()
    for conn in connections:
        G.add_edge(conn['source'], conn['target'])
    
    input_nodes = [n['id'] for n in nodes if n['type'] == 'input']
    if not input_nodes:
        raise Exception("Không tìm thấy node Input")
    
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        raise Exception("Flow chứa vòng lặp hoặc không hợp lệ")
    
    trans_nodes = [
        n for n in nodes
        if n['type'] not in ['input', 'output']
        and n['id'] in topo_order
    ]
    
    ordered_nodes = []
    for node_id in topo_order:
        node = next((n for n in trans_nodes if n['id'] == node_id), None)
        if node:
            ordered_nodes.append(node)
    
    transformations = [
        {'type': node['type'], 'config': node['config']}
        for node in ordered_nodes
    ]
    
    return transformations

def apply_transformations(df, flow, user_id):
    """Áp dụng các biến đổi dữ liệu theo thứ tự trong flow."""
    output_files = []
    user_output_dir = os.path.join(OUTPUT_FOLDER, user_id)
    os.makedirs(user_output_dir, exist_ok=True)
    removed_columns = []
    
    transformations = get_transformation_order(flow)
    
    for step, trans in enumerate(transformations):
        trans_type = trans['type']
        config = trans['config']
        logger.info(f"Applying transformation {trans_type} at step {step + 1}")

        try:
            if trans_type == 'select_columns':
                df = apply_select_columns(df, config, step, removed_columns)
            elif trans_type == 'filter':
                df = apply_filter(df, config, step, removed_columns)
            elif trans_type == 'python_script':
                df = apply_python_script(df, config, step, removed_columns)
            elif trans_type == 'remove_nulls':
                df = apply_remove_nulls(df, config, step, removed_columns)
            elif trans_type == 'deduplicate':
                df = apply_deduplicate(df, config, step, removed_columns)
            elif trans_type == 'replace_values':
                df = apply_replace_values(df, config, step, removed_columns)
            elif trans_type == 'join':
                df = apply_join(df, config, step, removed_columns)
            elif trans_type == 'aggregate':
                df = apply_aggregate(df, config, step, removed_columns)
            elif trans_type == 'pivot':
                df = apply_pivot(df, config, step, removed_columns)
            elif trans_type == 'drop_columns':
                df, removed_columns = apply_drop_columns(df, config, step, removed_columns)
            elif trans_type == 'split_records':
                df, new_files = apply_split_records(df, config, user_output_dir, step, removed_columns)
                output_files.extend(new_files)
            elif trans_type == 'merge_records':
                df = apply_merge_records(df, config, step, removed_columns)
            elif trans_type == 'route_on_attribute':
                df, new_files = apply_route_on_attribute(df, config, user_output_dir, step, removed_columns)
                output_files.extend(new_files)
            elif trans_type == 'enrich_data':
                df = apply_enrich_data(df, config, step, removed_columns)
            elif trans_type == 'convert_format':
                df, new_files = apply_convert_format(df, config, user_output_dir, step, removed_columns)
                output_files.extend(new_files)
            elif trans_type == 'replace_text':
                df = apply_replace_text(df, config, step, removed_columns)
            elif trans_type == 'execute_sql':
                df = apply_execute_sql(df, config, step, removed_columns)
            elif trans_type == 'branching':
                df, new_files = apply_branching(df, config, user_output_dir, step, removed_columns)
                output_files.extend(new_files)
            elif trans_type == 'external_task_sensor':
                apply_external_task_sensor(config)
            elif trans_type == 'email_notification':
                apply_email_notification(config)
            elif trans_type == 'file_sensor':
                apply_file_sensor(config)
            elif trans_type == 'data_quality_check':
                df = apply_data_quality_check(df, config, step, removed_columns)
            elif trans_type == 'dynamic_split':
                df, new_files = apply_dynamic_split(df, config, user_output_dir, step, removed_columns)
                output_files.extend(new_files)
            elif trans_type == 'normalize_data':
                df = apply_normalize_data(df, config, step, removed_columns)
            elif trans_type == 'fuzzy_match':
                df = apply_fuzzy_match(df, config, step, removed_columns)
            elif trans_type == 'data_masking':
                df = apply_data_masking(df, config, step, removed_columns)
            elif trans_type == 'json_flatten':
                df = apply_json_flatten(df, config, step, removed_columns)
            elif trans_type == 'stream_filter':
                df = apply_stream_filter(df, config, step, removed_columns)
            elif trans_type == 'http_enrich':
                df = apply_http_enrich(df, config, step, removed_columns)
            elif trans_type == 'window_function':
                df = apply_window_function(df, config, step, removed_columns)
            elif trans_type == 'cross_join':
                df = apply_cross_join(df, config, step, removed_columns)
            elif trans_type == 'conditional_split':
                df, new_files = apply_conditional_split(df, config, user_output_dir, step, removed_columns)
                output_files.extend(new_files)
            elif trans_type == 'aggregate_multiple':
                df = apply_aggregate_multiple(df, config, step, removed_columns)
            else:
                raise Exception(f"Loại biến đổi không được hỗ trợ: {trans_type}")
        except Exception as e:
            raise Exception(f"Lỗi trong biến đổi '{trans_type}' ở bước {step + 1}: {str(e)}")
    
    return df, output_files

def calculate_cost(rows_processed, is_subscribed):
    """Tính chi phí dựa trên số dòng xử lý."""
    if is_subscribed:
        # Người dùng Pro: 0.01 USD cho mỗi 1000 dòng
        cost = (rows_processed / 1000) * PRO_COST_PER_1000_ROWS
        return round(cost, 6)
    else:
        return 0.0  # Người dùng miễn phí không mất phí, chỉ kiểm tra giới hạn dòng

def check_daily_limit(user_id, rows_processed, is_subscribed):
    """Kiểm tra giới hạn hàng ngày cho người dùng miễn phí."""
    if is_subscribed:
        return True, None  # Người dùng Pro không có giới hạn hàng ngày

    today = datetime.now().strftime('%Y-%m-%d')
    
    # Lấy số dòng đã xử lý trong ngày từ database
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT SUM(rows_processed)
            FROM logs l
            JOIN workflows w ON l.workflow_id = w.id
            WHERE w.user_id = ? AND CAST(l.timestamp AS DATE) = ?
            AND l.status = 'Success'
        ''', (user_id, today))
        rows_used = cursor.fetchone()[0] or 0
        conn.close()
    except Exception as e:
        logger.error(f"Error checking daily limit: {str(e)}")
        return False, "Lỗi khi kiểm tra giới hạn hàng ngày."

    total_rows = rows_used + rows_processed
    if total_rows > FREE_DAILY_LIMIT:
        return False, f"Người dùng miễn phí chỉ được xử lý tối đa {FREE_DAILY_LIMIT} dòng mỗi ngày! Bạn đã sử dụng {rows_used} dòng."
    return True, None

def execute_workflow(workflow_id, user_id, source_file, dest_file, flow):
    """Thực thi workflow và trừ tiền từ ví."""
    errors = []
    log_id = None
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM workflows WHERE id = ? AND user_id = ?', (workflow_id, user_id))
        workflow = cursor.fetchone()
        cursor.execute('SELECT is_subscribed, balance FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        if not workflow:
            errors.append("Workflow không tồn tại hoặc bạn không có quyền truy cập")
            return False, errors

        # Load data to get number of rows
        df = load_data({'path': source_file})
        rows_processed = len(df)

        # Check daily limit for free users
        limit_ok, limit_error = check_daily_limit(user_id, rows_processed, user[0])
        if not limit_ok:
            errors.append(limit_error)
            return False, errors

        # Calculate cost and check balance for Pro users
        cost = calculate_cost(rows_processed, user[0])
        if user[0]:  # Người dùng Pro
            if user[1] < cost:
                errors.append(f"Số dư ví không đủ! Cần {cost:.6f} USD, hiện có {user[1]:.6f} USD.")
                return False, errors

        # Create log entry with cost set to 0 initially
        log_entry = {
            'workflow_id': workflow_id,
            'timestamp': datetime.now(),
            'status': 'Running',
            'rows_processed': 0,
            'error': None,
            'cost': 0,  # Khởi tạo cost là 0
            'source_file': source_file,
            'dest_file': dest_file
        }

        log_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO logs (id, workflow_id, timestamp, status, rows_processed, error, cost, source_file, dest_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (log_id, workflow_id, log_entry['timestamp'], log_entry['status'], log_entry['rows_processed'],
              log_entry['error'], log_entry['cost'], log_entry['source_file'], log_entry['dest_file']))
        conn.commit()
        conn.close()

        # Apply transformations
        df, output_files = apply_transformations(df, flow, user_id)

        # Save output if data is not empty
        if not df.empty:
            save_data(df, {'path': dest_file})

        # Update database with success status, actual cost, and output files
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE logs
            SET status = ?, rows_processed = ?, cost = ?, dest_file = ?
            WHERE id = ?
        ''', ('Success', rows_processed, cost, ','.join([dest_file] + output_files), log_id))
        if user[0]:  # Chỉ trừ tiền người dùng Pro
            cursor.execute('''
                UPDATE users
                SET balance = balance - ?
                WHERE id = ?
            ''', (cost, user_id))
            current_user.balance -= cost
        cursor.execute('UPDATE workflows SET status = ? WHERE id = ?', ('Completed', workflow_id))
        conn.commit()
        conn.close()

        return True, ["Workflow thực thi thành công"] + [f"Tạo file bổ sung: {f}" for f in output_files]
    except Exception as e:
        if log_id:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            # Update log with failed status, cost = 0, and dest_file = NULL
            cursor.execute('''
                UPDATE logs
                SET status = ?, error = ?, cost = ?, dest_file = ?
                WHERE id = ?
            ''', ('Failed', str(e), 0, 'X', log_id))  # Đặt dest_file = X khi failed
            cursor.execute('UPDATE workflows SET status = ? WHERE id = ?', ('Failed', workflow_id))
            conn.commit()
            conn.close()
        errors.append(str(e))
        return False, errors

# Routes
@app.route('/')
def index():
    """Trang chủ."""
    logger.debug("Accessing index page")
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Đăng ký người dùng mới."""
    logger.debug("Accessing register page")
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        
        if not re.match(email_regex, email):
            flash('Vui lòng nhập email hợp lệ!', 'error')
            logger.warning(f"Invalid email format: {email}")
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Mật khẩu phải có ít nhất 6 ký tự!', 'error')
            logger.warning("Password too short")
            return redirect(url_for('register'))
        
        try:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                flash('Email đã tồn tại!', 'error')
                logger.warning(f"Email already exists: {email}")
                return redirect(url_for('register'))
            
            user_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO users (id, email, password, is_subscribed, balance)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, email, generate_password_hash(password), False, 0.0))
            conn.commit()
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            logger.info(f"User registered: {email}")
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            flash('Lỗi khi đăng ký. Vui lòng thử lại.', 'error')
            return redirect(url_for('register'))
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Đăng nhập người dùng."""
    logger.debug("Accessing login page")
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        
        if not re.match(email_regex, email):
            flash('Vui lòng nhập email hợp lệ!', 'error')
            logger.warning(f"Invalid email format: {email}")
            return redirect(url_for('login'))
        
        try:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute('SELECT id, email, password, is_subscribed, balance FROM users WHERE email = ?', (email,))
            user_data = cursor.fetchone()
            if user_data and check_password_hash(user_data[2], password):
                user = User(user_data[0], user_data[1], user_data[3], user_data[4])
                login_user(user)
                flash('Đăng nhập thành công!', 'success')
                logger.info(f"User logged in: {email}")
                return redirect(url_for('dashboard'))
            else:
                flash('Email hoặc mật khẩu không đúng!', 'error')
                logger.warning(f"Failed login attempt for {email}")
                return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Error logging in: {str(e)}")
            flash('Lỗi khi đăng nhập. Vui lòng thử lại.', 'error')
            return redirect(url_for('login'))
        finally:
            conn.close()
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Đăng xuất người dùng."""
    logger.info(f"User {current_user.id} logged out")
    logout_user()
    flash('Đã đăng xuất thành công!', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Hiển thị danh sách workflows của người dùng."""
    logger.debug(f"User {current_user.id} accessing dashboard")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, status, source_path, dest_path FROM workflows WHERE user_id = ?', (current_user.id,))
        workflows = [{'id': row[0], 'name': row[1], 'status': row[2], 'source_path': row[3], 'dest_path': row[4]} for row in cursor.fetchall()]
        logger.debug(f"Found {len(workflows)} workflows for user {current_user.id}")
        return render_template('dashboard.html', workflows=workflows, email=current_user.email, 
                             balance=current_user.balance, is_subscribed=current_user.is_subscribed)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        flash('Lỗi khi tải dashboard. Vui lòng thử lại.', 'error')
        return redirect(url_for('index'))
    finally:
        conn.close()

@app.route('/workflow_detail/<workflow_id>')
@login_required
def workflow_detail(workflow_id):
    """Hiển thị chi tiết log của workflow."""
    logger.debug(f"User {current_user.id} accessing workflow_detail for workflow_id: {workflow_id}")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name FROM workflows WHERE id = ? AND user_id = ?', (workflow_id, current_user.id))
        workflow = cursor.fetchone()
        if not workflow:
            logger.warning(f"Workflow {workflow_id} not found or user {current_user.id} has no access")
            flash('Workflow không tồn tại hoặc bạn không có quyền truy cập!', 'error')
            return render_template('workflow_detail.html', workflow=None, logs=[], email=current_user.email,
                                 balance=current_user.balance, is_subscribed=current_user.is_subscribed)
        
        cursor.execute('''
            SELECT id, timestamp, status, rows_processed, error, cost, source_file, dest_file
            FROM logs
            WHERE workflow_id = ?
            ORDER BY timestamp DESC
        ''', (workflow_id,))
        logs = [
            {
                'id': row[0],
                'timestamp': row[1].strftime('%Y-%m-%d %H:%M:%S') if row[1] else '',
                'status': row[2],  # Sửa lỗi truy cập status từ row[3] thành row[2]
                'rows_processed': row[3],  # Điều chỉnh chỉ số
                'error': row[4],
                'cost': row[5],
                'source_file': row[6],
                'dest_file': row[7]
            } for row in cursor.fetchall()
        ]
        logger.debug(f"Found {len(logs)} logs for workflow {workflow_id}")
        return render_template('workflow_detail.html', 
                             workflow={'id': workflow[0], 'name': workflow[1]},
                             logs=logs, email=current_user.email,
                             balance=current_user.balance, is_subscribed=current_user.is_subscribed)
    except Exception as e:
        logger.error(f"Error loading workflow details for {workflow_id}: {str(e)}")
        flash('Lỗi khi tải chi tiết workflow. Vui lòng thử lại.', 'error')
        return render_template('workflow_detail.html', workflow=None, logs=[], email=current_user.email,
                             balance=current_user.balance, is_subscribed=current_user.is_subscribed)
    finally:
        conn.close()

@app.route('/workflow_view/<workflow_id>')
@login_required
def workflow_view(workflow_id):
    """Hiển thị thiết kế flow của workflow."""
    logger.debug(f"User {current_user.id} accessing workflow_view for workflow_id: {workflow_id}")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, transformations FROM workflows WHERE id = ? AND user_id = ?', (workflow_id, current_user.id))
        workflow = cursor.fetchone()
        if not workflow:
            logger.warning(f"Workflow {workflow_id} not found or user {current_user.id} has no access")
            flash('Workflow không tồn tại hoặc bạn không có quyền truy cập!', 'error')
            return redirect(url_for('dashboard'))

        # Kiểm tra và parse dữ liệu transformations
        transformations = workflow[2]
        if not transformations:
            logger.warning(f"Transformations data is empty for workflow {workflow_id}")
            flash('Dữ liệu thiết kế workflow trống!', 'error')
            return redirect(url_for('dashboard'))

        try:
            flow = json.loads(transformations)
            # Kiểm tra xem flow có nodes không
            if not flow or 'nodes' not in flow or not flow['nodes']:
                logger.warning(f"No nodes found in transformations for workflow {workflow_id}")
                flash('Workflow không chứa node nào!', 'warning')
                return redirect(url_for('dashboard'))
            
            logger.debug(f"Flow data loaded successfully for workflow {workflow_id}: {flow}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding transformations JSON for workflow {workflow_id}: {str(e)}")
            flash('Lỗi khi đọc dữ liệu thiết kế workflow!', 'error')
            return redirect(url_for('dashboard'))

        return render_template('workflow_view.html', 
                             workflow={'id': workflow[0], 'name': workflow[1], 'flow': flow},
                             email=current_user.email, balance=current_user.balance, 
                             is_subscribed=current_user.is_subscribed)
    except Exception as e:
        logger.error(f"Error loading workflow view for {workflow_id}: {str(e)}")
        flash('Lỗi khi tải thiết kế workflow. Vui lòng thử lại.', 'error')
        return redirect(url_for('dashboard'))
    finally:
        conn.close()

@app.route('/create_workflow', methods=['GET', 'POST'])
@login_required
def create_workflow():
    """Tạo và thực thi workflow mới với giao diện kéo thả."""
    logger.debug(f"User {current_user.id} accessing create_workflow")
    if request.method == 'POST':
        errors = []
        name = request.form.get('name')
        logger.debug(f"Received workflow name: '{name}'")
        flow_json = request.form.get('flow')
        
        # Validate inputs
        if not name or name.strip() == '' or name.lower() == 'undefined':
            errors.append('Vui lòng nhập tên workflow!')
            logger.warning(f"Invalid workflow name: '{name}'")
        if not flow_json:
            errors.append('Vui lòng thiết kế flow trước khi lưu!')
            logger.warning("Flow JSON is missing")
        
        try:
            flow = json.loads(flow_json)
            validate_flow(flow)
        except Exception as e:
            errors.append(f'Lỗi trong thiết kế flow: {str(e)}')
            logger.error(f"Flow validation error: {str(e)}")
        
        if errors:
            logger.warning(f"Validation errors: {errors}")
            return jsonify({'success': False, 'messages': errors}), 400

        # Check for duplicate workflow name
        try:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM workflows WHERE user_id = ? AND name = ?', (current_user.id, name))
            if cursor.fetchone():
                errors.append('Tên workflow đã tồn tại! Vui lòng chọn tên khác.')
                logger.warning(f"Duplicate workflow name: {name}")
                return jsonify({'success': False, 'messages': errors}), 400
        except Exception as e:
            errors.append('Lỗi khi kiểm tra tên workflow. Vui lòng thử lại.')
            logger.error(f"Error checking workflow name: {str(e)}")
            return jsonify({'success': False, 'messages': errors}), 500
        finally:
            conn.close()

        # Create user directories
        user_upload_dir = os.path.join(UPLOAD_FOLDER, current_user.id)
        user_output_dir = os.path.join(OUTPUT_FOLDER, current_user.id)
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_output_dir, exist_ok=True)

        # Clean old files
        clean_old_files(user_output_dir, max_age_days=7)
        logger.info(f"Cleaned old files in {user_output_dir}")

        # Process nodes
        source_path = None
        dest_path = None
        for node in flow['nodes']:
            logger.debug(f"Processing node {node['id']} of type {node['type']}")
            if node['type'] == 'input':
                file_key = f"node_{node['id']}_file"
                file = request.files.get(file_key)
                if not file:
                    errors.append(f'Vui lòng tải lên tệp cho node Input {node["id"]}!')
                    logger.warning(f"No file uploaded for Input node {node['id']}")
                else:
                    filename = secure_filename(file.filename)
                    if not allowed_file(filename):
                        errors.append(f'Tệp Input phải là CSV, Excel hoặc JSON!')
                        logger.warning(f"Invalid file format for Input node {node['id']}: {filename}")
                    else:
                        filename = generate_unique_filename(user_upload_dir, filename)
                        source_path = os.path.join(user_upload_dir, filename)
                        file.save(source_path)
                        node['config']['path'] = source_path
                        logger.debug(f"Input file saved: {source_path}")
            elif node['type'] == 'output':
                # Lấy tên file từ config
                file_name = node['config'].get('path', '')
                dest_filename = secure_filename(file_name)
                dest_filename = generate_unique_filename(user_output_dir, dest_filename)
                dest_path = os.path.join(user_output_dir, dest_filename)

            elif node['type'] in ['join', 'enrich_data', 'fuzzy_match', 'cross_join']:
                file_key = f"node_{node['id']}_file"
                if file_key in request.files:
                    file = request.files[file_key]
                    if file and file.filename:
                        filename = secure_filename(file.filename)
                        if not allowed_file(filename):
                            errors.append(f'Tệp cho node {node["type"]} phải là CSV, Excel hoặc JSON!')
                            logger.warning(f"Invalid file format for node {node['type']}: {filename}")
                        else:
                            filename = generate_unique_filename(user_upload_dir, filename)
                            file_path = os.path.join(user_upload_dir, filename)
                            file.save(file_path)
                            if node['type'] == 'join':
                                node['config']['path'] = file_path
                            elif node['type'] == 'enrich_data':
                                node['config']['enrich_file'] = file_path
                            elif node['type'] == 'fuzzy_match':
                                node['config']['reference_file'] = file_path
                            elif node['type'] == 'cross_join':
                                node['config']['join_file'] = file_path
                            logger.debug(f"File saved for node {node['type']}: {file_path}")
            elif node['type'] == 'merge_records':
                file_key = f"node_{node['id']}_file"
                if file_key in request.files:
                    files = request.files.getlist(file_key)
                    input_files = []
                    for file in files:
                        if file and file.filename:
                            filename = secure_filename(file.filename)
                            if not allowed_file(filename):
                                errors.append(f'Tệp cho node Merge Records phải là CSV, Excel hoặc JSON!')
                                logger.warning(f"Invalid file format for Merge Records node: {filename}")
                            else:
                                filename = generate_unique_filename(user_upload_dir, filename)
                                file_path = os.path.join(user_upload_dir, filename)
                                file.save(file_path)
                                input_files.append(file_path)
                                logger.debug(f"File saved for Merge Records: {file_path}")
                    if input_files:
                        node['config']['input_files'] = ','.join(input_files)

        if errors:
            logger.warning(f"Node processing errors: {errors}")
            return jsonify({'success': False, 'messages': errors}), 400

        if not source_path or not dest_path:
            errors.append('Flow phải có node Input và Output với tệp hợp lệ!')
            logger.warning("Missing source or destination path")
            return jsonify({'success': False, 'messages': errors}), 400
        # Save workflow to database
        workflow_id = str(uuid.uuid4())
        try:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO workflows (id, user_id, name, status, source_type, source_path, transformations, dest_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (workflow_id, current_user.id, name, 'Created', 'file', source_path, json.dumps(flow), dest_path))
            conn.commit()
            logger.info(f"Workflow {workflow_id} created for user {current_user.id}")
        except Exception as e:
            errors.append('Lỗi khi lưu workflow. Vui lòng thử lại.')
            logger.error(f"Error saving workflow: {str(e)}")
            return jsonify({'success': False, 'messages': errors}), 500
        finally:
            conn.close()
        # Execute workflow
        logger.debug(f"Executing workflow {workflow_id}")
        success, execution_messages = execute_workflow(workflow_id, current_user.id, source_path, dest_path, flow)
        if success:
            # Trả về JSON với trạng thái thành công, messages, và URL để redirect
            return jsonify({
                'success': True,
                'messages': execution_messages,
                'redirect_url': url_for('dashboard')
            })
        else:
            return jsonify({'success': False,
                            'messages': execution_messages,
                            'redirect_url': url_for('dashboard')}), 500

    return render_template('create_workflow.html', email=current_user.email, 
                         balance=current_user.balance, is_subscribed=current_user.is_subscribed)

@app.route('/delete_workflow/<workflow_id>', methods=['POST'])
@login_required
def delete_workflow(workflow_id):
    """Xóa workflow."""
    logger.debug(f"User {current_user.id} deleting workflow {workflow_id}")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM workflows WHERE id = ? AND user_id = ?', (workflow_id, current_user.id))
        if not cursor.fetchone():
            flash('Workflow không tồn tại hoặc bạn không có quyền xóa!', 'error')
            logger.warning(f"Workflow {workflow_id} not found for deletion")
            return redirect(url_for('dashboard'))
        
        cursor.execute('DELETE FROM logs WHERE workflow_id = ?', (workflow_id,))
        cursor.execute('DELETE FROM workflows WHERE id = ?', (workflow_id,))
        conn.commit()
        flash('Workflow đã được xóa thành công!', 'success')
        logger.info(f"Workflow {workflow_id} deleted by user {current_user.id}")
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Error deleting workflow {workflow_id}: {str(e)}")
        flash('Lỗi khi xóa workflow. Vui lòng thử lại.', 'error')
        return redirect(url_for('dashboard'))
    finally:
        conn.close()

@app.route('/download_output/<workflow_id>')
@login_required
def download_output(workflow_id):
    """Tải tệp đầu ra của workflow."""
    logger.debug(f"User {current_user.id} downloading output for workflow {workflow_id}")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute('SELECT dest_path FROM workflows WHERE id = ? AND user_id = ?', (workflow_id, current_user.id))
        workflow = cursor.fetchone()
        if not workflow:
            flash('Workflow không tồn tại hoặc bạn không có quyền truy cập!', 'error')
            logger.warning(f"Workflow {workflow_id} not found for download")
            return redirect(url_for('dashboard'))

        dest_filename = os.path.basename(workflow[0])
        dest_path = os.path.join(OUTPUT_FOLDER, current_user.id, dest_filename)
        if not os.path.exists(dest_path):
            flash('Tệp đầu ra đã hết hạn (xóa sau 7 ngày) hoặc không tồn tại!', 'error')
            logger.warning(f"Output file not found: {dest_path}")
            return redirect(url_for('dashboard'))

        return send_file(dest_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading output for workflow {workflow_id}: {str(e)}")
        flash('Lỗi khi tải tệp đầu ra. Vui lòng thử lại.', 'error')
        return redirect(url_for('dashboard'))
    finally:
        conn.close()

@app.route('/billing')
@login_required
def billing():
    """Hiển thị lịch sử giao dịch từ logs và transactions."""
    logger.debug(f"User {current_user.id} accessing billing")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT l.id, l.timestamp, l.cost, l.status, w.name
            FROM logs l
            JOIN workflows w ON l.workflow_id = w.id
            WHERE w.user_id = ? AND l.cost > 0
        ''', (current_user.id,))
        workflow_transactions = [
            {
                'id': row[0],
                'timestamp': row[1].strftime('%Y-%m-%d %H:%M:%S'),
                'amount': -row[2],
                'status': row[3],
                'description': f"Thực thi workflow {row[4]}"
            } for row in cursor.fetchall()
        ]
        
        cursor.execute('''
            SELECT id, timestamp, amount, type, description
            FROM transactions
            WHERE user_id = ?
        ''', (current_user.id,))
        deposit_transactions = [
            {
                'id': row[0],
                'timestamp': row[1].strftime('%Y-%m-%d %H:%M:%S'),
                'amount': row[2],
                'status': 'Success',
                'description': row[4]
            } for row in cursor.fetchall()
        ]
        
        transactions = workflow_transactions + deposit_transactions
        transactions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.debug(f"Found {len(transactions)} transactions for user {current_user.id}")
        return render_template('billing.html', transactions=transactions, email=current_user.email,
                             balance=current_user.balance, is_subscribed=current_user.is_subscribed)
    except Exception as e:
        logger.error(f"Error in billing: {str(e)}")
        flash('Lỗi khi tải trang lịch sử giao dịch. Vui lòng thử lại.', 'error')
        return redirect(url_for('dashboard'))
    finally:
        conn.close()

@app.route('/wallet', methods=['GET', 'POST'])
@login_required
def wallet():
    """Quản lý ví và nạp tiền giả lập."""
    logger.debug(f"User {current_user.id} accessing wallet")
    if request.method == 'POST':
        amount = request.form.get('amount', type=float)
        if not amount or amount <= 0:
            flash('Vui lòng nhập số tiền hợp lệ!', 'error')
            logger.warning(f"Invalid deposit amount: {amount}")
            return redirect(url_for('wallet'))
        
        try:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users
                SET balance = balance + ?
                WHERE id = ?
            ''', (amount, current_user.id))
            transaction_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO transactions (id, user_id, timestamp, amount, type, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (transaction_id, current_user.id, datetime.now(), amount, 'deposit', f"Nạp {amount:.2f} USD vào ví"))
            conn.commit()
            current_user.balance += amount
            flash(f'Nạp {amount:.2f} USD thành công!', 'success')
            logger.info(f"User {current_user.id} deposited {amount:.2f} USD")
            return redirect(url_for('wallet'))
        except Exception as e:
            logger.error(f"Error depositing money: {str(e)}")
            flash('Lỗi khi nạp tiền. Vui lòng thử lại.', 'error')
            return redirect(url_for('wallet'))
        finally:
            conn.close()
    
    return render_template('wallet.html', email=current_user.email, 
                         balance=current_user.balance, is_subscribed=current_user.is_subscribed)

@app.route('/upgrade', methods=['GET', 'POST'])
@login_required
def upgrade():
    """Nâng cấp tài khoản Pro miễn phí."""
    logger.debug(f"User {current_user.id} accessing upgrade")
    if request.method == 'POST':
        if current_user.is_subscribed:
            flash('Bạn đã là thành viên Pro!', 'error')
            logger.warning(f"User {current_user.id} already subscribed")
            return redirect(url_for('upgrade'))
        
        try:
            conn = pyodbc.connect(DB_CONNECTION_STRING)
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET is_subscribed = 1 WHERE id = ?', (current_user.id,))
            transaction_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO transactions (id, user_id, timestamp, amount, type, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (transaction_id, current_user.id, datetime.now(), 0, 'subscription', 'Nâng cấp tài khoản Pro'))
            conn.commit()
            current_user.is_subscribed = True
            flash('Nâng cấp tài khoản Pro thành công!', 'success')
            logger.info(f"User {current_user.id} upgraded to Pro")
            return redirect(url_for('upgrade'))
        except Exception as e:
            logger.error(f"Error upgrading user {current_user.id}: {str(e)}")
            flash('Lỗi khi nâng cấp tài khoản. Vui lòng thử lại.', 'error')
            return redirect(url_for('upgrade'))
        finally:
            conn.close()
    
    return render_template('upgrade.html', email=current_user.email, 
                         balance=current_user.balance, is_subscribed=current_user.is_subscribed)

if __name__ == '__main__':
    app.run(debug=True)
