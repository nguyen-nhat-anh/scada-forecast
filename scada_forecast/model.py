import pandas as pd
import numpy as np
import tensorflow as tf
import gc

from scada_forecast.metrics import ape, accuracy


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, batch_size, train_flag=True,
                 train_df=None, val_df=None, test_df=None, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.train_flag = train_flag
        # Work out the label column indices.
        self.label_indices = None
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
            
        if self.train_flag:
            self.column_indices = {name: i for i, name in
                                   enumerate(train_df.columns)}
            self.column_dtype = {name: train_df[name].dtype for name in train_df.columns}
        else: 
            self.column_indices = {name: i for i, name in
                                   enumerate(test_df.columns)}
            self.column_dtype = {name: test_df[name].dtype for name in test_df.columns}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        if self.train_flag:
            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        self.batch_size = batch_size

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = np.zeros(inputs.shape[0])
        if self.train_flag:
            labels = features[:, self.labels_slice, :]
            if self.label_columns is not None:
                labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        if self.train_flag:
            labels.set_shape([None, self.label_width, None])
        return inputs, labels
    
    def array_to_dict(self, features, labels):
        features_dict = {col_idx: tf.cast(features[:, :, col_val], self.column_dtype[col_idx]) 
                         for col_idx, col_val in self.column_indices.items()}
        if self.train_flag:
            return features_dict, labels
        else:
            return features_dict
    
    def make_dataset(self, data, shuffle):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=shuffle,
          batch_size=self.batch_size)

        ds = ds.map(self.split_window)
        ds = ds.map(self.array_to_dict)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df, shuffle=True)

    @property
    def val(self):
        return self.make_dataset(self.val_df, shuffle=False)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

def split(df, categorical_cols, numeric_cols, target_col, input_width, shift, n_test, n_val=None, hour_steps=1):
    input_cols = categorical_cols + numeric_cols
    useful_df = df[input_cols + [target_col]].copy()
    useful_df.dropna(inplace=True)
    
    res = 1 / hour_steps # time resolution (hours / step)
    total_window_size = input_width + shift
    split_date_test = str(df.index[-1] - 
                          pd.Timedelta(hours=res*(n_test - 1)) - 
                          pd.Timedelta(hours=res*(total_window_size - 1)))
    test_df = useful_df.loc[split_date_test:]
    
    if n_val is not None:
        split_date_val = str(pd.Timestamp(split_date_test) - 
                             pd.Timedelta(hours=res*(n_val - 1)) - 
                             pd.Timedelta(hours=res*(total_window_size - 1)))

    
        train_df = useful_df.loc[:split_date_val]
        val_df = useful_df.loc[split_date_val:split_date_test]
    
        return train_df, val_df, test_df
    else:
        train_df = useful_df.loc[:split_date_test]
        return train_df, test_df

def get_feature_columns(dtype_dict, categorical_unique_values_dict, categorical_cols, numeric_cols, input_width):
    feature_cols = []
    feature_inputs_layer = {}
    for col in categorical_cols:
        feature_col = tf.feature_column.categorical_column_with_vocabulary_list(col, categorical_unique_values_dict[col])
        feature_cols.append(tf.feature_column.indicator_column(feature_col))
        feature_inputs_layer[col] = tf.keras.layers.Input(shape=(input_width,), name=col, dtype=dtype_dict[col])
    for col in numeric_cols:
        feature_col = tf.feature_column.numeric_column(col)
        feature_cols.append(feature_col)
        feature_inputs_layer[col] = tf.keras.layers.Input(shape=(input_width,), name=col, dtype=dtype_dict[col])
    return feature_cols, feature_inputs_layer

def parse_inputs(inputs, feature_cols):
    """
    inputs - {feature_column: value}
    feature_cols - list of feature columns 
    """
    return tf.map_fn(lambda x: tf.keras.layers.DenseFeatures(feature_cols)(x), 
                     inputs, fn_output_signature=tf.float32) # apply DenseFeatures() for each sample in batch

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model, input_column):
        super().__init__()
        self.model = model
        self.input_column = input_column

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        latest_log_load = tf.keras.layers.Reshape((-1, 1))(inputs[self.input_column][:, -1])

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return latest_log_load + delta

def build_network(dtype_dict, categorical_unique_values_dict, categorical_cols, numeric_cols, input_width):
    feature_cols, inputs_layer = get_feature_columns(dtype_dict, categorical_unique_values_dict, 
                                                     categorical_cols, numeric_cols, input_width)
    merge = tf.keras.layers.Lambda(lambda x: parse_inputs(x, feature_cols))(inputs_layer) # (None, input_width, n_features)
    out = tf.keras.layers.LSTM(200, activation='tanh')(merge) # (None, 200)
    out = tf.keras.layers.Dense(1, kernel_initializer=tf.initializers.zeros)(out) # (None, 1)
    out = tf.keras.layers.Reshape((-1, 1))(out) # (None, 1, 1)
    return tf.keras.Model([v for v in inputs_layer.values()], [out])

def create_model(dtype_dict, categorical_unique_values_dict, categorical_cols, numeric_cols, input_width, lastest_load_column):
    model = build_network(dtype_dict, categorical_unique_values_dict, 
                          categorical_cols, numeric_cols, input_width)
    return ResidualWrapper(model, input_column=lastest_load_column)

def train_model(df, input_width, n_val, n_test, hour_steps, 
                dtype_dict, categorical_unique_values_dict, 
                chosen_features, categorical_cols, 
                numeric_cols, target_col, ckpt_path):
    label_width = 1
    shift = 0
    batch_size = 256
    lastest_load_column = 'Lag1h'
    learning_rate = 0.001
    epochs = 10
    
    chosen_categorical_cols = list(set.intersection(set(chosen_features), set(categorical_cols)))
    chosen_numeric_cols = list(set.intersection(set(chosen_features), set(numeric_cols)))
    
    train_df, val_df, test_df = split(df, chosen_categorical_cols, chosen_numeric_cols, target_col,
                                      input_width, shift, n_test, n_val, hour_steps)
    
    window = WindowGenerator(input_width, label_width, shift, batch_size, train_flag=True,
                             train_df=train_df, val_df=val_df, test_df=test_df, label_columns=[target_col])
    
    # model
    model = create_model(dtype_dict, categorical_unique_values_dict, 
                         chosen_categorical_cols, chosen_numeric_cols, 
                         input_width, lastest_load_column)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss=tf.losses.MeanAbsoluteError())
    
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_best_only=True,
        save_weights_only=True
    )

    model.fit(window.train, epochs=epochs, validation_data=window.val, callbacks=[ckpt_cb])
    model.load_weights(ckpt_path)
    
    # evaluate
    forecast = np.exp(model.predict(window.val).squeeze())
    actual = np.exp(val_df['LogLoad'][-n_val:].values)
    val_loss = np.mean(ape(actual, forecast))
    print('Validation Accuracy: {:.2f}%, MAPE: {:.2f}%'.format(accuracy(actual, forecast, threshold=0.02)*100, 
                                                               np.mean(ape(actual, forecast))*100))
    
    forecast = np.exp(model.predict(window.test).squeeze())
    actual = np.exp(test_df['LogLoad'][-n_test:].values)
    print('Test Accuracy: {:.2f}%, MAPE: {:.2f}%'.format(accuracy(actual, forecast, threshold=0.02)*100, 
                                                         np.mean(ape(actual, forecast))*100))
    
    tf.keras.backend.clear_session()
    del model
    gc.collect();
    return val_loss,

def inference(df, input_width, ckpt_path,
              dtype_dict, categorical_unique_values_dict, 
              chosen_features, categorical_cols, numeric_cols):
    label_width = 1
    shift = 0
    lastest_load_column = 'Lag1h'
    batch_size = 256
    chosen_categorical_cols = list(set.intersection(set(chosen_features), set(categorical_cols)))
    chosen_numeric_cols = list(set.intersection(set(chosen_features), set(numeric_cols)))
    
    window = WindowGenerator(input_width, label_width=label_width, shift=shift, batch_size=batch_size, 
                             train_flag=False, train_df=None, val_df=None, test_df=df, label_columns=None)
    
    # model
    model = create_model(dtype_dict, categorical_unique_values_dict, 
                         chosen_categorical_cols, chosen_numeric_cols, 
                         input_width, lastest_load_column)
    
    model.load_weights(ckpt_path)
    
    # forecast
    forecast = np.exp(model.predict(window.test).squeeze())
    
    forecast_df = df.iloc[-len(forecast):].copy()
    forecast_df['Forecast'] = forecast
    forecast_df = forecast_df[['Forecast']]
    return forecast_df