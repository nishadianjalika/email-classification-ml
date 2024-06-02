# %%
pip install pydot graphviz

# %%
dbutils.library.restartPython()

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
emails_df = pd.read_csv('/dbfs/nishadi/top_100_senders_output.csv')
emails_df.rename(columns={'from_': 'from'}, inplace=True)
emails_df = emails_df.dropna()

# Encode the labels
label_encoder = LabelEncoder()
emails_df['label_encoded'] = label_encoder.fit_transform(emails_df['from'])

# Split the data into train, test, and validation sets
train_df, remaining_df = train_test_split(emails_df, test_size=0.3, stratify=emails_df['label_encoded'], random_state=42)
val_df, test_df = train_test_split(remaining_df, test_size=0.5, stratify=remaining_df['label_encoded'], random_state=42)

Y_train = train_df['label_encoded'].values
Y_test = test_df['label_encoded'].values
Y_val = val_df['label_encoded'].values

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
class_weights = dict(enumerate(class_weights))

# Tokenizer and tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
max_len = 100

def tokenize_data(df):
    return tokenizer(
        text=df.body.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
    )

x_train = tokenize_data(train_df)
x_test = tokenize_data(test_df)
x_val = tokenize_data(val_df)

# %%
# Load BERT model for sequence classification
bert_model = TFBertModel.from_pretrained('bert-base-cased')

# Custom model with dropout layers
input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

embeddings = bert_model(input_ids, attention_mask=attention_mask)[0]
x = GlobalAveragePooling1D()(embeddings)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(100, activation=None)(x)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# %%
model.summary()

# %%
from tensorflow.keras.utils import plot_model

# Save the model architecture as a PNG file
plot_model(model, to_file='/dbfs/nishadi/model_architecture.png', show_shapes=True, show_layer_names=True)

# %%
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

# Train the model with class weights
history = model.fit(
    [x_train['input_ids'], x_train['attention_mask']],
    Y_train,
    validation_data=(
        [x_val['input_ids'], x_val['attention_mask']],
        Y_val
    ),
    epochs=6,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# %%
# Evaluate the model on the training data
train_results = model.evaluate([x_train['input_ids'], x_train['attention_mask']], Y_train)
print(f'Train Results: {train_results}')

# Evaluate the model on the validation data
val_results = model.evaluate([x_val['input_ids'], x_val['attention_mask']], Y_val)
print(f'Validation Results: {val_results}')

# Evaluate the model on the test data
test_results = model.evaluate([x_test['input_ids'], x_test['attention_mask']], Y_test)
print(f'Test Results: {test_results}')

# %%
# Predictions on the test set
predictions = model.predict([x_test['input_ids'], x_test['attention_mask']])
y_pred = np.argmax(predictions, axis=1)

# Calculate metrics for the test set
test_accuracy = accuracy_score(Y_test, y_pred)
test_f1 = f1_score(Y_test, y_pred, average='weighted')

print(f'Test Accuracy: {test_accuracy}')
print(f'Test F1 Score: {test_f1}')
print(classification_report(Y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix for test set
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix\nAccuracy: {test_accuracy:.2f}, F1 Score: {test_f1:.2f}')
plt.show()

# %%
# Plot training & validation loss and accuracy values
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
epochs = range(1, len(train_loss) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(epochs, train_loss, 'b', label='Training loss')
ax1.plot(epochs, val_loss, 'r', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, train_accuracy, 'b', label='Training accuracy')
ax2.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()


