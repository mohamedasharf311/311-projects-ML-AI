# 1️⃣ استيراد المكتبات
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 2️⃣ قراءة ملف CSV (تأكد إن الملف products.csv موجود في نفس المجلد)
csv_path = "MyDream/marketing_data.csv"
df = pd.read_csv(csv_path)
print("✅ تم تحميل البيانات")
print(df.head())

# 3️⃣ تحويل البيانات إلى نصوص تدريبية (من غير generated_ad)
texts = []
for _, row in df.iterrows():
    text = (
        f"المنتج: {row['product_name']} | "
        f"الجمهور المستهدف: {row['target_audience']} | "
        f"العمر: {row['target_age']} | "
        f"الفائدة الأساسية: {row['main_benefit']} | "
        f"المميزات: {row['features']} | "
        f"الدليل الاجتماعي: {row['social_proof']} | "
        f"الندرة/الاستعجال: {row['scarcity_urgency']} | "
        f"القصة: {row['storytelling']} | "
        f"ضمان المخاطرة: {row['risk_reversal']} | "
        f"المحفزات العاطفية: {row['emotional_triggers']} | "
        f"لغة قوية: {row['powerful_language']} | "
        f"استراتيجية التسعير: {row['pricing_strategy']} | "
        f"الصور: {row['visuals']} | "
        f"نقطة الألم: {row['pain_point']} | "
        f"الحل: {row['solution']}"
    )
    texts.append(text)

# 4️⃣ حفظ النصوص في ملف نصي
with open("ads.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")

# 5️⃣ تحميل البيانات للتدريب
dataset = load_dataset('text', data_files={'train': 'ads.txt'})

# 6️⃣ تحميل الموديل الداعم للعربية
model_name = "aubmindlab/aragpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 7️⃣ تجهيز البيانات (Tokenization)
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 8️⃣ إعداد التدريب
training_args = TrainingArguments(
    output_dir="./aragpt2-ads",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    save_strategy="epoch",
    report_to="none"
)

# 9️⃣ تعريف المدرب (Trainer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)

# 🔟 بدء التدريب
trainer.train()

# 1️⃣1️⃣ حفظ الموديل محليًا
save_path = "./aragpt2-ads"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ تم تدريب الموديل وحفظه محليًا بنجاح!")
