import pandas as pd
from sklearn.neighbors import NearestNeighbors

# 👥 تقييمات المستخدمين لبعض المنتجات
# (0 = ما قيّمش المنتج)
data = {
    'Product1': [5, 4, 1, 0],
    'Product2': [3, 0, 1, 1],
    'Product3': [0, 0, 0, 5],
    'Product4': [0, 5, 5, 4],
    'Product5': [2, 1, 4, 0],
}

# إنشاء جدول التقييمات
df = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4'])

print("🧾 تقييمات المستخدمين للمنتجات:")
print(df)

# 🔍 بناء نموذج KNN لقياس التشابه بين المستخدمين
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(df)

# 👤 المستخدم المطلوب نرشح له منتجات (مثلاً User1)
target_user = 'User1'
user_index = df.index.get_loc(target_user)

# 🧠 إيجاد أقرب المستخدمين
distances, indices = model.kneighbors([df.iloc[user_index]], n_neighbors=3)

print(f"\n📍 أقرب المستخدمين لـ {target_user}:")
for i in range(1, len(indices[0])):
    neighbor = df.index[indices[0][i]]
    print(f"👤 {neighbor} - المسافة: {distances[0][i]:.2f}")

# 🎯 التوصية بمنتجات حلوة من الجيران وهو لسه مجربهاش
user_ratings = df.iloc[user_index]
recommended_products = {}

for i in range(1, len(indices[0])):
    neighbor_ratings = df.iloc[indices[0][i]]
    for product in df.columns:
        if user_ratings[product] == 0 and neighbor_ratings[product] >= 4:
            if product in recommended_products:
                recommended_products[product] += 1
            else:
                recommended_products[product] = 1

# ✅ عرض التوصيات
print(f"\n🎁 المنتجات المقترحة لـ {target_user}:")
if recommended_products:
    sorted_recommendations = sorted(recommended_products.items(), key=lambda x: -x[1])
    for product, score in sorted_recommendations:
        print(f"✔️ {product} (موصى به من {score} مستخدمين مشابهين)")
else:
    print("❌ لا توجد توصيات حالياً. المستخدم محتاج يقيّم منتجات أكتر.")