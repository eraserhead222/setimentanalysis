from flask import Flask, render_template
from flask import Flask, render_template, request
import pandas as pd
import unicodedata
from pathlib import Path
from urllib.parse import quote

app = Flask(__name__)


def normalize_name(name):
    """ตัดช่องว่างหน้า-หลัง และทำ Unicode normalization ให้ชื่อร้าน
    กันปัญหาชื่อภาษาไทยที่หน้าตาเหมือนกันแต่ไบต์ต่างกัน (เช่น สระ/วรรณยุกต์
    คนละรูปแบบ) จนทำให้เทียบ/หา key ไม่เจอ"""
    return unicodedata.normalize('NFC', str(name).strip())



summary_df = pd.read_csv('data/restaurant_summary.csv')
detail_df = pd.read_csv('data/restaurant_with_sentiment.csv')
coords_df = pd.read_csv('data/restaurant_coords.csv')

# เก็บชื่อร้านต้นฉบับไว้แสดงผล แต่ merge coords ด้วยชื่อที่ normalize แล้ว
summary_df['_merge_key'] = summary_df['restaurant_name'].apply(normalize_name)
coords_df['_merge_key'] = coords_df['restaurant_name'].apply(normalize_name)

coords_cols = [c for c in coords_df.columns if c not in ('restaurant_name', '_merge_key')]
summary_df = summary_df.merge(
    coords_df[['_merge_key'] + coords_cols], on='_merge_key', how='left'
)

def load_local_restaurant_photos():
    """อ่านรูปใน static/images/restaurants โดยจับคู่ร้านจาก restaurant-folders.csv."""
    photo_root = Path(app.static_folder) / 'images' / 'restaurants'
    mapping_path = photo_root / 'restaurant-folders.csv'
    photos = {}
    if not mapping_path.is_file():
        return photos

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    for row in mapping.to_dict('records'):
        restaurant_name = str(row.get('restaurant_name', '')).strip()
        folder_name = Path(str(row.get('folder', '')).strip()).name
        folder_path = photo_root / folder_name
        if not restaurant_name or not folder_path.is_dir():
            continue

        image_files = sorted(
            (image for image in folder_path.iterdir()
             if image.is_file() and image.suffix.lower() in allowed_extensions),
            key=lambda image: image.name.casefold()
        )
        if not image_files:
            continue

        cover = next(
            (image for image in image_files if image.stem.casefold() == 'cover'),
            image_files[0]
        )

        def static_url(image):
            relative_path = image.relative_to(app.static_folder).as_posix()
            return '/static/' + quote(relative_path, safe='/')

        photos[normalize_name(restaurant_name)] = {
            'cover': static_url(cover),
            'gallery': [static_url(image) for image in image_files]
        }
    return photos


# ใช้รูปที่เก็บในโปรเจกต์แทน URL ภายนอกที่หมดอายุ
LOCAL_PHOTOS = load_local_restaurant_photos()
LOCAL_COVERS = {key: value['cover'] for key, value in LOCAL_PHOTOS.items()}
summary_df['photo_url'] = summary_df['_merge_key'].map(LOCAL_COVERS)
summary_df['photo_url'] = summary_df['photo_url'].where(summary_df['photo_url'].notna(), None)

summary_df = summary_df.drop(columns=['_merge_key'])

print("DEBUG จับคู่รูปได้:", summary_df['photo_url'].notna().sum(), "/", len(summary_df))

@app.route('/')
def dashborad():
    total_reviews = len(detail_df)
    pos_count = int((detail_df['sentiment'] == 'pos').sum())
    neu_count = int((detail_df['sentiment'] == 'neu').sum())
    neg_count = int((detail_df['sentiment'] == 'neg').sum())
    pos_pct = round(pos_count / total_reviews * 100, 1)
    total_restaurants = summary_df['restaurant_name'].nunique()

    aspects = ['ราคา', 'บริการ' , 'รสชาติ','ความสะอาด']
    aspect_labels = []
    aspect_values = []
    for aspect in aspects:
        col_name = f'{aspect}_pos_pct'
        mention_col = f'{aspect}_mentions'
        filtered = summary_df[summary_df[mention_col] > 0]
        avg_val = filtered[col_name].mean()
        aspect_labels.append(aspect)
        aspect_values.append(round(avg_val, 1)if not pd.isna(avg_val) else 0)

    summary_df['overall_pos'] = (
        summary_df[['ราคา_pos_pct', 'บริการ_pos_pct','รสชาติ_pos_pct', 'ความสะอาด_pos_pct']]
        .mean(axis=1)
    )
    summary_df['stars'] = (summary_df['overall_pos'] / 20).round(1)
    top_restaurants = summary_df.sort_values('overall_pos', ascending=False).head(4)
    top_restaurants_list = top_restaurants[['restaurant_name', 'total_reviews', 'overall_pos', 'stars', 'photo_url']].to_dict('records')

    map_restaurants = []
    for _, row in summary_df.iterrows():
        lat, lng = row.get('latitude'), row.get('longitude')
        if pd.notna(lat) and pd.notna(lng):
            map_restaurants.append({
                'name': str(row['restaurant_name']),
                'lat': float(lat),
                'lng': float(lng),
                'sentiment': round(float(row['overall_pos']), 1) if pd.notna(row['overall_pos']) else 0,
                'stars': float(row['stars']) if pd.notna(row['stars']) else 0,
                'reviews': int(row['total_reviews']) if pd.notna(row['total_reviews']) else 0,
                'url': f"/restaurant/{row['restaurant_name']}"
            })

    from collections import Counter
    import re
    all_words = []
    for text in detail_df['review_body'].dropna():
        words = re.findall(r'[ก-๙]{3,}', str(text))
        all_words.extend(words)
    common_words = Counter(all_words).most_common(8)


    return render_template('dashboard.html', 
                          total_reviews=total_reviews,
                          pos_count=pos_count,
                          neu_count=neu_count,
                          neg_count=neg_count,
                          pos_pct=pos_pct,
                          total_restaurants=total_restaurants,
                          aspect_labels=aspect_labels,
                          aspect_values=aspect_values,
                          top_restaurants=top_restaurants_list,
                          common_words=common_words,
                          map_restaurants=map_restaurants)


@app.route('/search')
def search():
    summary_df['overall_pos'] = (
        summary_df[['ราคา_pos_pct', 'บริการ_pos_pct', 'รสชาติ_pos_pct', 'ความสะอาด_pos_pct']]
        .mean(axis=1)
    )
    summary_df['stars'] = round(summary_df['overall_pos'] / 100 * 5, 1)
    
    all_restaurants = summary_df.sort_values('overall_pos', ascending=False)
    restaurants_list = all_restaurants[['restaurant_name', 'total_reviews', 'overall_pos', 'stars','photo_url']].to_dict('records')
    
    return render_template('search.html', 
                          restaurants=restaurants_list,
                          total_count=len(restaurants_list))


@app.route('/restaurants')
def restaurants():
    summary_df['overall_pos'] = (
        summary_df[['ราคา_pos_pct', 'บริการ_pos_pct', 'รสชาติ_pos_pct', 'ความสะอาด_pos_pct']]
        .mean(axis=1)
    )
    summary_df['stars'] = round(summary_df['overall_pos'] / 100 * 5, 1)
    
    all_restaurants = summary_df.sort_values('overall_pos', ascending=False)
    restaurants_list = all_restaurants[['restaurant_name', 'total_reviews', 'overall_pos', 'stars','photo_url']].to_dict('records')
    
    return render_template('restaurants.html', 
                          restaurants=restaurants_list,
                          total_count=len(restaurants_list))
                          

@app.route('/review')
def review():
    df = pd.read_csv('data/restaurant_final.csv')
    df = df.dropna(subset=['review_body', 'sentiment', 'stars'])

    total = len(df)
    pos_count = int((df['sentiment'] == 'pos').sum())
    neu_count = int((df['sentiment'] == 'neu').sum())
    neg_count = int((df['sentiment'] == 'neg').sum())
    pos_pct = round(pos_count / total * 100, 1)
    neu_pct = round(neu_count / total * 100, 1)
    neg_pct = round(neg_count / total * 100, 1)

    avatars = ['😺', '🐶', '🐰', '🦊', '🐼', '🐨', '🐯', '🐸', '🐵', '🐹']

    reviews = []
    for i, row in df.iterrows():
        aspects_str = str(row.get('aspects', ''))
        reviews.append({
            'avatar': avatars[i % len(avatars)],
            'text': str(row['review_body'])[:200],
            'stars': int(row['stars']),
            'sentiment': row['sentiment'],
            'restaurant': row['restaurant_name'],
            'has_ราคา': 'ราคา' in aspects_str,
            'has_บริการ': 'บริการ' in aspects_str,
            'has_รสชาติ': 'รสชาติ' in aspects_str,
            'has_ความสะอาด': 'ความสะอาด' in aspects_str,
        })

    return render_template('review.html',
                          total=total, pos_count=pos_count, neu_count=neu_count, neg_count=neg_count,
                          pos_pct=pos_pct, neu_pct=neu_pct, neg_pct=neg_pct,
                          reviews=reviews)

@app.route('/analysis')
def analysis():
    df = pd.read_csv('data/restaurant_final.csv')
    df = df.dropna(subset=['review_body', 'sentiment', 'stars'])

    total = len(df)
    pos_count = int((df['sentiment'] == 'pos').sum())
    neu_count = int((df['sentiment'] == 'neu').sum())
    neg_count = int((df['sentiment'] == 'neg').sum())
    pos_pct = round(pos_count / total * 100, 1)
    neu_pct = round(neu_count / total * 100, 1)
    neg_pct = round(neg_count / total * 100, 1)

    # หา sentiment ที่มากที่สุด (2) แสดงอัตราความคิดเห็นโดยรวม
    counts = {'pos': pos_count, 'neu': neu_count, 'neg': neg_count}
    dominant = max(counts, key=counts.get)
    dominant_map = {'pos': ('เชิงบวก', '😊', pos_pct), 'neu': ('ปานกลาง', '😐', neu_pct), 'neg': ('เชิงลบ', '☹️', neg_pct)}
    dominant_label, dominant_icon, dominant_pct = dominant_map[dominant]

    avg_stars = round(df['stars'].mean(), 2)  # (3) คะแนนเฉลี่ยโดยรวม
    total_restaurants = df['restaurant_name'].nunique()  # (4)

    # (5) ความคิดเห็นแยกตามหมวดหมู่ (ใช้ 4 ด้านที่มีข้อมูลจริงแทนประเภทร้าน)
    aspects = ['ราคา', 'บริการ', 'รสชาติ', 'ความสะอาด']
    aspect_data = []
    for a in aspects:
        col = f'sentiment_{a}'
        sub = df[col].dropna()
        total_a = len(sub)
        if total_a == 0:
            aspect_data.append({'label': a, 'pos': 0, 'neu': 0, 'neg': 0})
        else:
            aspect_data.append({
                'label': a,
                'pos': round((sub == 'pos').sum() / total_a * 100, 1),
                'neu': round((sub == 'neu').sum() / total_a * 100, 1),
                'neg': round((sub == 'neg').sum() / total_a * 100, 1),
            })

    # (7) คำที่ถูกกล่าวถึงมากที่สุด
    from collections import Counter
    import re
    all_words = []
    for text in df['review_body'].dropna():
        words = re.findall(r'[ก-๙]{3,}', str(text))
        all_words.extend(words)
    common_words = Counter(all_words).most_common(8)

    # (9) ข้อมูลเชิงลึก - คำนวณจากข้อมูลจริง
    best_aspect = max(aspect_data, key=lambda x: x['pos'])
    worst_aspect = min(aspect_data, key=lambda x: x['pos'])
    top_word = common_words[0] if common_words else ('-', 0)
    top_restaurant_row = df['restaurant_name'].value_counts().idxmax()
    top_restaurant_count = int(df['restaurant_name'].value_counts().max())

    return render_template('analysis.html',
                          total=total, pos_pct=pos_pct, neu_pct=neu_pct, neg_pct=neg_pct,
                          pos_count=pos_count, neu_count=neu_count, neg_count=neg_count,
                          dominant_label=dominant_label, dominant_icon=dominant_icon, dominant_pct=dominant_pct,
                          avg_stars=avg_stars, total_restaurants=total_restaurants,
                          aspect_data=aspect_data, common_words=common_words,
                          best_aspect=best_aspect, worst_aspect=worst_aspect,
                          top_word=top_word, top_restaurant_row=top_restaurant_row,
                          top_restaurant_count=top_restaurant_count)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    submitted = False
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        topic = request.form.get('topic')
        message = request.form.get('message')
        # ตอนนี้แค่รับข้อมูลเฉยๆ ยังไม่ได้ส่งอีเมลจริง (ทำได้ในอนาคตถ้าต้องการ)
        print(f"ข้อความใหม่จาก {name} ({email}) หัวข้อ: {topic}")
        submitted = True

    return render_template('contact.html', submitted=submitted)

@app.route('/restaurant/<name>')
def restaurant_detail(name):
    df = pd.read_csv('data/restaurant_final.csv')
    df = df[df['restaurant_name'] == name]
    coord_row = coords_df[coords_df['restaurant_name'].apply(normalize_name) == normalize_name(name)]
    lat = float(coord_row['latitude'].values[0]) if len(coord_row) > 0 else None
    lng = float(coord_row['longitude'].values[0]) if len(coord_row) > 0 else None
    photo_data = LOCAL_PHOTOS.get(normalize_name(name), {})
    df = df.dropna(subset=['review_body', 'sentiment', 'stars'])

    if len(df) == 0:
        return "ไม่พบร้านอาหารนี้", 404

    total = len(df)
    pos_count = int((df['sentiment'] == 'pos').sum())
    neu_count = int((df['sentiment'] == 'neu').sum())
    neg_count = int((df['sentiment'] == 'neg').sum())
    pos_pct = round(pos_count / total * 100, 1)
    neu_pct = round(neu_count / total * 100, 1)
    neg_pct = round(neg_count / total * 100, 1)
    avg_stars = round(df['stars'].mean(), 1)

    # คะแนนรีวิวแยกตามระดับดาว 1-5
    star_dist = []
    for s in [5, 4, 3, 2, 1]:
        count_s = int((df['stars'] == s).sum())
        pct_s = round(count_s / total * 100, 1) if total > 0 else 0
        star_dist.append({'star': s, 'count': count_s, 'pct': pct_s})

    # ความคิดเห็นแยกตามหมวดหมู่ (4 ด้านที่มีข้อมูลจริง)
    aspects = ['ราคา', 'บริการ', 'รสชาติ', 'ความสะอาด']
    aspect_data = []
    for a in aspects:
        col = f'sentiment_{a}'
        sub = df[col].dropna()
        total_a = len(sub)
        pos_a = round((sub == 'pos').sum() / total_a * 100, 1) if total_a > 0 else 0
        aspect_data.append({'label': a, 'pos': pos_a})

    # คำที่พบบ่อยของร้านนี้
    from collections import Counter
    import re
    all_words = []
    for text in df['review_body'].dropna():
        words = re.findall(r'[ก-๙]{3,}', str(text))
        all_words.extend(words)
    common_words = Counter(all_words).most_common(5)

    # ตัวอย่างรีวิว 3 อัน
    avatars = ['😺', '🐶', '🐰', '🦊', '🐼']
    sample_reviews = []
    for i, row in df.head(3).iterrows():
        sample_reviews.append({
            'avatar': avatars[i % len(avatars)],
            'stars': int(row['stars']),
            'text': str(row['review_body'])[:150]
        })

    return render_template('restaurant_detail.html',
                          name=name, total=total, pos_pct=pos_pct, neu_pct=neu_pct, neg_pct=neg_pct,
                          pos_count=pos_count, neu_count=neu_count, neg_count=neg_count,
                          avg_stars=avg_stars, star_dist=star_dist, aspect_data=aspect_data,
                          lat=lat, lng=lng,
                          photo_url=photo_data.get('cover'),
                          gallery_photos=photo_data.get('gallery', []),
                          common_words=common_words, sample_reviews=sample_reviews)
                          
if __name__ == '__main__':
    app.run(debug=True, port=5000)






