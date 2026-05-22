import random
import pandas as pd
from datetime import datetime, timedelta

# 1. SETUP produks (Tetap 40 produks)
product_meta = {
    1: ["Kripik Dage", 18000], 2: ["Kripik Mini", 8000], 3: ["Kripik 22", 10000],
    4: ["Kripik 180", 62000], 5: ["Peyek Rasa", 15000], 6: ["Peyek Rame", 12000],
    7: ["Peyek Barokah", 12000], 8: ["Makroni Keju", 5000], 9: ["Makroni Evi", 5000],
    10: ["Pang-Pang Majorian", 8000], 11: ["Keg Koro Kulit Majorian", 8000],
    12: ["Keg Koro Kupas Majorian", 8000], 13: ["Stik Banyu Emas", 25000],
    14: ["Stik Arjuna", 20000], 15: ["Stik Arjuna Kacang Umpet", 25000],
    16: ["Sale Crispy 1kg", 40000], 17: ["Sale Crispy Inti 10pcs", 25000],
    18: ["Sriping Pisang Eka Rasa", 20000], 19: ["Tenggiri Ada Rasa", 15000],
    20: ["Tenggiri Mandiri", 10000], 21: ["Citruk", 17000], 22: ["Bolu Ketapang", 22000],
    23: ["Kuping Gajah", 15000], 24: ["Rengginang Koin", 35000], 25: ["Combro Mania", 15000],
    26: ["Sale Gulung Sekar Prisma", 22000], 27: ["Sale Keju Sari Manis", 22000],
    28: ["Sale Mutiara Jaya", 22000], 29: ["Sale Clifa", 22000], 30: ["Sale Wahyu", 22000],
    31: ["Lanting Udang 200g", 20000], 32: ["Lanting Pedas Manis", 20000],
    33: ["Lanting Jagung", 20000], 34: ["Lanting Keju", 20000], 35: ["Lanting Bawang", 20000],
    36: ["Lanting Original", 18000], 37: ["Lanting 3 Rasa", 22000],
    38: ["Lanting Jumbo", 22000], 39: ["Sondok Vina", 27000], 40: ["Marning Difa", 20000]
}

# Kapasitas restok tetap per produks
caps = {i: random.randint(30, 87) for i in range(1, 41)}
total_sold = {i: 0 for i in range(1, 41)}
last_update = {i: "2023-01-01 08:00:00" for i in range(1, 41)}

# 2. LOGIKA POLA transaksis
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 4, 22)
transactions = []
t_id = 1

curr = start_date
while curr <= end_date:
    # --- FAKTOR POLA ---
    # a. Pola Bulanan (Ramadhan & Desember naik 2x lipat)
    month_multiplier = 2.0 if curr.month in [4, 12] else 1.0
    
    # b. Pola Mingguan (Weekend naik 1.5x lipat)
    day_multiplier = 1.5 if curr.weekday() >= 4 else 1.0 # Jumat-Minggu
    
    # c. Tren Tahunan (Setiap tahun naik 20%)
    year_trend = 1.0 + (curr.year - 2023) * 0.2
    
    # Hitung jumlah transaksis hari ini berdasarkan pola (rata-rata 8-15 transaksis/hari)
    base_daily = random.randint(8, 15)
    daily_count = int(base_daily * month_multiplier * day_multiplier * year_trend)
    
    for _ in range(daily_count):
        p_id = random.randint(1, 40)
        qty = random.randint(1, 5)
        
        total_sold[p_id] += qty
        ts = curr.replace(hour=random.randint(8, 21), minute=random.randint(0, 59))
        
        if ts > datetime.strptime(last_update[p_id], '%Y-%m-%d %H:%M:%S'):
            last_update[p_id] = ts.strftime('%Y-%m-%d %H:%M:%S')
            
        transactions.append({
            "id": t_id,
            "tgl": curr.strftime('%Y-%m-%d'),
            "pid": p_id,
            "qty": qty,
            "total": qty * product_meta[p_id][1],
            "m_name": curr.strftime('%B'),
            "year": curr.year,
            "week": curr.isocalendar()[1],
            "created": ts.strftime('%Y-%m-%d %H:%M:%S')
        })
        t_id += 1
    
    curr += timedelta(days=1)

# 3. GENERATE SQL FILE
with open("data_berpola_3tahun.sql", "w") as f:
    f.write("SET FOREIGN_KEY_CHECKS = 0;\nTRUNCATE TABLE transaksis;\nTRUNCATE TABLE produks;\nSET FOREIGN_KEY_CHECKS = 1;\n\n")
    
    # Insert produks
    f.write("INSERT INTO produks (id, image, nama_produk, harga_produk, stok, total_terjual, kategori_id, status, pemasok, created_at, updated_at) VALUES\n")
    p_vals = []
    for p_id, info in product_meta.items():
        sold = total_sold[p_id]
        cap = caps[p_id]
        cycles = (sold // cap) + 1
        stok_akhir = (cycles * cap) - sold
        status = 'aktif' if random.random() > 0.2 else 'tidak_lanjut'
        p_vals.append(f"({p_id}, NULL, '{info[0]}', {info[1]}, {stok_akhir}, {sold}, NULL, '{status}', 'orisinil', '2023-01-01 00:00:00', '{last_update[p_id]}')")
    f.write(",\n".join(p_vals) + ";\n\n")
    
    # Insert transaksis (Chunked)
    for i in range(0, len(transactions), 1000):
        chunk = transactions[i:i+1000]
        f.write("INSERT INTO transaksis (id, tanggal_transaksi, produks_id, quantity, total, month_name, year, week_number, created_at) VALUES\n")
        t_vals = [f"({t['id']}, '{t['tgl']}', {t['pid']}, {t['qty']}, {t['total']}, '{t['m_name']}', {t['year']}, {t['week']}, '{t['created']}')" for t in chunk]
        f.write(",\n".join(t_vals) + ";\n\n")

print(f"Selesai! {len(transactions)} transaksis berpola telah dibuat di 'data_berpola_3tahun.sql'")