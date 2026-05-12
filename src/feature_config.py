"""
Model feature konfigürasyonu.

Amaç:
    Modele verilecek/verilmeyecek kolonları tek merkezden yönetmek.
"""

# Modele verilmemesi gereken kolonlar
DROP_COLUMNS = [
    # ID / foreign key kolonları
    "order_id",
    "customer_id",
    "customer_unique_id",
    "main_seller_id",

    # Leakage riski taşıyan kolonlar
    "order_status",

    # Gürültülü veya yüksek cardinality kolonlar
    "customer_city",
    "customer_zip_code_prefix",

    # Raw datetime kolonları
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",

    # Historical raw count kolonları
    "seller_historical_problem_count",
    "seller_historical_order_count",
    "category_historical_problem_count",
    "category_historical_order_count",
    "customer_historical_problematic",
    "customer_historical_orders",
]

# High-cardinality categorical kolonlar
HIGH_CARDINALITY_COLS = [
    "main_product_category",
]

# Low-cardinality categorical kolonlar
LOW_CARDINALITY_COLS = [
    "payment_type",
    "customer_state",
    "seller_state_mode",
]

TARGET_COLUMN = "problematic_order"