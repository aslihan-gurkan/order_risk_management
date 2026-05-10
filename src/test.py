def validate_all_tables(
    orders, items, reviews, products, customers, sellers, payments
) -> None:
    """Tüm tabloların beklenen kolonlarını kontrol eder."""
    
    validations = {
        "orders": (orders, [
            "order_id", "customer_id", "order_status",
            "order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]),
        "items": (items, [
            "order_id", "product_id", "seller_id", "price", "freight_value"
        ]),
        "reviews": (reviews, [
            "order_id", "review_score", "review_creation_date"
        ]),
        "products": (products, [
            "product_id", "product_category_name"
        ]),
        "customers": (customers, [
            "customer_id", "customer_unique_id"
        ]),
        "sellers": (sellers, [
            "seller_id"
        ]),
        "payments": (payments, [
            "order_id", "payment_type", "payment_installments", "payment_value"
        ]),
    }
    
    for name, (df, required_cols) in validations.items():
        validate_dataframe(df, name, required_cols)


def merge_tables(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tüm tabloları birleştirir.
    
    NOT: orders + items sonrası veri item-level'a iner.
    Sipariş sayısı için nunique(order_id) kullanılmalı,
    count(order_id) yanlış sonuç verir.
    """
    logger.info("Tablolar birlestiriliyor...")

    # orders + items → item-level'a iner, bu beklenen davranış
    df = orders.merge(items, on="order_id", how="left")
    logger.info(
        f"orders + items → {df.shape[0]:,} satir "
        f"({df['order_id'].nunique():,} unique siparis)"
    )

    # + customers
    df = df.merge(customers, on="customer_id", how="left")
    logger.info(f"+ customers → {df.shape[0]:,} satir")

    # + products
    df = df.merge(products, on="product_id", how="left")
    logger.info(f"+ products → {df.shape[0]:,} satir")

    # + sellers
    df = df.merge(sellers, on="seller_id", how="left")
    logger.info(f"+ sellers → {df.shape[0]:,} satir")

    # + payments
    # Birden fazla ödeme tipi varsa hepsini sakla, bilgi kaybetme
    payments_agg = payments.groupby("order_id").agg(
        odeme_turu=("payment_type", lambda x: ",".join(x.dropna().unique())),
        taksit_sayisi=("payment_installments", "max"),
        toplam_odeme=("payment_value", "sum"),
    ).reset_index()
    df = df.merge(payments_agg, on="order_id", how="left")
    logger.info(f"+ payments → {df.shape[0]:,} satir")

    # + reviews
    # review_creation_date önce datetime'a çevrilmeli, sonra sırala
    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"], errors="coerce"
    )
    reviews_agg = (
        reviews.sort_values("review_creation_date")
        .groupby("order_id")
        .last()
        .reset_index()
    )
    df = df.merge(
        reviews_agg[["order_id", "review_score", "review_comment_message"]],
        on="order_id",
        how="left"
    )
    logger.info(f"+ reviews → {df.shape[0]:,} satir")

    return df