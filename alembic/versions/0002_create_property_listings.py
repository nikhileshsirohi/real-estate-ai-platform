"""create property_listings table"""

from alembic import op
import sqlalchemy as sa


revision = "0002_create_property_listings"
down_revision = "0001_create_prediction_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "property_listings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("listing_code", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=False),
        sa.Column("locality", sa.String(length=150), nullable=False),
        sa.Column("property_type", sa.String(length=50), nullable=False),
        sa.Column("bedrooms", sa.Integer(), nullable=False),
        sa.Column("bathrooms", sa.Float(), nullable=False),
        sa.Column("area_sqft", sa.Float(), nullable=False),
        sa.Column("asking_price_usd", sa.Float(), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("listing_code"),
    )
    op.create_index("ix_property_listings_listing_code", "property_listings", ["listing_code"], unique=False)
    op.create_index("ix_property_listings_city", "property_listings", ["city"], unique=False)
    op.create_index("ix_property_listings_locality", "property_listings", ["locality"], unique=False)
    op.create_index("ix_property_listings_property_type", "property_listings", ["property_type"], unique=False)
    op.create_index("ix_property_listings_asking_price_usd", "property_listings", ["asking_price_usd"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_property_listings_asking_price_usd", table_name="property_listings")
    op.drop_index("ix_property_listings_property_type", table_name="property_listings")
    op.drop_index("ix_property_listings_locality", table_name="property_listings")
    op.drop_index("ix_property_listings_city", table_name="property_listings")
    op.drop_index("ix_property_listings_listing_code", table_name="property_listings")
    op.drop_table("property_listings")
