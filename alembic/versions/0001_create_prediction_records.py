"""create prediction_records table"""

from alembic import op
import sqlalchemy as sa


revision = "0001_create_prediction_records"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("predicted_price", sa.Float(), nullable=False),
        sa.Column("median_income", sa.Float(), nullable=False),
        sa.Column("house_age", sa.Float(), nullable=False),
        sa.Column("average_rooms", sa.Float(), nullable=False),
        sa.Column("average_bedrooms", sa.Float(), nullable=False),
        sa.Column("population", sa.Float(), nullable=False),
        sa.Column("average_occupancy", sa.Float(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("prediction_records")
