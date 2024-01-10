import plotly.express as px
from enum import Enum

USERNAME = "snowflake_username"  # your username
ACCOUNT = "account"
REGION = "region"
AUTHENTICATOR = "EXTERNALBROWSER"
DATABASE = "db"
WAREHOUSE = "warehouse"
# path to table with final results
RESULT_DATABASE = "SANDBOX_DB"
# path to table with final results
RESULT_SCHEMA = "sandbox_firstname_lastname"
# path to table with final results
RESULT_TABLE = "result_table"
# path to table with user_id, timestamp
SOURCE_DATABASE = "source_db"
# path to table with user_id, timestamp
SOURCE_SCHEMA = "source_schema"
# path to table with user_id, timestamp
SOURCE_TABLE = "source_table"
# user_id or profile_id column name (if profile_id, then set is_profile_id=True)
ID_COLUMN = "id_column"
# timestamp column name
TIMESTAMP_COLUMN = "timestamp_column"

WISE_COLORS = dict(
    dark_green="#163300",
    bright_orange="#ffc091",
    bright_blue="#a0e1e1",
    bright_yellow="#ffeb69",
    dark_purple="#ca7ce3",  #'#260a2f',
    dark_charcoal="#21231d",
    dark_gold="#ae9d58",  #'#3a341c',
    bright_pink=" #ffd7ef",
    dark_maroon="#320707",
    bright_light_green="#9fe870",
)

PLOTLY_COLOR_PALETTE = list(WISE_COLORS.values()) + px.colors.qualitative.G10
COLORSCALES = ["Blues", "Greens", "Purples", "Reds", "Oranges", "Greys", "BuGn"]


class MetricType(Enum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
