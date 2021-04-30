from typing import Dict, Sequence, Tuple
import datetime


import numpy as np
from torch.utils.data import Dataset

from clickhouse_driver import Client
from clickhouse_utils.schema import dtypes_from_table
from ml_utils.data.sequence_dataset import BatchingSequenceDataset


class ClickhouseBatchingSequenceDataset(BatchingSequenceDataset):
    def __init__(
        self,
        host: str,
        database: str,
        table_name: str,
        uid_name: str,
        time_col: str,
        asof_time: datetime.datetime,  # return no records after this
        min_items_per_uid: int = 1,
    ):
        self.host = host
        self.conn = Client(host)
        self.database = database
        self.table_name = table_name
        self.uid_name = uid_name
        self.time_col = time_col
        self.asof_time = asof_time

        # get all the UIDs with at least min_items_per_uid events
        date_filt = self.asof_filter.replace("and", "where") if self.asof_filter else ""

        uid_query = f"""SELECT * from (
        SELECT {uid_name}, 
            count(*) as cnt,
            min({time_col}) as first_t,
            min({time_col}) as last_t
        from {database}.{table_name} 
        {date_filt}
        group by {uid_name} 
        order by {uid_name}
        ) as tmp 
        where cnt >={min_items_per_uid}
        """
        result = np.array(self.conn.execute(uid_query))
        # ordered list of all the ids we're considering
        self.uids = result[:, 0]
        self.first_t = result[:, 2]
        self.last_t = result[:, 3]

        # number of events for each ID
        self.uid_sizes = {x[0]: x[1] for x in result}

    def __getstate__(self):
        out = self.__dict__.copy()
        del out["conn"]
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.conn = Client(self.host)

    @property
    def asof_filter(self):
        return (
            ""
            if self.asof_time is None
            else f" and {self.time_col} < toDate('{self.asof_time}')"
        )

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i: int) -> int:
        return self.uid_sizes[self.uids[i]]

    def get_bulk(self, inds: Sequence[int]) -> Sequence[Dict[str, np.ndarray]]:
        # get sequences for a list of UIDS,
        # so we call the database only once
        uids = np.array(sorted([self.uids[i] for i in inds]))
        # get the data for all the ids
        query = f"""
            SELECT * from {self.database}.{self.table_name}
            where {self.uid_name} in ({','.join(uids.astype(str))})
            {self.asof_filter}
            order by {self.uid_name}, {self.time_col}
        """
        raw_data = self.conn.execute(query)
        data = np.array(raw_data).T
        pre_out = {}

        # get the data types and column names
        dtypes = dtypes_from_table(self.host, self.database, table=self.table_name)

        # match data to column names and cast the variables to correct types
        for x, (cname, ctype) in zip(data, dtypes.iteritems()):
            pre_out[cname] = x.astype(ctype)
            if cname == self.time_col:
                pre_out["t"] = x

        # slice it up by ID and apply the transform
        seqs = []
        offsets = [0]
        # split the query result into sequences by uid
        for next_item in uids:
            len_ = self.uid_sizes[next_item]
            offsets.append(offsets[-1] + len_)
            # split out the data for a particular ID
            this_seq = {k: v[offsets[-2] : offsets[-1]] for k, v in pre_out.items()}
            seqs.append(this_seq)

        return seqs
