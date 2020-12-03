#Chata Technologies Inc 2020

"""
Created on 2019-03-19
@author: duytinvo
This module is used to generate parsed_sql from sql files which are saved in the json format
"""
################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except

# val: number(float)/string(str)/sql(dict)/tuple(agg_op_id, col_id, isDistinct)
# col_unit: (agg_op_id, col_id, isDistinct(bool))/None
# val_unit: (unit_op_id, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: [not_op_id(bool), WHERE_op_id, val_unit, val1, val2]
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]


# sql {
#   'select': (isDistinct(bool), [(agg_op_id, val_unit), (agg_op_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit_value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################
import re
import os
import json
import copy
import pandas as pd
from collections import defaultdict
from nltk import word_tokenize
# import settings

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
# settings.date_agg = True
# if settings.date_agg:
#     # TODO: ask @Henning when 'day', 'week', 'month', 'year' appear in the QUERY
#     AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg', 'day', 'week', 'month', 'year')
# else:
#     AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg', 'day', 'week', 'month', 'year')
TABLE_TYPE = {'sql': "sql", 'table_unit': "table_unit"}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


class sqlParser(object):
    def __init__(self, schema, tables=None, original=True):
        self.schema = schema
        self.tb_names = list(schema.keys())
        self.tables_with_alias = defaultdict()
        self.tbcoldict = self.tbcolmap() if not original else self._map(tables)

    def _map(self, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        for i, v in enumerate(column_names_original):
            if len(v) == 3:
                tab_id, col_name, _ = v
            else:
                tab_id, col_name = v
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col_name.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap

    def tbcolmap(self):
        """
        :return: dictionary containing table and table.column
        """
        idMap = {'*': "__all__"}
        idx = 1
        for key, cols in self.schema.items():
            # cols = vals[-1].keys()
            for val in cols:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                idx += 1
        for key in self.schema.keys():
            idMap[key.lower()] = "__" + key.lower() + "__"
            idx += 1
        return idMap

    @staticmethod
    def load_json(fpath):
        with open(fpath) as f:
            data = json.load(f)
        return data

    @staticmethod
    def tokenize(string):
        string = str(string)
        string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
        quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
        # string = string.replace("\"", "\'")  # ensures all string values wrapped by '' problem??
        # quote_idxs = [idx for idx, char in enumerate(string) if char == "\'"]
        assert len(quote_idxs) % 2 == 0, "Unexpected quote: {}".format(string)
        # keep string value as token
        vals = {}
        for i in range(len(quote_idxs) - 1, -1, -2):
            qidx1 = quote_idxs[i - 1]
            qidx2 = quote_idxs[i]
            val = string[qidx1: qidx2 + 1]
            key = "__val_{}_{}__".format(qidx1, qidx2)
            string = string[:qidx1] + key + string[qidx2 + 1:]
            vals[key] = val

        toks = [word.lower() for word in word_tokenize(string)]
        # replace with string value token
        for i in range(len(toks)):
            if toks[i] in vals:
                toks[i] = vals[toks[i]]
        # find if there exists !=, >=, <=
        eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
        eq_idxs.reverse()
        prefix = ('!', '>', '<')
        for eq_idx in eq_idxs:
            pre_tok = toks[eq_idx - 1]
            if pre_tok in prefix:
                toks = toks[:eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1:]
        return toks

    def get_tables_with_alias(self, toks):
        """
        :return: a dict {alias: tb_name; tb_name: tb_name}
        """
        # Scan the index of 'as' and build the map for all alias
        as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
        alias = {}
        for idx in as_idxs:
            alias[toks[idx + 1]] = toks[idx - 1]
        # set table names itself as aliases
        for key in self.tb_names:
            assert key not in alias, "Alias {} has the same name in table".format(key)
            alias[key] = key
        return alias

    @staticmethod
    def skip_semicolon(toks, start_idx):
        idx = start_idx
        while idx < len(toks) and toks[idx] == ";":
            idx += 1
        return idx

    def parse_table_unit(self, toks, start_idx):
        """
            :returns next_idx, table_id, table_name
            e.g. 5, __department__, department
        """
        idx = start_idx
        len_ = len(toks)
        key = self.tables_with_alias[toks[idx]]

        if idx + 1 < len_ and toks[idx + 1] == "as":
            idx += 3
        else:
            idx += 1
        return idx, self.tbcoldict[key], key

    def parse_col(self, toks, start_idx, default_tables=None):
        """
            :returns next_idx, column_id (table[_id]/table.column[_id])
            e.g. 3, __department.staff_id__
        """
        tok = toks[start_idx]
        # * == __all__
        if tok == "*":
            return start_idx + 1, self.tbcoldict[tok]

        if '.' in tok:  # if token is a composite between table.column
            alias, col = tok.split('.')
            # parse the alias name of a table into the original name
            key = self.tables_with_alias[alias] + "." + col
            return start_idx + 1, self.tbcoldict[key]

        assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

        for alias in default_tables:
            table = self.tables_with_alias[alias]
            # table_names
            if tok in self.schema[table]:
                # if tok in self.schema[table][-1].keys():
                key = table + "." + tok
                return start_idx + 1, self.tbcoldict[key]
        assert False, "Error col: {}, {}".format(tok, toks)

    def parse_col_unit(self, toks, start_idx, default_tables=None):
        """
            :returns next_idx, (agg_op_id, col_id, isDistinct)
            e.g. col_unit = (agg_op_id, col_id, isDistinct)
                          = (3_count, __department.courseid__/0, True)
                          = DISTINCT COUNT(__department.courseid__)
        """
        idx = start_idx
        len_ = len(toks)
        isBlock = False
        isDistinct = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] in AGG_OPS:
            # agg_op_id = ('none', 'max', 'min', 'count', 'sum', 'avg')
            agg_op_id = AGG_OPS.index(toks[idx])
            idx += 1
            # assert idx < len_ and toks[idx] == '('
            if idx < len_ and toks[idx] == '(':
                idx += 1
                if toks[idx] == "distinct":
                    idx += 1
                    isDistinct = True
                idx, col_id = self.parse_col(toks, idx, default_tables)
                assert idx < len_ and toks[idx] == ')'
                idx += 1
                return idx, (agg_op_id, col_id, isDistinct)

        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        agg_op_id = AGG_OPS.index("none")
        idx, col_id = self.parse_col(toks, idx, default_tables)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1  # skip ')'

        return idx, (agg_op_id, col_id, isDistinct)

    def parse_val_unit(self, toks, start_idx, default_tables=None):
        """
        :return: next_idx, (unit_op_id, col_unit1, col_unit2)
        e.g.    val_unit = (unit_op_id, col_unit1, col_unit2)
                unit_op_id = 1_'-'
                col_unit1 = (agg_op_id, col_id, isDistinct)
                          = (3_count, __department.courseid__/0, True)
                          = DISTINCT COUNT(__department.courseid__)
                (unit_op_id, col_unit1, col_unit2) = col_unit1 unit_op_id col_unit2
        """
        idx = start_idx
        len_ = len(toks)
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        col_unit1, col_unit2 = None, None
        # unit_op_id = ('none', '-', '+', '*', '/')
        unit_op_id = UNIT_OPS.index('none')
        # col_unit1 = (agg_op_id, col_id, isDistinct)
        idx, col_unit1 = self.parse_col_unit(toks, idx, default_tables)
        if idx < len_ and toks[idx] in UNIT_OPS:
            # unit_op_id = ('none', '-', '+', '*', '/')
            unit_op_id = UNIT_OPS.index(toks[idx])
            idx += 1
            # col_unit2 = (agg_op_id, col_id, isDistinct)
            idx, col_unit2 = self.parse_col_unit(toks, idx, default_tables)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1  # skip ')'

        return idx, (unit_op_id, col_unit1, col_unit2)

    def parse_value(self, toks, start_idx, default_tables=None):
        """
        :return: next_idx, val: number(float)/string(str)/sql(dict)/tuple(agg_op_id, col_id, isDistinct)
        val could be a number, string, col_unit or sql
        """
        idx = start_idx
        len_ = len(toks)

        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            # return sql
            idx, val = self.parse_sql(toks, idx)
        elif "\"" in toks[idx]:
            if not isBlock:
                # token is a string value
                val = toks[idx]
                idx += 1
            else:
                val = []
                while isBlock:
                    if toks[idx] == ')':
                        isBlock = False
                        idx += 1
                    elif toks[idx] == ',':
                        idx += 1
                    else:
                        val += [toks[idx]]
                        idx += 1
        else:
            try:
                # token is a float value
                val = float(toks[idx])
                idx += 1
            except:
                end_idx = idx
                while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')' \
                        and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS \
                        and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1
                # val = (agg_op_id, col_id, isDistinct)
                idx, val = self.parse_col_unit(toks[start_idx: end_idx], 0, default_tables)
                idx = end_idx

        if isBlock:
            assert toks[idx] == ')', print(toks)
            idx += 1

        return idx, val

    def parse_condition(self, toks, start_idx, default_tables=None):
        """
        :return: next_idx, conds: [cond_unit1, 'and'/'or', cond_unit2, ...]
        e.g.    cond_unit: [not_op(bool), op_id, val_unit, val1, val2]
                val_unit: (unit_op_id, col_unit1, col_unit2)
        """
        idx = start_idx
        len_ = len(toks)
        conds = []

        while idx < len_:
            # val_unit = (unit_op_id, col_unit1, col_unit2)
            idx, val_unit = self.parse_val_unit(toks, idx, default_tables)
            not_op = False
            if toks[idx] == 'not':
                not_op = True
                idx += 1

            assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
            # op_id = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
            op_id = WHERE_OPS.index(toks[idx])
            idx += 1
            # between..and... special case: dual values
            if op_id == WHERE_OPS.index('between'):
                # val1 = number(float)/string(str)/sql(dict)/tuple(agg_op_id, col_id, isDistinct)
                idx, val1 = self.parse_value(toks, idx, default_tables)
                assert toks[idx] == 'and'
                idx += 1
                # val2 = number(float)/string(str)/sql(dict)/tuple(agg_op_id, col_id, isDistinct)
                idx, val2 = self.parse_value(toks, idx, default_tables)
            # normal case: single value
            else:
                # val1 = number(float)/string(str)/sql(dict)/tuple(agg_op_id, col_id, isDistinct)
                idx, val1 = self.parse_value(toks, idx, default_tables)
                val2 = None

            conds.append((not_op, op_id, val_unit, val1, val2))

            if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
                break

            if idx < len_ and toks[idx] in COND_OPS:
                conds.append(toks[idx])
                idx += 1  # skip and/or
        return idx, conds

    def parse_from(self, toks, start_idx):
        """
        :return: idx, table_units, conds, default_tables
        Parse `FROM clause` (including block) until encountering next clause keywords
        Assume in the from clause, all table units are combined with join
        e.g.
         - FROM artists AS T1 JOIN paintings AS T2 ON T1.artistID  =  T2.painterID
         - FROM (SELECT * FROM artists)
        """
        assert 'from' in toks[start_idx:], "'from' not found"
        len_ = len(toks)
        # return the index of the first FROM token starting from start_idx
        idx = toks.index('from', start_idx) + 1
        # default_tables changes depending on FROM clause of each sub_query
        default_tables = []
        table_units = []
        conds = []
        while idx < len_:
            isBlock = False
            if toks[idx] == '(':
                isBlock = True
                idx += 1
            # If there is a sub-query in the block
            if toks[idx] == 'select':
                # return sql
                idx, sql = self.parse_sql(toks, idx)
                table_units.append((TABLE_TYPE['sql'], sql))
            else:
                if idx < len_ and toks[idx] == 'join':
                    idx += 1  # skip join
                # look up tables used in the query
                # next_idx, __table_id__, table_name
                idx, table_unit, table_name = self.parse_table_unit(toks, idx)
                table_units.append((TABLE_TYPE['table_unit'], table_unit))
                default_tables.append(table_name)
            # Search for condition
            if idx < len_ and toks[idx] == "on":
                idx += 1  # skip on
                # this_conds: [cond_unit1, 'and'/'or', cond_unit2, ...];
                # cond_unit: [not_op(bool), op_id, val_unit, val1, val2]
                idx, this_conds = self.parse_condition(toks, idx, default_tables)
                if len(conds) > 0:
                    conds.append('and')
                conds.extend(this_conds)

            if isBlock:
                assert toks[idx] == ')'
                idx += 1
            if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
                break
        return idx, table_units, conds, default_tables

    def parse_select(self, toks, start_idx, default_tables=None):
        """
        :return: idx, (isDistinct, val_units)
        e.g.    (isDistinct, (unit_op_id, col_unit1, col_unit2))
                col_unit = (agg_op_id, col_id, isDistinct)
        """
        idx = start_idx
        len_ = len(toks)

        assert toks[idx] == 'select', "'select' not found"
        idx += 1
        isDistinct = False
        if idx < len_ and toks[idx] == 'distinct':
            idx += 1
            isDistinct = True
        val_units = []

        while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
            agg_id = AGG_OPS.index("none")
            if toks[idx] in AGG_OPS:
                agg_id = AGG_OPS.index(toks[idx])
                idx += 1
            # val_unit: (unit_op_id, col_unit1, col_unit2)
            idx, val_unit = self.parse_val_unit(toks, idx, default_tables)
            val_units.append((agg_id, val_unit))
            if idx < len_ and toks[idx] == ',':
                idx += 1  # skip ','
        return idx, (isDistinct, val_units)

    def parse_where(self, toks, start_idx, default_tables):
        """
        :return:  idx, conds: [cond_unit1, 'and'/'or', cond_unit2, ...]
        e.g.    cond_unit: (not_op(bool), op_id, val_unit, val1, val2)
                val_unit: (unit_op_id, col_unit1, col_unit2)
        """
        idx = start_idx
        len_ = len(toks)

        if idx >= len_ or toks[idx] != 'where':
            return idx, []

        idx += 1
        idx, conds = self.parse_condition(toks, idx, default_tables)
        return idx, conds

    def parse_group_by(self, toks, start_idx, default_tables):
        """
        :return: idx, col_units
        e.g.    col_units: (not_op(bool), op_id, val_unit, val1, val2)
                val_unit: (unit_op_id, col_unit1, col_unit2)
        """
        idx = start_idx
        len_ = len(toks)
        col_units = []

        if idx >= len_ or toks[idx] != 'group':
            return idx, col_units

        idx += 1
        assert toks[idx] == 'by'
        idx += 1

        while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, col_unit = self.parse_col_unit(toks, idx, default_tables)
            col_units.append(col_unit)
            if idx < len_ and toks[idx] == ',':
                idx += 1  # skip ','
            else:
                break

        return idx, col_units

    def parse_having(self, toks, start_idx, default_tables):
        """
        :return: idx, conds: [cond_unit1, 'and'/'or', cond_unit2, ...]
        e.g.    cond_unit: (not_op(bool), op_id, val_unit, val1, val2)
                val_unit: (unit_op_id, col_unit1, col_unit2)
        """
        idx = start_idx
        len_ = len(toks)

        if idx >= len_ or toks[idx] != 'having':
            return idx, []

        idx += 1
        idx, conds = self.parse_condition(toks, idx, default_tables)
        return idx, conds

    def parse_order_by(self, toks, start_idx, default_tables):
        """
        :return: idx, (order_type, val_units)
        e.g.    val_unit: (unit_op_id, col_unit1, col_unit2)
                order_type: asc
        """
        idx = start_idx
        len_ = len(toks)
        val_units = []
        order_type = 'asc'  # default type is 'asc'

        if idx >= len_ or toks[idx] != 'order':
            return idx, val_units

        idx += 1
        assert toks[idx] == 'by'
        idx += 1

        while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            idx, val_unit = self.parse_val_unit(toks, idx, default_tables)
            val_units.append(val_unit)
            if idx < len_ and toks[idx] in ORDER_OPS:
                order_type = toks[idx]
                idx += 1
            if idx < len_ and toks[idx] == ',':
                idx += 1  # skip ','
            else:
                break

        return idx, (order_type, val_units)

    @staticmethod
    def parse_limit(toks, start_idx):
        """
        :return: idx, num
        """
        idx = start_idx
        len_ = len(toks)
        if idx < len_ and toks[idx] == 'limit':
            idx += 2
            # TODO: removed int(toks[idx - 1])
            return idx, toks[idx - 1]
        return idx, None

    def parse_sql(self, toks, start_idx):
        """
        :return: idx, sql
        """
        isBlock = False  # indicate whether this is a block of sql/sub-sql
        len_ = len(toks)
        idx = start_idx
        sql = {}
        if toks[idx] == '(':
            isBlock = True
            idx += 1
        # parse from clause in order to get default tables
        from_end_idx, table_units, conds, default_tables = self.parse_from(toks, start_idx)
        sql['from'] = {'table_units': table_units, 'conds': conds}
        # select clause
        _, select_col_units = self.parse_select(toks, idx, default_tables)
        idx = from_end_idx
        sql['select'] = select_col_units
        # where clause
        idx, where_conds = self.parse_where(toks, idx, default_tables)
        sql['where'] = where_conds
        # group by clause
        idx, group_col_units = self.parse_group_by(toks, idx, default_tables)
        sql['groupBy'] = group_col_units
        # having clause
        idx, having_conds = self.parse_having(toks, idx, default_tables)
        sql['having'] = having_conds
        # order by clause
        idx, order_col_units = self.parse_order_by(toks, idx, default_tables)
        sql['orderBy'] = order_col_units
        # limit clause
        idx, limit_val = sqlParser.parse_limit(toks, idx)
        sql['limit'] = limit_val

        idx = sqlParser.skip_semicolon(toks, idx)
        if isBlock:
            assert toks[idx] == ')'
            idx += 1  # skip ')'
        idx = sqlParser.skip_semicolon(toks, idx)

        # intersect/union/except clause
        for op in SQL_OPS:  # initialize IUE
            sql[op] = None
        if idx < len_ and toks[idx] in SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, IUE_sql = self.parse_sql(toks, idx)
            sql[sql_op] = IUE_sql
        return idx, sql

    def get_sql(self, query):
        """
        :return: sql
        """
        toks = sqlParser.tokenize(query)
        # scan through query to get all alias
        self.tables_with_alias = self.get_tables_with_alias(toks)
        _, sql = self.parse_sql(toks, 0)
        return sql


class csv2json(object):
    def __init__(self, wtfrom=False, dbid=None):
        self.wtfrom = wtfrom
        self.dbid = dbid

    @staticmethod
    def get_schemas_from_json(fpath):
        with open(fpath) as f:
            data = json.load(f)
        db_names = [db['db_id'] for db in data]
        tables = {}
        schemas = {}
        for db in data:
            db_id = db['db_id']
            schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
            column_names_original = db['column_names_original']
            table_names_original = db['table_names_original']
            tables[db_id] = {'column_names_original': column_names_original,
                             'table_names_original': table_names_original}
            for i, tabn in enumerate(table_names_original):
                table = str(tabn.lower())
                cols = [str(col[1].lower()) for col in column_names_original if col[0] == i]
                schema[table] = cols
            schemas[db_id] = schema
        return schemas, db_names, tables

    def process_CSV(self, labeled_csv):
        df = pd.read_csv(labeled_csv)
        if len(df.columns) == 2:
            df.columns = ['Eng_query', 'SQL_query']  # , 'Eng_Answer']
        if len(df.columns) == 3:
            df.columns = ['Eng_query', 'SQL_query', 'SQL_query_no_from']
        if len(df.columns) == 4:
            df.columns = ['Eng_query', 'SQL_query', 'SQL_query_no_from', 'Eng_query_mapper']

        training_list_global = []

        for _, row in df.iterrows():
            eng_query = row['Eng_query']
            # if self.wtfrom and len(df.columns) >= 3:
            #     sql_query = row['SQL_query_no_from']
            # else:
            #     sql_query = row['SQL_query']
            sql_query = row['SQL_query']
            label_dict = dict()
            if "db_id" not in df.columns:
                label_dict['db_id'] = self.dbid
            else:
                label_dict['db_id'] = row['db_id']
            # sql_query = self.preprocess_sql(sql_query)
            label_dict['query'] = sql_query
            label_dict['question'] = eng_query
            training_list_global.append(label_dict)
        return training_list_global

    @staticmethod
    def preprocess_sql(sql_query):
        # print(sql_query)
        sql_query = sql_query.replace("`", "")  # remove `` used in col and tb names
        # sql_query = sql_query.replace("?", "'terminal'")  # remove `` used in col and tb names
        sql_query = sql_query.replace("groupBy", "GROUP BY")  # remove inner join = Join
        sql_query = sql_query.replace("orderBy", "ORDER BY")  # remove inner join = Join
        sql_query = sql_query.replace("inner join", "join")  # remove inner join = Join
        sql_query = sql_query.replace("T5 T1.currency_id = T5.id", "T5 on T1.currency_id = T5.id")  # remove inner join = Join
        # matchObjs = re.findall(r'\w+\.*\w+ in \( .*? \)', sql_query, re.M|re.I)
        # for matchObj in matchObjs:
        #     col = matchObj.split(" in ")[0]
        #     vls = re.findall(r'\( (.*?) \)', matchObj, re.M|re.I)[0].split(",")
        return sql_query

    @staticmethod
    def write_to_json_file(data, json_file):
        if not os.path.exists(os.path.dirname(json_file)):
            os.mkdir(os.path.dirname(json_file))

        with open(json_file, 'w') as outfile:
            json.dump(data, outfile, indent=2)

    def build(self, labeled_csv, json_file):
        data = self.process_CSV(labeled_csv)
        self.write_to_json_file(data, json_file)


if __name__ == '__main__':
    """
    Parse SQL to a tree consisting different component in a hierarchy structure 
    """
    import settings
    # settings.date_agg = True
    if settings.date_agg:
        # TODO: ask @Henning when 'day', 'week', 'month', 'year' appear in the QUERY
        AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg', 'day', 'week', 'month', 'year')
    else:
        AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')

    csv2json_gen = csv2json(use_sql=False, wtfrom=False)

    csv2json_gen.build(labeled_csv="../../data_locate/csv/human_train.csv", json_file="../../data_locate/nl2sql/human.json")


