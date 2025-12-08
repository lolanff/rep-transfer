from collections.abc import Callable, Iterable, Sequence
import importlib
from pathlib import Path
from PyExpUtils.models.ExperimentDescription import ExperimentDescription, loadExperiment
from PyExpUtils.results.tools import getHeader, getParamsAsDict
from PyExpUtils.results.indices import listIndices
from typing import Any
import connectorx as cx
from sqlite3 import Cursor
from functools import reduce
import sqlite3
import polars as pl
from typing import TypeVar, Generic
import time

Exp = TypeVar('Exp', bound=ExperimentDescription)

def maybe_quote(v: Any):
    if isinstance(v, str):
        return quote(v)
    return v

def quote(s: str):
    return f'"{s}"'

def get_tables(cur: Cursor) -> set[str]:
    cur.row_factory = None
    res = cur.execute("SELECT name FROM sqlite_master")
    return set(r[0] for r in res.fetchall())

def get_run_ids(db_path: str | Path, params: dict[str, Any]):
    meta = cx.read_sql(f'sqlite://{db_path}', 'SELECT * FROM _metadata_', return_type='polars')

    f = pl.lit(True)
    for k, v in params.items():
        f = f & (pl.col(k) == v)

    return meta.filter(f)['id'].to_list()

def read_metrics(db_path: str | Path, metrics: Iterable[str], ids: Iterable[int] | None = None):
    dfs = list(map(lambda m: read_to_df(db_path, m, ids), metrics))
    df = reduce(lambda df1, df2: df1.join(df2, how='full', on=['id', 'frame'], coalesce=True), dfs)
    df = df.sort('frame')
    return df

def read_to_df(db_path: str | Path, metric: str, ids: Iterable[int] | None = None):
    db_path = Path(db_path)
    constraints = ''
    if ids is not None:
        str_ids = map(str, map(maybe_quote, ids))
        constraints = f'WHERE id IN ({",".join(str_ids)})'
    query = f'SELECT * FROM {metric} {constraints}'
    df = cx.read_sql(f'sqlite://{db_path}', query, partition_on='id', partition_num=1, return_type='polars')
    df = df.rename({'measurement': metric})
    return df.lazy()

def load_all_results(db_path: str | Path, metrics: Iterable[str] | None = None, ids: Iterable[int] | None = None):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = get_tables(cur)
    if metrics is None:
        metrics = tables - {'_metadata_'}
    df = read_metrics(db_path, metrics, ids)
    if '_metadata_' not in tables:
        return df.collect()

    meta = cx.read_sql(f'sqlite://{db_path}', 'SELECT * FROM _metadata_', return_type='polars', partition_on='id', partition_num=1)
    meta = meta.lazy()
    return df.join(meta, how='left', on=['id']).collect()

class Result(Generic[Exp]):
    def __init__(self, exp_path: str | Path, exp: Exp, metrics: Sequence[str] | None = None):
        self.exp_path = str(exp_path)
        self.exp = exp
        self.metrics = metrics

    def load(self):
        db_path = self.exp.buildSaveContext(0).resolve('results.db')

        if not Path(db_path).exists():
            return None

        print(f'Loading {self.filename}...', end='', flush=True)
        start = time.time()
        try:
            meta = cx.read_sql(f'sqlite://{db_path}', 'SELECT * FROM _metadata_', return_type='polars')
            print(' done! ', end='', flush=True)
        except Exception:
            # table might not exist
            return None

        all_run_ids = []
        for param_id in range(self.exp.numPermutations()):
            params = getParamsAsDict(self.exp, param_id)

            filt = meta
            for k, v in params.items():
                filt = filt.filter(pl.col(k) == v)

            run_ids = filt['id'].to_list()
            all_run_ids.extend(run_ids)

        if not all_run_ids:
            return None
        print(f' found {len(all_run_ids)} runs. Loading metrics...', end='', flush=True)
        df = load_all_results(db_path, self.metrics, all_run_ids)
        end = time.time()
        print(f' done! Took {end - start:.2f}s for {len(all_run_ids)} runs')
        return df

    @property
    def filename(self):
        return self.exp_path.split('/')[-1].removesuffix('.json')


class ResultCollection(Generic[Exp]):
    def __init__(self, path: str | Path | None = None, metrics: Sequence[str] | None = None, Model: type[Exp] = ExperimentDescription):
        self.metrics = metrics
        self.Model = Model

        if path is None:
            main_file = importlib.import_module('__main__').__file__
            assert main_file is not None
            path = Path(main_file).parent

        self.path = Path(path)

        project = Path.cwd()
        paths = self.path.glob('**/*.json')
        paths = map(lambda p: p.relative_to(project), paths)
        paths = map(str, paths)
        self.paths = list(paths)


    def _result(self, path: str):
        exp = loadExperiment(path, self.Model)
        return Result[Exp](path, exp, self.metrics)


    def get_hyperparameter_columns(self):
        hypers = set[str]()

        for path in self.paths:
            exp = loadExperiment(path, self.Model)
            hypers |= set(getHeader(exp))

        return sorted(hypers)


    def groupby_directory(self, level: int):
        uniques = set(
            p.split('/')[level] for p in self.paths
        )

        for group in uniques:
            group_paths = [p for p in self.paths if p.split('/')[level] == group]
            results = map(self._result, group_paths)
            yield group, list(results)


    def __iter__(self):
        return map(self._result, self.paths)


def detect_missing_indices(exp: ExperimentDescription, runs: int, base: str = './'):
    context = exp.buildSaveContext(0, base=base)
    header = getHeader(exp)
    path = context.resolve('results.db')

    if not context.exists('results.db'):
        yield from listIndices(exp, runs)
        return

    n_params = exp.numPermutations()
    for param_id in range(n_params):
        run_ids = set(get_run_ids(path, getParamsAsDict(exp, param_id, header=header)))

        for seed in range(runs):
            run_id = seed * n_params + param_id
            if run_id not in run_ids:
                yield run_id


def gather_missing_indices(experiment_paths: Iterable[str], runs: int, loader: Callable[[str], ExperimentDescription] = loadExperiment, base: str = './'):
    path_to_indices: dict[str, list[int]] = {}

    for path in experiment_paths:
        exp = loader(path)
        indices = detect_missing_indices(exp, runs, base=base)
        indices = sorted(indices)
        path_to_indices[path] = indices

        size = exp.numPermutations() * runs
        print(path, f'{len(indices)} / {size}')

    return path_to_indices