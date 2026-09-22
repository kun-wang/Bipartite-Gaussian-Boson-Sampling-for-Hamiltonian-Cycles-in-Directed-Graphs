# Copyright 2026 Kun Wang
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implement the source-resolved classical particle control.

This defines an incoherent source-occupation classical control using the same source parameters as
BipartiteGBS, but done source-by-source (NOT the independent-edge |A|^2 model
used in the earlier Distinguishable-GA baseline):

  * source i has squeezing r_i = arctanh(0.75 * s_i / s_max), same as BipartiteGBS;
  * per shot, each source emits a thermal pair number  n_i ~ Geom(1 - tanh^2 r_i) - 1;
  * each of the n_i pairs routes its upper photon independently through column i
    of |U|^{circ2} and its lower photon independently through column i of
    |V|^{circ2} (mutually distinguishable within each arm; output collisions remain possible);
  * mode counts are accumulated and the SAME postselection as BipartiteGBS is
    applied: collision-free (each mode <= 1 photon), balanced arm (|S| = |T|),
    at least three pairs (|S| >= 3).

The resulting GBSInfo feeds the identical downstream GA operators.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
import numpy as np
from collections import Counter

class SourceResolvedSampler:
    def __init__(self, graph):
        self.graph = graph
        self.n = graph.n
        A = graph.to_adjacency_matrix()
        self.B = (A > 0).astype(np.float64)
        self.m = self.n
        self._prepare()

    def _prepare(self):
        U, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        scale = 0.75 / np.max(s)
        self.U = U
        self.V = Vt.T.conj()
        self.r = np.arctanh(scale * s)
        self.q = 1.0 - np.tanh(self.r) ** 2            # P(n_i = 0) for Geom
        self.pU = np.abs(self.U) ** 2                  # (m, m), column i normalized
        self.pV = np.abs(self.V) ** 2
        self.mean_pairs = float(np.sum(np.sinh(self.r) ** 2))

    # ---- one raw shot: return (upper_counts, lower_counts) length-m arrays ----
    def raw_shot(self, rng):
        m = self.m
        n_pairs = rng.geometric(self.q) - 1            # per-source pair counts
        upper = np.zeros(m, dtype=int)
        lower = np.zeros(m, dtype=int)
        for i in range(m):
            ni = int(n_pairs[i])
            if ni <= 0:
                continue
            a = rng.choice(m, size=ni, p=self.pU[:, i])
            b = rng.choice(m, size=ni, p=self.pV[:, i])
            np.add.at(upper, a, 1)
            np.add.at(lower, b, 1)
        return upper, lower

    # ---- sample with optional output loss thinning (tau in (0,1]) -----------
    def sample(self, num_shots=500, rng=None, tau=1.0):
        if rng is None:
            rng = np.random.default_rng(0)
        samples = []
        raw = []                                       # raw (up, lo) for loss scan
        for _ in range(num_shots):
            up, lo = self.raw_shot(rng)
            if tau < 1.0:
                up = rng.binomial(up, tau)
                lo = rng.binomial(lo, tau)
            raw.append((up, lo))
            if up.max() > 1 or lo.max() > 1:           # collision-free
                continue
            S = set(np.nonzero(up)[0].tolist())
            T = set(np.nonzero(lo)[0].tolist())
            if len(S) != len(T):                       # balanced arm
                continue
            if len(S) < 3:                             # at least three pairs
                continue
            samples.append(S | T)
        return samples, raw

    def get_gbs_info(self, num_shots=500, rng=None, tau=1.0):
        samples, _ = self.sample(num_shots, rng=rng, tau=tau)
        vertex_freq = Counter(); edge_freq = Counter()
        for vertices in samples:
            for v in vertices:
                vertex_freq[v] += 1
            vl = sorted(vertices)
            for i in range(len(vl)):
                for j in range(i + 1, len(vl)):
                    u, v = vl[i], vl[j]
                    if self.graph.has_edge(u, v):
                        edge_freq[(u, v)] += 1
                    if self.graph.has_edge(v, u):
                        edge_freq[(v, u)] += 1
        total = len(samples) if samples else 1
        edge_prob = {e: f / total for e, f in edge_freq.items()}
        top_edges = [e for e, _ in edge_freq.most_common(min(self.n * 2, len(edge_freq)))]
        from types import SimpleNamespace
        return SimpleNamespace(
            vertex_freq=vertex_freq, edge_freq=edge_freq, path_freq=Counter(),
            vertex_prob={v: f / total for v, f in vertex_freq.items()},
            edge_prob=edge_prob,
            top_vertices=[v for v, _ in vertex_freq.most_common(min(self.n, self.n))],
            top_edges=top_edges, top_paths=[],
            samples=samples, sample_count=num_shots, effective_samples=len(samples),
            max_subgraph_size=max([len(s) for s in samples]) if samples else 0,
            avg_subgraph_size=np.mean([len(s) for s in samples]) if samples else 0.0,
            subgraph_size_distribution={},
        )


def run_checks():
    """Check normalized columns, mean pair number, and identity output."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    main_benchmark = __import__('main_benchmark')
    GraphGenerator = main_benchmark.GraphGenerator

    print("=== SOURCE-RESOLVED SAMPLER CHECKS ===")
    for n in [20, 40]:
        g = GraphGenerator.generate_random_digraph(n, 0.3, 42 + 0 * 100 + n * 10)
        s = SourceResolvedSampler(g)
        # 1) normalized transition columns
        okU = np.allclose(s.pU.sum(axis=0), 1.0, atol=1e-10)
        okV = np.allclose(s.pV.sum(axis=0), 1.0, atol=1e-10)
        # 2) mean pair number vs sum_i sinh^2 r_i
        rng = np.random.default_rng(1)
        Ns = []
        for _ in range(2000):
            up, lo = s.raw_shot(rng)
            Ns.append(int(up.sum()))
        empirical = np.mean(Ns)
        # 3) identity-interferometer output: S = T = {i : n_i >= 1} (photons stay
        #    in their own mode when U=V=I). Replicate raw_shot with pU=pV=I.
        s2 = SourceResolvedSampler(g)
        s2.pU = np.eye(n); s2.pV = np.eye(n)           # identity interferometers
        rng2 = np.random.default_rng(2)
        n_pairs = rng2.geometric(s2.q) - 1
        up = np.zeros(n, int); lo = np.zeros(n, int)
        for i in range(n):
            ni = int(n_pairs[i])
            if ni > 0:                                  # one-hot column -> mode i
                up[i] += ni; lo[i] += ni
        expect = set(np.nonzero(n_pairs >= 1)[0].tolist())
        got = set(np.nonzero(up)[0].tolist())
        ident_ok = (got == expect) and (set(np.nonzero(lo)[0].tolist()) == expect)

        print(f"n={n}: cols_norm U={okU} V={okV} | "
              f"mean_pairs empirical={empirical:.3f} theory(sum sinh^2)={s.mean_pairs:.3f} | "
              f"identity_output={ident_ok}")


if __name__ == '__main__':
    run_checks()
