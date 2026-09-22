# LLM4Rec homework

## Homework 1 — Random Lunch Menu Generator

Open [hw1/index.html](hw1/index.html) in a browser. The application is a single,
self-contained HTML file; no installation, build, API key, or network connection
is required. All lunches begin equally weighted; rejecting the displayed result
halves its weight before the next recommendation.

- [Homework README](hw1/README.md)
- [Bug report](hw1/BUG_REPORT.md)
- [Project description](hw1/prompt.md)

The application is in `hw1/`. Once GitHub Pages is enabled from the `master`
branch and repository root, open
<https://noriginalno.github.io/llm4rec/hw1/>.

Adapted from [dryjins/RecSys-LLMs/week1](https://github.com/dryjins/RecSys-LLMs/tree/7b4e5a35e15dda8d87e73636e0a6fb178d203534/week1).
The original author's [MIT license](LICENSE) is included.

## Homework 2 — Content-Based Movie Recommender

Explore [the movie recommendation service](hw2/index.html): select several liked
movies and compare an active movie's Top-5 genre matches with recommendations
from your averaged preference profile. Both use cosine similarity; rating counts
are displayed but never used to rank.

- [Setup, sources, and test commands](hw2/README.md)
- [Analysis: normalization, long-tail discovery, and repetition](hw2/ANALYSIS.md)
- [Reproducible experiment results](hw2/results/comparison.md)

With Python 3 and Git installed, run this from anywhere inside the local clone
(including `hw2/`):

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory "$(git rev-parse --show-toplevel)"
```

Open <http://127.0.0.1:8000/hw2/>. Stop any existing server on port 8000 with Ctrl+C
before restarting. The explicit `--directory` always serves the repository root.
If you previously started Python without `--directory` from inside `hw2/`, use
<http://127.0.0.1:8000/> for that server instead; `/hw2/` will return 404.
This project uses local HTTP-served MovieLens data; no backend, API key, or build
step is required.

With GitHub Pages enabled from `master` and the repository root, the service URL is
<https://noriginalno.github.io/llm4rec/hw2/>.
