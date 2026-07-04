import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()
    links = corpus[page]

    if not links:
        links = set(corpus.keys())

    n_pages = len(corpus)
    n_links = len(links)

    for p in corpus:
        distribution[p] = (1 - damping_factor) / n_pages

    for link in links:
        distribution[link] += damping_factor / n_links

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {page: 0 for page in corpus}

    sample = random.choice(list(corpus.keys()))
    counts[sample] += 1

    for _ in range(n - 1):
        model = transition_model(corpus, sample, damping_factor)
        sample = random.choices(
            list(model.keys()), weights=list(model.values()), k=1
        )[0]
        counts[sample] += 1

    return {page: count / n for page, count in counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n_pages = len(corpus)
    ranks = {page: 1 / n_pages for page in corpus}

    linkers = {page: set() for page in corpus}
    for page, links in corpus.items():
        targets = links if links else set(corpus.keys())
        for target in targets:
            linkers[target].add(page)

    num_links = {
        page: len(links) if links else n_pages
        for page, links in corpus.items()
    }

    while True:
        new_ranks = dict()
        for page in corpus:
            total = sum(
                ranks[i] / num_links[i] for i in linkers[page]
            )
            new_ranks[page] = (1 - damping_factor) / n_pages + damping_factor * total

        if all(abs(new_ranks[page] - ranks[page]) < 0.001 for page in ranks):
            ranks = new_ranks
            break

        ranks = new_ranks

    return ranks


if __name__ == "__main__":
    main()