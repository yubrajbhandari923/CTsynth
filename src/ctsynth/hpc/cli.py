"""`ctsynth` command line: prepare | verify | export | import | report (stdlib argparse).

Exit codes: 0 ok; 1 validation failure or refusal (reasons on stderr, one per line);
2 usage error (argparse). No subcommand opens a network connection.
"""


def main(argv: list[str] | None = None) -> int:
    """Subcommands (all paths explicit, nothing defaults to a cluster):
      prepare --repo R --config C --out-root D --entry ARGV... [--require-output F]...
              [--data F]... [--checkpoint PATH[:external]]... [--hw KEY=VAL]...
              [--env-setup LINE]...      -> prints the bundle dir
      verify PATH [--repo R] [--data-root D]
      export --bundle B --out O --exit-code N --started T --ended T [--slurm-job J]
      import RESULT --repo R --runs-root D [--research-bin P] [--research-root P] [--label L]
              -> prints run id and destination
      report RUN_ID --runs-root D        -> prints markdown, writes <RUN_ID>.report.md
    `--entry` is last and takes the remainder (argparse.REMAINDER)."""
    raise NotImplementedError
