# Sourced by the XCIST HPC scripts: where the isolated XCIST env lives on the cluster.
# CTSYNTH_XCIST_ENV overrides; the default is the project's approved directory under $HOME
# (the Research OS workspace_root of the cluster profile), never a path with a user name in it.
CTSYNTH_XCIST_ENV="${CTSYNTH_XCIST_ENV:-$HOME/research-os/projects/CTsynth/env}"
