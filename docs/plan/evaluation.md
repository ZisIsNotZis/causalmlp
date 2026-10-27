# Evaluation claim repair
Authored work record; owner is the next evaluator of the archived prefix-MLP experiment. This records an issue, not authority to run training or revive the project.
## Finding
Static inspection at e4046d7b3b158911e5e59a38a26390d7917c955b shows train.py::evaluate_causal_mlp calling causal_mlp.py::causal_accuracy, which computes mean exp(-cross_entropy), while train.py::evaluate_regular_mlp computes top-1 classification accuracy. Those outputs are not the same metric, so their direct subtraction cannot establish an accuracy gap.
README.md's 0.755 versus 0.889 comparison is not established as an apples-to-apples accuracy result unless results.json has a different, documented producer; that producer/run binding is currently unknown. This finding is source-inspected, not reproduced by running the experiment.

The current README therefore presents the two numbers only under an explicit metric caveat and does not read them as an accuracy gap. The caveat qualifies the claim; it does not repair it — the producer binding and the regression fixture below are still open.
## Required action
Trace results.json to its producer, code revision, data, seed, configuration and metric definitions before retaining a comparison; if the inspected producer is the source, compare both models on the same held-out examples using top-1 accuracy and report the probability/loss statistic separately under its own name.
Retain a faithful regression fixture whose expected top-1 accuracy and mean assigned probability deliberately differ; acceptance requires that mislabeled or mixed metrics fail visibly. A rerun, if separately authorized, must bind results to the complete run inputs and revision, with independent metrics and uncertainty appropriate to the comparison.
Until that evidence exists, withdraw or explicitly qualify the numerical accuracy-gap claim in README.md/report.md rather than presenting this work item as a verified fix. No code, metric, result file or experimental outcome was changed by recording this issue.
## Retention question
The owner questions whether the experiment retains value relative to current alternatives; that is a separate unresolved curation decision, not a reason to launch new experiments automatically.
An earlier README draft claimed February 2026 authorship; the current README makes no date claim, and available Git history begins with a migration commit and does not independently establish the original code's date, so pre-LLM authorship is unknown. Age alone does not establish worth: retention needs a distinctive mechanism, trustworthy result, reusable implementation or another owner-valued contribution.
