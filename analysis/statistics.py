import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.stats.multitest as mt
from scipy import stats
import ccobra

def is_nvc_resp(elem):
    return elem == "NVC"

# Syllogisms of the test phase
test_syllogs = "EE2;EO3;II4;IO3;OA2;OE1;OI3;OO1;AA4;AE2;AO3;EA1;EI1;IA4;IE4;OA3".split(";")

# Determine which test syllogisms were valid and which were invalid
valid_test = [x for x in test_syllogs if x in ccobra.syllogistic.VALID_SYLLOGISMS]
invalid_test = [x for x in test_syllogs if x in ccobra.syllogistic.INVALID_SYLLOGISMS]

print("General information")
print("===================")
print("Test phase syllogisms: ", ", ".join(test_syllogs))
print("Valid: ", ", ".join(valid_test))
print("Invalid: ", ", ".join(invalid_test))
print()

# Load the dataset
df = pd.read_csv("../data/Brand2021/Brand2021_FeedbackEffect.csv")

# Add easy retrievable information for NVC
df["is_nvc"] = df["enc_response"].apply(is_nvc_resp)

# Split into control and feedback
control_df = df[df["condition"] == "control"]
feedback_df = df[df["condition"] == "feedback"]

# Get control and feedback results for test phase
control_test = control_df[control_df["enc_task"].isin(test_syllogs)]
feedback_test = feedback_df[feedback_df["enc_task"].isin(test_syllogs)]

print("Descriptives")
print("============")

corr_control = control_test.groupby("id").agg("mean")["correct"].values
corr_feedback = feedback_test.groupby("id").agg("mean")["correct"].values
print("Correctness control (test phase)  : ", np.mean(corr_control), np.std(corr_control))
print("Correctness feedback (test phase) : ", np.mean(corr_feedback), np.std(corr_feedback))
print()

corr_control = control_df.groupby("id").agg("mean")["correct"].values
corr_feedback = feedback_df.groupby("id").agg("mean")["correct"].values
print("Correctness control (total)  : ", np.mean(corr_control), np.std(corr_control))
print("Correctness feedback (total) : ", np.mean(corr_feedback), np.std(corr_feedback))
print()

nvc_control = control_test.groupby("id").agg("mean")["is_nvc"].values
nvc_feedback = feedback_test.groupby("id").agg("mean")["is_nvc"].values
print("NVC control (test phase)  : ", np.mean(nvc_control), np.std(nvc_control))
print("NVC feedback (test phase) : ", np.mean(nvc_feedback), np.std(nvc_feedback))
print()

nvc_control = control_df.groupby("id").agg("mean")["is_nvc"].values
nvc_feedback = feedback_df.groupby("id").agg("mean")["is_nvc"].values
print("NVC control (total)  : ", np.mean(nvc_control), np.std(nvc_control))
print("NVC feedback (total) : ", np.mean(nvc_feedback), np.std(nvc_feedback))
print()
print()


# calculating tests for hypotheses

# Hypothesis PHM: Confidence is lower when not responding with NVC
# Get non-NVC responses for the control group
control_not_nvc = control_test[control_test["enc_response"] != "NVC"]

# Get non-NVC responses for the feedback group
feedback_not_nvc = feedback_test[feedback_test["enc_response"] != "NVC"]

# Calculate the mean confidence per person for both groups
grp_conf_control_not_nvc = control_not_nvc.groupby("id").agg("mean")["confidence"].values
grp_conf_feedback_not_nvc = feedback_not_nvc.groupby("id").agg("mean")["confidence"].values

# Calculate Mann-Whitney-U test
confidence_not_nvc_mwu = ss.mannwhitneyu(grp_conf_control_not_nvc, grp_conf_feedback_not_nvc)

# Hypothesis TransSet: Time for NVC responses is lower
# Hypothesis mReasoner: Time for NVC reponses is higher
# calculate times for both groups
time_control_nvc = control_test[control_test["enc_response"] == "NVC"].groupby("id").agg("mean")["rt"].values
time_feedback_nvc = feedback_test[feedback_test["enc_response"] == "NVC"].groupby("id").agg("mean")["rt"].values

# Calculate Mann-Whitney-U test
time_mwu_nvc = ss.mannwhitneyu(time_control_nvc, time_feedback_nvc)

# Hypothesis TransSet: Time difference between NVC and Non-NVC is lower for the feedback condition
# Calculate time differences for control
time_diff_control = []
for _, person in control_test.groupby("id"):
    nvcs = []
    non_nvcs = []
    for _, row in person.iterrows():
        if row["enc_response"] == "NVC":
            nvcs.append(row["rt"])
        else:
            non_nvcs.append(row["rt"])
    if nvcs and non_nvcs:
        time_diff_control.append(np.median(nvcs) - np.median(non_nvcs))

# Calculate time differences for feedback
time_diff_feedback = []
for _, person in feedback_test.groupby("id"):
    nvcs = []
    non_nvcs = []
    for _, row in person.iterrows():
        if row["enc_response"] == "NVC":
            nvcs.append(row["rt"])
        else:
            non_nvcs.append(row["rt"])
    if nvcs and non_nvcs:
        time_diff_feedback.append(np.median(nvcs) - np.median(non_nvcs))

# Calculate Mann-Whitney-U test
time_mwu_diff = ss.mannwhitneyu(time_diff_control, time_diff_feedback)

# All p-values have to be in an array for the bonferroni correction with multipletests
p_values = [
    confidence_not_nvc_mwu.pvalue,
    time_mwu_nvc.pvalue,
    time_mwu_diff.pvalue
]

# Perform bonferroni correction
reject, p_corrected, _, _ = mt.multipletests(p_values, method="bonferroni", alpha=0.05)

# Hypotheses for PHM
print("Hypothesis PHM: Confidence for Non-NVC responses is lower in the feedback group")
print("===============================================================================")
print("    Control confidence:")
print("        mean={}, std={}, median={}, mad={}".format(
    np.mean(grp_conf_control_not_nvc),
    np.std(grp_conf_control_not_nvc),
    np.median(grp_conf_control_not_nvc),
    stats.median_abs_deviation(grp_conf_control_not_nvc)))
print()

print("    Feedback confidence:")
print("        mean={}, std={}, median={}, mad={}".format(
    np.mean(grp_conf_feedback_not_nvc),
    np.std(grp_conf_feedback_not_nvc),
    np.median(grp_conf_feedback_not_nvc),
    stats.median_abs_deviation(grp_conf_feedback_not_nvc)))
print()

print("    p={} (Bonferroni corrected) U={}, reject: {}".format(p_corrected[0], confidence_not_nvc_mwu.statistic, reject[0]))
print()
print()

# Hypotheses for TransSet, mReasoner
print("Hypothesis mReasoner/TransSet: The time for NVC responses is higher/lower in the feedback group")
print("===============================================================================================")
# Calculate descriptives
print("Control Time for NVC:")
print("    mean={}, std={}, median={}, mad={}".format(
    np.mean(time_control_nvc),
    np.std(time_control_nvc),
    np.median(time_control_nvc),
    stats.median_abs_deviation(time_control_nvc)))
print()

print("Feedback Time for NVC:")
print("    mean={}, std={}, median={}, mad={}".format(
    np.mean(time_control_nvc),
    np.std(time_control_nvc),
    np.median(time_control_nvc),
    stats.median_abs_deviation(time_control_nvc)))
print()

print("    p={} (Bonferroni corrected) U={}, reject: {}".format(p_corrected[1], time_mwu_nvc.statistic, reject[1]))
print()
print()

# Hypotheses for TransSet
print("Hypothesis TransSet: The time gain between NVC and Non-NVC responses is higher for the feedback group")
print("=====================================================================================================")
print("Control Time difference:")
print("    mean={}, std={}, median={}, mad={}".format(
    np.mean(time_diff_control),
    np.std(time_diff_control),
    np.median(time_diff_control),
    stats.median_abs_deviation(time_diff_control)))
print()

print("Feedback Time difference:")
print("    mean={}, std={}, median={}, mad={}".format(
    np.mean(time_diff_feedback),
    np.std(time_diff_feedback),
    np.median(time_diff_feedback),
    stats.median_abs_deviation(time_diff_feedback)))
print()
print("    p={} (Bonferroni corrected) U={}, reject: {}".format(p_corrected[2], time_mwu_diff.statistic, reject[2]))

